"""
Adapter for UniDepth v1/v2
Reference: refer_test/UniDepth/
"""

import sys
from pathlib import Path
import torch
from .base_adapter import MethodAdapter


class UniDepthAdapter(MethodAdapter):
    """Adapter for UniDepth v1/v2"""

    def __init__(self, version='v2', variant='l'):
        super().__init__()
        self.version = version
        self.variant = variant  # s, b, l (small, base, large)

        ud_path = Path(__file__).parent.parent / 'refer_test' / 'UniDepth'
        if str(ud_path) not in sys.path:
            sys.path.insert(0, str(ud_path))

    def load_model(self, checkpoint_path=None):
        """
        Load UniDepth model from pretrained checkpoint

        UniDepth automatically resizes inputs based on resolution_level and pixels_bounds
        """
        from unidepth.models import UniDepthV1, UniDepthV2, UniDepthV2old

        model_name = f"unidepth-{self.version}-vit{self.variant}14"

        # Check for local checkpoint
        repo_path = Path(__file__).parent.parent / 'refer_test'
        checkpoint_dir = repo_path / 'UniDepth' / 'checkpoints'

        # Check for safetensors or bin file
        local_safetensors = checkpoint_dir / f'unidepth{self.version}_model.safetensors'
        local_bin = checkpoint_dir / f'unidepth{self.version}_pytorch_model.bin'

        if local_safetensors.exists() or local_bin.exists():
            # Load from local checkpoint
            print(f"Loading UniDepth model: {model_name} from local checkpoint")
            print(f"Using checkpoint directory: {checkpoint_dir}")

            # Detect checkpoint version for v2 by checking level_embeds shape
            use_old_v2 = False
            if self.version == 'v2':
                # Load checkpoint to check shape
                import safetensors.torch as st
                checkpoint_file = local_safetensors if local_safetensors.exists() else local_bin
                try:
                    state_dict = st.load_file(str(checkpoint_file))
                    if 'pixel_decoder.level_embeds' in state_dict:
                        level_embeds_shape = state_dict['pixel_decoder.level_embeds'].shape
                        # Old version: [4, 512], New version: [1, 1, 4, 512]
                        if len(level_embeds_shape) == 2:
                            use_old_v2 = True
                            print(f"Detected old V2 checkpoint (level_embeds shape: {level_embeds_shape})")
                        else:
                            print(f"Detected new V2 checkpoint (level_embeds shape: {level_embeds_shape})")
                except Exception as e:
                    print(f"Warning: Could not detect checkpoint version: {e}")
                    print("Defaulting to new V2 format")

            # Choose config based on detected version
            if self.version == 'v2':
                if use_old_v2:
                    config_name = f'config_v2old_vit{self.variant}14.json'
                else:
                    config_name = f'config_{self.version}_vit{self.variant}14.json'
            else:
                config_name = f'config_{self.version}_vit{self.variant}14.json'

            config_path = checkpoint_dir / config_name
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")

            import shutil
            main_config = checkpoint_dir / 'config.json'
            # Always overwrite config.json with the correct version-specific config
            # This ensures the config matches the model version being loaded
            shutil.copy2(config_path, main_config)
            print(f"Copied {config_name} to config.json")

            # Ensure model.safetensors exists (copy from version-specific file)
            model_file = checkpoint_dir / 'model.safetensors'
            if local_safetensors.exists():
                if not model_file.exists() or model_file.stat().st_size != local_safetensors.stat().st_size:
                    shutil.copy2(local_safetensors, model_file)
                    print(f"Copied {local_safetensors.name} to model.safetensors")
            elif local_bin.exists():
                # If only .bin exists, copy it as pytorch_model.bin
                model_bin = checkpoint_dir / 'pytorch_model.bin'
                if not model_bin.exists():
                    shutil.copy2(local_bin, model_bin)
                    print(f"Copied {local_bin.name} to pytorch_model.bin")

            if self.version == 'v2':
                # Use appropriate version based on checkpoint
                if use_old_v2:
                    print("Using UniDepthV2old for old checkpoint")
                    self.model = UniDepthV2old.from_pretrained(str(checkpoint_dir))
                else:
                    print("Using UniDepthV2 for new checkpoint")
                    self.model = UniDepthV2.from_pretrained(str(checkpoint_dir))
                # Don't set resolution_level - use default adaptive bounds
                # Set interpolation mode for better quality
                self.model.interpolation_mode = "bilinear"
            else:
                self.model = UniDepthV1.from_pretrained(str(checkpoint_dir))
        else:
            # Load from HuggingFace
            print(f"Loading UniDepth model: {model_name} from lpiccinelli/{model_name}")
            print(f"Local checkpoint not found. Will download from HuggingFace...")

            if self.version == 'v2':
                # HuggingFace uses the new version
                self.model = UniDepthV2.from_pretrained(f"lpiccinelli/{model_name}")
                # Don't set resolution_level - use default adaptive bounds
                # Set interpolation mode for better quality
                self.model.interpolation_mode = "bilinear"
            else:
                self.model = UniDepthV1.from_pretrained(f"lpiccinelli/{model_name}")

        self.model.eval()

        # Enable FP16 for faster inference (optional, sacrifices minimal accuracy)
        # Uncomment to enable:
        # self.model.half()
        # print("UniDepth running in FP16 mode for faster inference")

        print(f"UniDepth {self.version} ({self.variant}) model loaded successfully")
        return self.model

    def inference(self, image, intrinsics=None):
        """
        Run UniDepth inference

        Args:
            image: torch.Tensor [1, 3, H, W] - Input image (0-1 normalized, RGB, range 0-255)
            intrinsics: Optional torch.Tensor [1, 4] - Camera intrinsics [fx, fy, cx, cy]
                       If not provided, identity intrinsics will be used

        Returns:
            depth: torch.Tensor [1, H, W] - Metric depth in meters
        """
        from unidepth.utils.camera import Pinhole

        # UniDepth expects input in range [0, 255]
        # Convert from [0-1] to [0-255]
        rgb_torch = (image[0] * 255.0).to(torch.uint8)  # [3, H, W]

        # Prepare camera intrinsics
        if intrinsics is not None:
            # Handle both tensor [4] and scalar inputs
            if isinstance(intrinsics, torch.Tensor):
                if intrinsics.numel() == 4:
                    # Full intrinsics [fx, fy, cx, cy]
                    fx, fy, cx, cy = intrinsics
                elif intrinsics.numel() == 1:
                    # Scalar focal length - estimate cx, cy from image center
                    H, W = image.shape[2:]
                    fx = fy = intrinsics.item()
                    cx, cy = W / 2, H / 2
                else:
                    raise ValueError(f"Unexpected intrinsics shape: {intrinsics.shape}")
            else:
                # Scalar value
                H, W = image.shape[2:]
                fx = fy = float(intrinsics)
                cx, cy = W / 2, H / 2

            # Build K matrix
            K = torch.tensor([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=torch.float32).unsqueeze(0)  # [1, 3, 3]
            camera = Pinhole(K=K)
        else:
            # Use identity intrinsics if not provided
            H, W = image.shape[2:]
            K = torch.tensor([
                [W, 0, W/2],
                [0, H, H/2],
                [0, 0, 1]
            ], dtype=torch.float32).unsqueeze(0)
            camera = Pinhole(K=K)

        # For V1 and V2old, camera should be K matrix directly
        from unidepth.models import UniDepthV1, UniDepthV2old
        if isinstance(self.model, (UniDepthV1, UniDepthV2old)):
            camera = camera.K.squeeze(0)

        # Move to device
        if self.device is not None:
            rgb_torch = rgb_torch.to(self.device)
            if isinstance(camera, Pinhole):
                camera = camera.to(self.device)
            else:
                camera = camera.to(self.device)

        # Run inference
        # UniDepth automatically resizes input based on resolution constraints
        with torch.no_grad():
            predictions = self.model.infer(rgb_torch, camera)

        # Extract depth (already in meters, already resized back to original resolution)
        depth = predictions["depth"].squeeze()  # [H, W]

        # Record processing resolution on first inference (from internal features)
        if self.processing_resolution is None:
            # UniDepth processes adaptively, record from depth_features shape
            if "depth_features" in predictions and predictions["depth_features"] is not None:
                feat_shape = predictions["depth_features"].shape[-2:]
                # Both v1 and v2 use 14-pixel patches (with DINOv2), convert to actual resolution
                if self.version == 'v2':
                    actual_h = feat_shape[0] * 14
                    actual_w = feat_shape[1] * 14
                    self.processing_resolution = (actual_h, actual_w)
                    print(f"[UniDepth-v2] Processing resolution: {actual_h}×{actual_w} ({feat_shape[0]}×{feat_shape[1]} patches, 14px/patch)")
                elif self.version == 'v1':
                    # v1 also uses 14-pixel patches with DINOv2 encoder
                    actual_h = feat_shape[0] * 14
                    actual_w = feat_shape[1] * 14
                    self.processing_resolution = (actual_h, actual_w)
                    print(f"[UniDepth-v1] Processing resolution: {actual_h}×{actual_w} ({feat_shape[0]}×{feat_shape[1]} patches, 14px/patch)")
                else:
                    self.processing_resolution = (feat_shape[0], feat_shape[1])
                    print(f"[UniDepth-{self.version}] Processing resolution: {feat_shape[0]}×{feat_shape[1]}")
            else:
                # Fallback: estimate from config
                if self.version == 'v1':
                    # v1 uses fixed resolution from config (462×616)
                    if hasattr(self.model, 'image_shape'):
                        h, w = self.model.image_shape
                        self.processing_resolution = (h, w)
                        print(f"[UniDepth-v1] Processing resolution: {h}×{w} (fixed, config-based)")
                    else:
                        self.processing_resolution = (image.shape[2], image.shape[3])
                        print(f"[UniDepth-v1] Processing resolution: {image.shape[2]}×{image.shape[3]} (original)")
                else:
                    self.processing_resolution = "adaptive (200k-600k pixels)"
                    print(f"[UniDepth-v2] Processing resolution: adaptive (200k-600k pixels)")

        # Add batch dimension
        depth = depth.unsqueeze(0)  # [1, H, W]

        return depth

    def get_required_env(self):
        return "unidepth"
