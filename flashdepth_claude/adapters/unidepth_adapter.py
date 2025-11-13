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
        from unidepth.models import UniDepthV1, UniDepthV2

        model_name = f"unidepth-{self.version}-vit{self.variant}14"

        # Check for local checkpoint first
        repo_path = Path(__file__).parent.parent / 'refer_test'
        local_ckpt_paths = [
            repo_path / 'configs' / 'unidepth' / f'unidepth{self.version}_pytorch_model.bin',
            repo_path / 'UniDepth' / 'checkpoints' / f'unidepth{self.version}_pytorch_model.bin',
        ]

        local_ckpt = None
        for ckpt_path in local_ckpt_paths:
            if ckpt_path.exists():
                local_ckpt = ckpt_path
                break

        if local_ckpt:
            # Load from local checkpoint
            print(f"Loading UniDepth model: {model_name} from local checkpoint")
            print(f"Using local checkpoint: {local_ckpt}")

            if self.version == 'v2':
                self.model = UniDepthV2.from_pretrained(str(local_ckpt.parent))
                # Set interpolation mode for better quality
                self.model.interpolation_mode = "bilinear"
            else:
                self.model = UniDepthV1.from_pretrained(str(local_ckpt.parent))
        else:
            # Load from HuggingFace
            print(f"Loading UniDepth model: {model_name} from lpiccinelli/{model_name}")
            print(f"Local checkpoint not found. Will download from HuggingFace...")
            print(f"To avoid downloading, place checkpoint in: {local_ckpt_paths[0]}")

            if self.version == 'v2':
                self.model = UniDepthV2.from_pretrained(f"lpiccinelli/{model_name}")
                # Set interpolation mode for better quality
                self.model.interpolation_mode = "bilinear"
            else:
                self.model = UniDepthV1.from_pretrained(f"lpiccinelli/{model_name}")

        self.model.eval()

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

        # For V1, camera should be K matrix directly
        from unidepth.models import UniDepthV1
        if isinstance(self.model, UniDepthV1):
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

        # Add batch dimension
        depth = depth.unsqueeze(0)  # [1, H, W]

        return depth

    def get_required_env(self):
        return "unidepth"
