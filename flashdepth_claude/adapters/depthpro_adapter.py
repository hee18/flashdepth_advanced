"""
Adapter for DepthPro (Apple ML)
Reference: refer_test/ml-depth-pro/
"""

import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from .base_adapter import MethodAdapter


class DepthProAdapter(MethodAdapter):
    """Adapter for DepthPro (Apple ML)"""

    def __init__(self):
        super().__init__()
        dp_path = Path(__file__).parent.parent / 'refer_test' / 'ml-depth-pro' / 'src'
        if str(dp_path) not in sys.path:
            sys.path.insert(0, str(dp_path))

    def load_model(self, checkpoint_path=None):
        """
        Load DepthPro model using create_model_and_transforms

        DepthPro automatically resizes to 1536x1536 in infer() method
        """
        from depth_pro import create_model_and_transforms
        from depth_pro.depth_pro import DepthProConfig

        # Check for local checkpoint first
        if checkpoint_path is None:
            repo_path = Path(__file__).parent.parent / 'refer_test'
            local_ckpt_paths = [
                repo_path / 'ml-depth-pro' / 'checkpoints' / 'depth_pro.pt',
            ]

            for ckpt_path in local_ckpt_paths:
                if ckpt_path.exists():
                    checkpoint_path = str(ckpt_path)
                    break

        if checkpoint_path:
            print(f"Loading DepthPro model from local checkpoint...")
            print(f"Using local checkpoint: {checkpoint_path}")
            # Create config with custom checkpoint path
            config = DepthProConfig(
                patch_encoder_preset="dinov2l16_384",
                image_encoder_preset="dinov2l16_384",
                checkpoint_uri=checkpoint_path,
                decoder_features=256,
                use_fov_head=True,
                fov_encoder_preset="dinov2l16_384",
            )
        else:
            print(f"Loading DepthPro model...")
            print(f"Local checkpoint not found. Will download from HuggingFace...")
            # Use default config (downloads from HF)
            config = None

        # Create model and transforms
        # Model will automatically resize inputs to 1536x1536
        # Suppress verbose model architecture output
        import warnings
        import logging
        import sys
        import io

        # Temporarily redirect stdout to suppress model architecture print
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        # Temporarily suppress warnings and logging
        old_warning_filters = warnings.filters[:]
        warnings.filterwarnings('ignore')
        logging.disable(logging.CRITICAL)

        try:
            if config:
                self.model, self.transform = create_model_and_transforms(
                    config=config,
                    device=torch.device("cpu"),  # Will be moved to GPU later
                    precision=torch.half,
                )
            else:
                self.model, self.transform = create_model_and_transforms(
                    device=torch.device("cpu"),
                    precision=torch.half,
                )
        finally:
            # Restore stdout, warnings, and logging
            sys.stdout = old_stdout
            warnings.filters[:] = old_warning_filters
            logging.disable(logging.NOTSET)

        self.model.eval()

        print(f"DepthPro model loaded successfully")
        return self.model

    def inference(self, image, intrinsics=None):
        """
        Run DepthPro inference

        Args:
            image: torch.Tensor [1, 3, H, W] - Input image (0-1 normalized, RGB)
            intrinsics: Optional torch.Tensor [1, 4] - Camera intrinsics [fx, fy, cx, cy]
                       If provided, fx will be used for metric depth estimation

        Returns:
            depth: torch.Tensor [1, H, W] - Metric depth in meters
        """
        # Convert torch tensor to PIL Image for DepthPro
        # Input is [1, 3, H, W] in RGB 0-1 range

        # Optimize: Convert to uint8 on GPU first (faster for large images)
        image_uint8 = (image[0] * 255.0).to(torch.uint8)  # [3, H, W] on GPU

        # Transfer to CPU (uint8 is 4x smaller than float32)
        image_np = image_uint8.cpu().numpy()  # [3, H, W]
        image_np = image_np.transpose(1, 2, 0)  # [H, W, 3]
        image_pil = Image.fromarray(image_np)

        # Apply transform (normalizes to [-1, 1] range)
        transformed = self.transform(image_pil)  # [3, H, W]

        # Move to device
        if self.device is not None:
            transformed = transformed.to(self.device)

        # Extract focal length if intrinsics provided
        f_px = None
        if intrinsics is not None:
            # Handle different intrinsics formats
            if isinstance(intrinsics, torch.Tensor):
                if intrinsics.numel() >= 4:
                    # Full intrinsics - [fx, fy, cx, cy]
                    # Handle both 1D [4] and 2D [1, 4] or [batch, 4]
                    if intrinsics.dim() == 1:
                        # 1D tensor [4]
                        f_px = intrinsics[0].item()
                    else:
                        # 2D tensor [1, 4] or [batch, 4]
                        f_px = intrinsics[0, 0].item()
                elif intrinsics.numel() == 1:
                    # Scalar focal length
                    f_px = intrinsics.item()
                else:
                    raise ValueError(f"Unexpected intrinsics shape: {intrinsics.shape}")
            else:
                # Scalar value
                f_px = float(intrinsics)

            # Convert f_px to tensor (DepthPro expects tensor, not float)
            f_px = torch.tensor(f_px, dtype=torch.float32)
            if self.device is not None:
                f_px = f_px.to(self.device)

        # Run inference
        # DepthPro automatically resizes to 1536x1536 and back
        with torch.no_grad():
            prediction = self.model.infer(transformed, f_px=f_px)

        # Extract depth (already in meters)
        depth = prediction["depth"]  # [H, W]

        # Record processing resolution on first inference
        if self.processing_resolution is None:
            self.processing_resolution = (1536, 1536)
            print(f"[DepthPro] Processing resolution: 1536×1536")

        # Add batch dimension
        depth = depth.unsqueeze(0)  # [1, H, W]

        return depth

    def get_required_env(self):
        return "depthpro"
