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

        # Check for local checkpoint first
        if checkpoint_path is None:
            repo_path = Path(__file__).parent.parent / 'refer_test'
            local_ckpt_paths = [
                repo_path / 'configs' / 'depthpro' / 'depth_pro.pt',
                repo_path / 'ml-depth-pro' / 'checkpoints' / 'depth_pro.pt',
            ]

            for ckpt_path in local_ckpt_paths:
                if ckpt_path.exists():
                    checkpoint_path = str(ckpt_path)
                    break

        if checkpoint_path:
            print(f"Loading DepthPro model from local checkpoint...")
            print(f"Using local checkpoint: {checkpoint_path}")
        else:
            print(f"Loading DepthPro model...")
            print(f"Local checkpoint not found. Will download from HuggingFace...")

        # Create model and transforms
        # Model will automatically resize inputs to 1536x1536
        self.model, self.transform = create_model_and_transforms(
            device=torch.device("cpu"),  # Will be moved to GPU later
            precision=torch.half,
            checkpoint_path=checkpoint_path if checkpoint_path else None,
        )
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
        image_np = image[0].cpu().numpy()  # [3, H, W]
        image_np = image_np.transpose(1, 2, 0)  # [H, W, 3]
        image_np = (image_np * 255).astype(np.uint8)  # 0-1 -> 0-255
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

        # Run inference
        # DepthPro automatically resizes to 1536x1536 and back
        with torch.no_grad():
            prediction = self.model.infer(transformed, f_px=f_px)

        # Extract depth (already in meters)
        depth = prediction["depth"]  # [H, W]

        # Add batch dimension
        depth = depth.unsqueeze(0)  # [1, H, W]

        return depth

    def get_required_env(self):
        return "depthpro"
