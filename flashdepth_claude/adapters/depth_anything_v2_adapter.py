"""
Adapter for Depth-Anything-V2 (Metric Depth)

Reference: refer_test/Depth-Anything-V2/metric_depth/
"""

import sys
from pathlib import Path
import torch
import numpy as np
from .base_adapter import MethodAdapter


class DepthAnythingV2Adapter(MethodAdapter):
    """Adapter for Depth-Anything-V2 with metric depth head"""

    def __init__(self, indoor=False, encoder='vitl'):
        super().__init__()
        self.indoor = indoor
        self.encoder = encoder

        # Add DA-V2 metric_depth path to sys.path
        dav2_path = Path(__file__).parent.parent / 'refer_test' / 'Depth-Anything-V2' / 'metric_depth'
        if str(dav2_path) not in sys.path:
            sys.path.insert(0, str(dav2_path))

    def load_model(self, checkpoint_path=None):
        """Load Depth-Anything-V2 model"""
        from depth_anything_v2.dpt import DepthAnythingV2

        # Model configuration for different encoders
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        # Choose checkpoint and max_depth based on indoor/outdoor and encoder
        if checkpoint_path is None:
            base_path = Path(__file__).parent.parent / 'refer_test' / 'Depth-Anything-V2' / 'checkpoints'
            if self.indoor:
                checkpoint_path = str(base_path / f'depth_anything_v2_metric_hypersim_{self.encoder}.pth')
                max_depth = 10.0  # Indoor: 10 meters
            else:
                checkpoint_path = str(base_path / f'depth_anything_v2_metric_vkitti_{self.encoder}.pth')
                max_depth = 80.0  # Outdoor: 80 meters
        else:
            # If checkpoint is explicitly provided, use default max_depth
            max_depth = 10.0 if self.indoor else 80.0

        # Create model with specified encoder
        depth_anything = DepthAnythingV2(
            **model_configs[self.encoder],
            max_depth=max_depth
        )

        # Load checkpoint
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        depth_anything.load_state_dict(state_dict)

        self.model = depth_anything
        self.input_size = 518  # Default input size
        return self.model

    def inference(self, image, intrinsics=None):
        """
        Run Depth-Anything-V2 inference

        Args:
            image: torch.Tensor [1, 3, H, W] - Input image (0-1 normalized, RGB)
            intrinsics: Optional camera intrinsics (not used)

        Returns:
            depth: torch.Tensor [1, H, W] - Metric depth in meters
        """
        # Optimize: Convert to uint8 on GPU, then transfer to CPU
        # For large images (e.g., ETH3D 6205x4135), this is much faster
        image_uint8 = (image[0] * 255.0).to(torch.uint8)  # [3, H, W] on GPU

        # Transfer to CPU (uint8 is 4x smaller than float32)
        image_np = image_uint8.cpu().numpy()  # [3, H, W]
        image_np = image_np.transpose(1, 2, 0)  # [H, W, 3]
        image_np = image_np[:, :, ::-1]  # RGB -> BGR

        # Run inference (returns numpy array in meters)
        with torch.no_grad():
            depth_np = self.model.infer_image(image_np, self.input_size)  # [H, W]

        # Record processing resolution on first inference
        if self.processing_resolution is None:
            self.processing_resolution = (self.input_size, self.input_size)
            print(f"[DepthAnythingV2] Processing resolution: {self.input_size}×{self.input_size}")

        # Convert back to torch tensor
        depth = torch.from_numpy(depth_np).unsqueeze(0)  # [1, H, W]

        if self.device is not None:
            depth = depth.to(self.device)

        return depth

    def get_required_env(self):
        return "depthanythingv2"
