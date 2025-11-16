"""
Adapter for Video-Depth-Anything

Reference: refer_test/Video-Depth-Anything/
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from .base_adapter import MethodAdapter


class VideoDepthAnythingAdapter(MethodAdapter):
    """Adapter for Video-Depth-Anything with metric/non-metric mode"""

    def __init__(self, metric=False):
        super().__init__()
        self.metric = metric

        # Add Video-Depth-Anything path to sys.path
        vda_path = Path(__file__).parent.parent / 'refer_test' / 'Video-Depth-Anything'
        if str(vda_path) not in sys.path:
            sys.path.insert(0, str(vda_path))

    def load_model(self, checkpoint_path=None):
        """Load Video-Depth-Anything model"""
        from video_depth_anything.video_depth import VideoDepthAnything

        # Model configuration for ViT-L
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }

        # Choose checkpoint based on metric mode
        if checkpoint_path is None:
            base_path = Path(__file__).parent.parent / 'refer_test' / 'Video-Depth-Anything' / 'checkpoints'
            encoder = 'vitl'
            if self.metric:
                checkpoint_path = str(base_path / f'metric_video_depth_anything_{encoder}.pth')
            else:
                checkpoint_path = str(base_path / f'video_depth_anything_{encoder}.pth')

        # Create model
        encoder_type = 'vitl'  # Use ViT-L by default
        video_depth_anything = VideoDepthAnything(
            **model_configs[encoder_type],
            metric=self.metric
        )

        # Load checkpoint
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        video_depth_anything.load_state_dict(state_dict, strict=True)

        self.model = video_depth_anything
        self.input_size = 518
        return self.model

    def preprocess(self, image):
        """
        Preprocess image for Video-Depth-Anything

        Args:
            image: torch.Tensor [1, 3, H, W] - Input image (0-1 normalized, RGB)

        Returns:
            processed: torch.Tensor [1, 1, 3, H, W] - Preprocessed with temporal dim
        """
        from torchvision.transforms import Compose
        from video_depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

        # Get original size
        _, _, H, W = image.shape

        # Adjust input size based on aspect ratio
        ratio = max(H, W) / min(H, W)
        input_size = self.input_size
        if ratio > 1.78:
            input_size = int(input_size * 1.777 / ratio)
            input_size = round(input_size / 14) * 14

        # Define transform
        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        # Optimize: Convert to uint8 on GPU first (4x smaller transfer)
        # For large images (e.g., ETH3D 6205x4135), this is much faster
        image_uint8 = (image[0] * 255.0).to(torch.uint8)  # [3, H, W] on GPU

        # Transfer to CPU (uint8 is 4x smaller than float32)
        image_np = image_uint8.cpu().numpy()  # [3, H, W]
        image_np = image_np.transpose(1, 2, 0)  # [H, W, 3]

        # Convert back to float for transform (expects 0-255 range)
        image_np = image_np.astype(np.float32)

        # Apply transform
        transformed = transform({'image': image_np})['image']  # [3, H', W']

        # Convert back to torch and add batch + temporal dims
        processed = torch.from_numpy(transformed).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, H', W']

        return processed

    def inference(self, image, intrinsics=None):
        """
        Run Video-Depth-Anything inference on single frame

        Args:
            image: torch.Tensor [1, 3, H, W] - Input image (0-1 normalized, RGB)
            intrinsics: Optional camera intrinsics (not used)

        Returns:
            depth: torch.Tensor [1, H, W] - Predicted depth
                   Metric mode: depth in meters
                   Non-metric mode: relative depth (0-1 normalized)
        """
        orig_H, orig_W = image.shape[2:]

        # Preprocess image (adds temporal dimension)
        processed = self.preprocess(image)  # [1, 1, 3, H', W']

        if self.device is not None:
            processed = processed.to(self.device)

        # Run inference
        with torch.no_grad():
            with torch.autocast(device_type=str(self.device).split(':')[0] if self.device else 'cpu',
                               dtype=torch.float16, enabled=True):
                depth = self.model.forward(processed)  # [1, 1, H', W']

        # Remove temporal dimension
        depth = depth.squeeze(1)  # [1, H', W']

        # Resize back to original size
        depth = F.interpolate(
            depth.unsqueeze(1),
            size=(orig_H, orig_W),
            mode='bilinear',
            align_corners=True
        ).squeeze(1)  # [1, H, W]

        return depth

    def get_required_env(self):
        return "vda"
