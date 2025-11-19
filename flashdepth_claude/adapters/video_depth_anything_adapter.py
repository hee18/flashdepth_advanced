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
        self._first_inference = True  # Flag for first inference logging

        # Add Video-Depth-Anything path to sys.path
        vda_path = Path(__file__).parent.parent / 'refer_test' / 'Video-Depth-Anything'
        if str(vda_path) not in sys.path:
            sys.path.insert(0, str(vda_path))

    def load_model(self, checkpoint_path=None):
        """Load Video-Depth-Anything model"""
        # Ensure utils path is available before importing
        vda_path = Path(__file__).parent.parent / 'refer_test' / 'Video-Depth-Anything'
        utils_path = vda_path / 'utils'
        if str(utils_path.parent) not in sys.path:
            sys.path.insert(0, str(utils_path.parent))

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

        print(f"[VDA] Loading checkpoint: {checkpoint_path}")
        print(f"[VDA] Metric mode: {self.metric}")

        # Create model
        encoder_type = 'vitl'  # Use ViT-L by default
        video_depth_anything = VideoDepthAnything(
            **model_configs[encoder_type],
            metric=self.metric
        )

        print(f"[VDA] Model created with metric={self.metric}")

        # Load checkpoint
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        print(f"[VDA] Checkpoint loaded, keys: {len(state_dict)} parameters")

        # Check if checkpoint has metric head
        has_metric_head = any('metric' in k.lower() for k in state_dict.keys())
        print(f"[VDA] Checkpoint has metric head: {has_metric_head}")

        video_depth_anything.load_state_dict(state_dict, strict=True)
        print(f"[VDA] Checkpoint loaded successfully!")

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
        Run Video-Depth-Anything inference on video sequence using official infer_video_depth()

        Args:
            image: torch.Tensor [1, T, 3, H, W] - Input video sequence (0-1 normalized, RGB)
            intrinsics: Optional camera intrinsics (not used)

        Returns:
            depth: torch.Tensor [1, T, H, W] - Predicted depth sequence
                   Metric mode: depth in meters
                   Non-metric mode: relative depth (0-1 normalized)
        """
        # Handle both single frame [1, 3, H, W] and sequence [1, T, 3, H, W]
        if image.ndim == 4:
            # Single frame - add temporal dimension
            image = image.unsqueeze(1)  # [1, 1, 3, H, W]

        B, T, C, orig_H, orig_W = image.shape
        assert B == 1, "Batch size must be 1"

        # Convert torch tensor to numpy array for official VDA code
        # [1, T, 3, H, W] -> [T, H, W, 3]
        image_np = image[0].permute(0, 2, 3, 1).cpu().numpy()  # [T, H, W, 3]

        # VDA expects 0-255 range
        image_np = (image_np * 255.0).astype(np.uint8)

        # Use official infer_video_depth method with sliding window + alignment
        device_str = str(self.device) if self.device is not None else 'cuda'

        # Record processing resolution on first inference (estimated from input)
        if self.processing_resolution is None:
            # Estimate processing resolution from VDA's transform logic
            ratio = max(orig_H, orig_W) / min(orig_H, orig_W)
            input_size = self.input_size
            if ratio > 1.78:
                input_size = int(input_size * 1.777 / ratio)
                input_size = round(input_size / 14) * 14

            # Rough estimate - actual may vary slightly
            if orig_H > orig_W:
                proc_H = input_size
                proc_W = int(orig_W * input_size / orig_H / 14) * 14
            else:
                proc_W = input_size
                proc_H = int(orig_H * input_size / orig_W / 14) * 14

            self.processing_resolution = (proc_H, proc_W)
            print(f"[Video-Depth-Anything] Processing resolution: ~{proc_H}×{proc_W} (adaptive, aspect-ratio preserved)")
            print(f"[VDA] Using official infer_video_depth() with sliding window (INFER_LEN=32, OVERLAP=10)")

        # Call official VDA inference
        depth_np, _ = self.model.infer_video_depth(
            frames=image_np,
            target_fps=30,  # Dummy value, not used for depth prediction
            input_size=self.input_size,
            device=device_str,
            fp32=False  # Use FP16 for speed
        )

        # Debug: Check depth range (first inference only)
        if self._first_inference:
            depth_min = depth_np.min()
            depth_max = depth_np.max()
            depth_mean = depth_np.mean()
            print(f"[VDA Debug] Output depth range: min={depth_min:.3f}, max={depth_max:.3f}, mean={depth_mean:.3f}")
            if self.metric:
                if depth_max < 10:
                    print(f"[VDA WARNING] Metric mode but depth_max={depth_max:.3f} < 10m. This looks like RELATIVE depth!")
                else:
                    print(f"[VDA OK] Depth range looks like metric depth (max={depth_max:.1f}m)")
            self._first_inference = False

        # Convert back to torch tensor
        # [T, H, W] -> [1, T, H, W]
        depth = torch.from_numpy(depth_np).unsqueeze(0)  # [1, T, H, W]

        if self.device is not None:
            depth = depth.to(self.device)

        return depth

    def get_required_env(self):
        return "vda"
