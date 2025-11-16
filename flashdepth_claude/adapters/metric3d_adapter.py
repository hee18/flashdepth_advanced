"""
Adapter for Metric3D v1/v2
Reference: refer_test/Metric3D/
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from .base_adapter import MethodAdapter

class Metric3DAdapter(MethodAdapter):
    """Adapter for Metric3D v1/v2"""

    def __init__(self, version='v2', variant=None):
        super().__init__()
        self.version = version

        # Auto-select variant based on version if not specified
        if variant is None:
            if version == 'v1':
                self.variant = 'convnext_large'  # v1 uses ConvNeXt
            else:
                self.variant = 'vit_large'  # v2 uses ViT
        else:
            self.variant = variant  # convnext_tiny, convnext_large, vit_small, vit_large, vit_giant2

        m3d_path = Path(__file__).parent.parent / 'refer_test' / 'Metric3D'
        if str(m3d_path) not in sys.path:
            sys.path.insert(0, str(m3d_path))

    def load_model(self, checkpoint_path=None):
        """Load Metric3D model using torch.hub"""
        repo_path = Path(__file__).parent.parent / 'refer_test' / 'Metric3D'

        # Map version+variant to model name and checkpoint filename
        variant_map = {
            'convnext_tiny': {
                'model_name': 'metric3d_convnext_tiny',
                'ckpt_name': 'convtiny_hourglass_v1.pth'
            },
            'convnext_large': {
                'model_name': 'metric3d_convnext_large',
                'ckpt_name': 'convlarge_hourglass_0.3_150_step750k_v1.1.pth'
            },
            'vit_small': {
                'model_name': 'metric3d_vit_small',
                'ckpt_name': 'metric_depth_vit_small_800k.pth'
            },
            'vit_large': {
                'model_name': 'metric3d_vit_large',
                'ckpt_name': 'metric_depth_vit_large_800k.pth'
            },
            'vit_giant2': {
                'model_name': 'metric3d_vit_giant2',
                'ckpt_name': 'metric_depth_vit_giant2_800k.pth'
            },
        }

        variant_info = variant_map.get(self.variant, variant_map['vit_small'])
        model_name = variant_info['model_name']
        ckpt_name = variant_info['ckpt_name']

        # Check for local checkpoint in multiple locations
        local_ckpt_paths = [
            repo_path / 'checkpoints' / ckpt_name,
            repo_path / 'weight' / ckpt_name,
            repo_path / ckpt_name,
        ]

        local_ckpt = None
        for ckpt_path in local_ckpt_paths:
            if ckpt_path.exists():
                local_ckpt = ckpt_path
                break

        if local_ckpt:
            # Load model without pretrained weights, then load manually
            print(f"Loading Metric3D model: {model_name} from {repo_path}")
            print(f"Using local checkpoint: {local_ckpt}")
            self.model = torch.hub.load(str(repo_path), model_name, source='local', pretrain=False)

            # Load checkpoint manually
            checkpoint = torch.load(local_ckpt, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded checkpoint from {local_ckpt}")
        else:
            # Auto-download from HuggingFace (original behavior)
            print(f"Loading Metric3D model: {model_name} from {repo_path}")
            print(f"Local checkpoint not found. Will download from HuggingFace...")
            print(f"To avoid downloading, place checkpoint in: {repo_path / 'weight' / ckpt_name}")
            self.model = torch.hub.load(str(repo_path), model_name, source='local', pretrain=True)

        # Set input size based on variant
        if 'convnext' in self.variant:
            self.input_size = (544, 1216)
        else:  # ViT variants
            self.input_size = (616, 1064)

        print(f"Metric3D {self.variant} loaded successfully (input_size: {self.input_size})")
        return self.model

    def inference(self, image, intrinsics=None):
        """
        Run Metric3D inference

        Args:
            image: torch.Tensor [1, 3, H, W] - Input RGB image (0-1 normalized)
            intrinsics: torch.Tensor [1, 4] - Camera intrinsics [fx, fy, cx, cy]

        Returns:
            depth: torch.Tensor [1, H, W] - Metric depth in meters
        """
        # Get original size
        H_orig, W_orig = image.shape[2:]

        # Resize keeping aspect ratio (do this on GPU first to avoid large CPU transfer)
        scale = min(self.input_size[0] / H_orig, self.input_size[1] / W_orig)
        new_h, new_w = int(H_orig * scale), int(W_orig * scale)

        # Resize on GPU using PyTorch (much faster than CPU cv2.resize for large images)
        image_resized = F.interpolate(
            image,  # [1, 3, H, W]
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        )  # [1, 3, new_h, new_w]

        # Now convert to numpy (much smaller image)
        rgb_origin = image_resized[0].permute(1, 2, 0).cpu().numpy()  # [new_h, new_w, 3]
        rgb = (rgb_origin * 255).astype(np.uint8)

        # Scale intrinsics
        if intrinsics is not None:
            # Handle different intrinsics formats
            if isinstance(intrinsics, torch.Tensor):
                if intrinsics.numel() >= 4:
                    # Full intrinsics - [fx, fy, cx, cy]
                    # Handle both 1D [4] and 2D [1, 4] or [batch, 4]
                    if intrinsics.dim() == 1:
                        # 1D tensor [4]
                        fx = intrinsics[0].item() * scale
                    else:
                        # 2D tensor [1, 4] or [batch, 4]
                        fx = intrinsics[0, 0].item() * scale
                elif intrinsics.numel() == 1:
                    # Scalar focal length
                    fx = intrinsics.item() * scale
                else:
                    raise ValueError(f"Unexpected intrinsics shape: {intrinsics.shape}")
            else:
                # Scalar value
                fx = float(intrinsics) * scale
        else:
            # Default focal length if not provided
            fx = 1000.0 * scale

        # Convert back to torch tensor on GPU for padding and normalization
        # (faster than doing it in numpy on CPU)
        rgb_torch = torch.from_numpy(rgb.transpose((2, 0, 1))).float().to(self.device)  # [3, H, W]
        rgb_torch = rgb_torch.unsqueeze(0)  # [1, 3, H, W]

        # Padding to input_size (on GPU)
        h, w = rgb_torch.shape[2:]
        pad_h = self.input_size[0] - h
        pad_w = self.input_size[1] - w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

        # Use PyTorch padding with constant value
        padding_value = torch.tensor([123.675, 116.28, 103.53], device=self.device).view(1, 3, 1, 1)
        rgb_torch = F.pad(rgb_torch, (pad_w_half, pad_w - pad_w_half, pad_h_half, pad_h - pad_h_half),
                         mode='constant', value=0)
        # Apply padding color
        mask = torch.ones_like(rgb_torch)
        mask = F.pad(torch.zeros(1, 1, h, w, device=self.device),
                    (pad_w_half, pad_w - pad_w_half, pad_h_half, pad_h - pad_h_half),
                    mode='constant', value=1)
        rgb_torch = rgb_torch + mask * padding_value

        # Normalize (on GPU)
        mean = torch.tensor([123.675, 116.28, 103.53], device=self.device).float().view(1, 3, 1, 1)
        std = torch.tensor([58.395, 57.12, 57.375], device=self.device).float().view(1, 3, 1, 1)
        rgb = torch.div((rgb_torch - mean), std)

        # Inference
        with torch.no_grad():
            pred_depth, confidence, output_dict = self.model.inference({'input': rgb})

        # Un-pad
        pred_depth = pred_depth.squeeze()
        pred_depth = pred_depth[pad_info[0]:pred_depth.shape[0] - pad_info[1],
                                pad_info[2]:pred_depth.shape[1] - pad_info[3]]

        # Upsample to original size
        pred_depth = F.interpolate(pred_depth[None, None, :, :], (H_orig, W_orig),
                                   mode='bilinear', align_corners=False).squeeze()

        # De-canonical transform: convert to metric depth
        canonical_to_real_scale = fx / 1000.0  # 1000.0 is canonical focal length
        pred_depth = pred_depth * canonical_to_real_scale
        pred_depth = torch.clamp(pred_depth, 0, 300)

        return pred_depth.unsqueeze(0)  # [1, H, W]

    def get_required_env(self):
        return "metric3d"
