"""
Onepiece Depth Estimator Wrapper

Wraps the FlashDepth model with Onepiece V3 (SpatialMamba + CLSMetricHead)
for single-frame streaming metric depth estimation.
"""

import sys
import os
import torch
import torch.nn.functional as F
import logging
import yaml
from pathlib import Path
from omegaconf import OmegaConf

# Add flashdepth project root to path
FLASHDEPTH_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(FLASHDEPTH_ROOT))

from flashdepth.model import FlashDepth

logger = logging.getLogger(__name__)


class OnepieceDepthEstimator:
    """
    Onepiece V3 metric depth estimator with temporal streaming.

    Uses forward_onepiece_single_frame() for per-frame inference
    with Mamba hidden state maintained across frames.
    Scene cuts are automatically detected via CLS token cosine distance.
    """

    def __init__(self, config_path, checkpoint_path, device='cuda', use_bfloat16=True):
        """
        Args:
            config_path: Path to onepiece config.yaml (relative to FLASHDEPTH_ROOT)
            checkpoint_path: Path to .pth checkpoint (relative to FLASHDEPTH_ROOT)
            device: 'cuda' or 'cpu'
            use_bfloat16: Use bfloat16 mixed precision
        """
        self.device = device
        self.use_bfloat16 = use_bfloat16
        self.prev_cls = None

        # Load config
        config_abs = FLASHDEPTH_ROOT / config_path
        with open(config_abs, 'r') as f:
            config = OmegaConf.create(yaml.safe_load(f))

        # Build model
        model_config = dict(config.model)
        model_config['batch_size'] = 1
        model_config['use_metric_head'] = False
        model_config['use_onepiece'] = True
        model_config['spatial_mamba_layers'] = config.model.get('spatial_mamba_layers', 4)
        model_config['spatial_mamba_d_state'] = config.model.get('spatial_mamba_d_state', 256)
        model_config['spatial_mamba_d_conv'] = config.model.get('spatial_mamba_d_conv', 4)
        model_config['spatial_mamba_downsample'] = config.model.get('spatial_mamba_downsample', 0.1)
        model_config['onepiece_train_mode'] = config.get('train_mode', 'metric')

        scene_cut_config = config.get('scene_cut', {})
        model_config['scene_cut_tau'] = scene_cut_config.get('tau', 0.05) if scene_cut_config else 0.05
        model_config['scene_cut_k'] = scene_cut_config.get('k', 80) if scene_cut_config else 80

        model_config['hybrid_configs'] = config.get('hybrid_configs', None)

        self.model = FlashDepth(**model_config)

        # Load checkpoint
        ckpt_abs = FLASHDEPTH_ROOT / checkpoint_path
        logger.info(f"Loading Onepiece checkpoint: {ckpt_abs}")
        checkpoint = torch.load(str(ckpt_abs), map_location='cpu')

        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
        logger.info("Onepiece model loaded successfully")

        self.model = self.model.to(self.device)
        self.model.eval()

        # ImageNet normalization constants
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def reset(self):
        """Reset Mamba temporal state for a new video sequence."""
        self.model.spatial_mamba.start_new_sequence()
        self.prev_cls = None

    def preprocess(self, frame_bgr):
        """
        Preprocess a BGR frame (from OpenCV) to model input tensor.
        No resizing — frame should already be at the desired resolution.

        Args:
            frame_bgr: [H, W, 3] uint8 BGR numpy array (already resized)

        Returns:
            tensor: [1, 3, H, W] normalized float tensor
        """
        import cv2

        # BGR -> RGB, normalize
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # To tensor [1, 3, H, W], float32, 0-1 range
        tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = tensor.to(self.device)

        # ImageNet normalization
        tensor = (tensor - self.mean) / self.std

        return tensor

    @torch.no_grad()
    def estimate(self, frame_tensor):
        """
        Run single-frame depth estimation.

        Args:
            frame_tensor: [1, 3, H, W] preprocessed tensor

        Returns:
            dict with:
                metric_depth: [H, W] numpy array (meters)
                relative_depth: [H, W] numpy array
                scale: float
                shift: float
                is_reset: bool (scene cut detected)
                d_cls: float (CLS cosine distance)
        """
        ctx = torch.autocast('cuda', dtype=torch.bfloat16) if self.use_bfloat16 else torch.no_grad()
        with ctx:
            outputs = self.model.forward_onepiece_single_frame(
                frame_tensor, prev_cls=self.prev_cls
            )

        self.prev_cls = outputs['cls_token']

        return {
            'metric_depth': outputs['metric_depth'][0].float().cpu().numpy(),  # [H, W]
            'relative_depth': outputs['relative_depth'][0].float().cpu().numpy(),
            'scale': float(outputs['scale']),
            'shift': float(outputs['shift']),
            'is_reset': outputs['is_reset'],
            'd_cls': outputs['d_cls'],
        }

    def estimate_from_bgr(self, frame_bgr):
        """
        Convenience: preprocess + estimate in one call.
        No resizing — frame should already be at the desired resolution.

        Args:
            frame_bgr: [H, W, 3] uint8 BGR numpy array (already resized)

        Returns:
            dict with metric_depth at input frame resolution [H, W]
        """
        tensor = self.preprocess(frame_bgr)
        return self.estimate(tensor)
