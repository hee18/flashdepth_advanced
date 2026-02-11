"""
Adapter for original FlashDepth (with Mamba temporal processing)

Loads FlashDepth model and runs video inference frame-by-frame with Mamba state.
Outputs depth in meters (converted from inverse depth * 100).
"""

import sys
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from .base_adapter import MethodAdapter

logger = logging.getLogger(__name__)


class FlashDepthAdapter(MethodAdapter):
    """Adapter for original FlashDepth with Mamba temporal processing"""

    CONFIG_MAP = {
        'l': 'flashdepth-l',
        's': 'flashdepth-s',
        'hybrid': 'flashdepth',
    }

    DEFAULT_CHECKPOINTS = {
        'l': 'configs/flashdepth-l/iter_10001.pth',
        's': 'configs/flashdepth-s/iter_14001.pth',
        'hybrid': 'configs/flashdepth/iter_43002.pth',
    }

    # Resolution mapping from combined_dataset.py (width, height), all 14x-divisible
    BASE_RESOLUTION_MAP = {
        'eth3d': (784, 518),
        'waymo': (784, 518),
        'waymo_seg': (784, 518),
        'sintel': (1022, 434),
        'urbansyn': (1036, 518),
        'unreal4k': (924, 518),
        'tartanair': (518, 518),
        'vkitti': (1246, 378),
        'nuscenes': (924, 518),
        'bonn': (630, 476),
    }

    def __init__(self, config_variant='l', checkpoint_path=None):
        super().__init__()
        self.config_variant = config_variant
        self._checkpoint_path = checkpoint_path
        self._first_inference = True
        self._target_resolution = None  # (H, W) set via set_dataset()

    def set_dataset(self, dataset_name):
        """Set target processing resolution based on dataset name.

        Uses the same resolution mapping as combined_dataset.py (base, test split).
        """
        ds = dataset_name.lower().split('/')[0] if dataset_name else ''
        if ds in self.BASE_RESOLUTION_MAP:
            w, h = self.BASE_RESOLUTION_MAP[ds]
            self._target_resolution = (h, w)
            logger.info(f"[FlashDepth] Target resolution for '{ds}': {h}x{w}")
        else:
            self._target_resolution = None
            logger.info(f"[FlashDepth] No resolution mapping for '{ds}', will round to 14x multiple")

    def load_model(self, checkpoint_path=None):
        """Load FlashDepth model"""
        from flashdepth.model import FlashDepth

        # Load config
        config_dir = self.CONFIG_MAP.get(self.config_variant, 'flashdepth-l')
        config_path = Path(__file__).parent.parent / 'configs' / config_dir / 'config.yaml'
        cfg = OmegaConf.load(config_path)

        # Build model
        model = FlashDepth(
            batch_size=1,
            training=False,
            hybrid_configs=cfg.hybrid_configs,
            **dict(cfg.model),
        )

        # Determine checkpoint
        ckpt_path = checkpoint_path or self._checkpoint_path
        if ckpt_path is None:
            ckpt_path = str(
                Path(__file__).parent.parent
                / self.DEFAULT_CHECKPOINTS.get(self.config_variant, self.DEFAULT_CHECKPOINTS['l'])
            )

        logger.info(f"[FlashDepth] Loading checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')

        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Remove DDP 'module.' prefix
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=False)
        logger.info(f"[FlashDepth] Model loaded (variant={self.config_variant})")

        self.model = model
        return model

    def inference(self, image, intrinsics=None):
        """
        Run FlashDepth inference on video sequence.

        Args:
            image: torch.Tensor [1, T, 3, H, W] - Input video (0-1 range, RGB)
            intrinsics: Optional (not used)

        Returns:
            depth: torch.Tensor [1, T, H, W] - Predicted depth (meters)
        """
        if image.ndim == 4:
            image = image.unsqueeze(1)

        B, T, C, H, W = image.shape
        assert B == 1

        # Apply ImageNet normalization (FlashDepth expects normalized input)
        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(1, 1, 3, 1, 1)
        image_norm = (image - mean) / std

        # Determine processing resolution (must be multiple of patch_size=14)
        # Priority: dataset-specific mapping > round down to nearest 14x multiple
        ps = self.model.patch_size  # 14
        if self._target_resolution is not None:
            H_proc, W_proc = self._target_resolution
        else:
            H_proc = (H // ps) * ps
            W_proc = (W // ps) * ps
        need_resize = (H_proc != H) or (W_proc != W)

        if need_resize:
            image_norm = F.interpolate(
                image_norm.view(B * T, C, H, W),
                size=(H_proc, W_proc), mode='bilinear', align_corners=False
            ).view(B, T, C, H_proc, W_proc)

        if self.processing_resolution is None:
            self.processing_resolution = (H, W)
            if need_resize:
                logger.info(f"[FlashDepth] Processing resolution: {H}x{W} -> resized to {H_proc}x{W_proc}")
            else:
                logger.info(f"[FlashDepth] Processing resolution: {H}x{W}")

        # Run inference frame-by-frame with Mamba temporal state
        self.model.mamba.start_new_sequence()

        patch_h = H_proc // ps
        patch_w = W_proc // ps

        preds = []
        for i in range(T):
            frame = image_norm[:, i]  # [1, 3, H_proc, W_proc]
            dpt_features = self.model.get_dpt_features(frame, input_shape=(1, C, H_proc, W_proc))
            pred_depth = self.model.final_head(dpt_features, patch_h, patch_w)  # [1, H_proc, W_proc]
            pred_depth = torch.clamp(pred_depth, min=0)
            preds.append(pred_depth)

        # Stack: [1, T, H_proc, W_proc]
        pred_inverse = torch.stack(preds, dim=1)

        # Convert from inverse depth * 100 to depth in meters
        # Output stays at processing resolution (no upsample to original)
        depth = 100.0 / (pred_inverse + 1e-8)

        if self._first_inference:
            logger.info(f"[FlashDepth] Output shape: {depth.shape}")
            logger.info(f"[FlashDepth] Depth range: min={depth.min():.3f}, max={depth.max():.3f}, mean={depth.mean():.3f}")
            self._first_inference = False

        return depth

    def get_required_env(self):
        return "flashdepth"
