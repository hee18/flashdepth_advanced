#!/usr/bin/env python3
"""
FG Mask Generation Script

Generate foreground masks using FlashDepth-L's ViT attention weights.
Saves binary PNG masks (0=background, 255=foreground) for each frame.

Usage:
    python scripts/generate_fg_masks.py \
        --data-root /path/to/datasets \
        --checkpoint configs/flashdepth-l/iter_10001.pth \
        --datasets eth3d,sintel,waymo_seg,vkitti,unreal4k \
        --gpu 0

Author: Claude Code
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flashdepth.model import FlashDepth

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FGMaskGenerator:
    """Generate foreground masks from FlashDepth-L attention weights."""

    # ViT-L intermediate layer indices
    INTERMEDIATE_LAYER_IDX = [4, 11, 17, 23]
    # Default: use layers 2 and 4 (blocks 11 and 23)
    TARGET_BLOCKS = [11, 23]

    def __init__(self, checkpoint_path: str, device: str = 'cuda:0'):
        self.device = device
        self.model = self._load_model(checkpoint_path)
        self.patch_size = 14  # ViT patch size

    def _load_model(self, checkpoint_path: str) -> FlashDepth:
        """Load FlashDepth-L model with attention storage enabled."""
        logger.info(f"Loading FlashDepth-L from {checkpoint_path}")

        # Model config for ViT-L
        model_config = {
            'vit_size': 'vitl',
            'batch_size': 1,
            'use_metric_head': False,
            'use_mamba': False,  # No temporal processing needed
        }

        model = FlashDepth(**model_config)

        # Load checkpoint
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # Remove module. prefix if present
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            # Load (strict=False for flexibility)
            model.load_state_dict(state_dict, strict=False)
            logger.info("Checkpoint loaded successfully")
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")

        # Enable attention storage for target blocks
        for i, block in enumerate(model.pretrained.blocks):
            if i in self.TARGET_BLOCKS:
                block.attn.store_attn_weights = True
                logger.info(f"Enabled attention storage for block {i}")
            else:
                block.attn.store_attn_weights = False

        model = model.to(self.device)
        model.eval()

        return model

    @torch.no_grad()
    def generate_fg_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Generate foreground mask from a single image.

        Args:
            image: RGB image as numpy array (H, W, 3) uint8

        Returns:
            fg_mask: Binary mask (H, W) uint8, 0=background, 255=foreground
        """
        orig_h, orig_w = image.shape[:2]

        # Preprocess: resize to 518x518, normalize
        img_resized = cv2.resize(image, (518, 518), interpolation=cv2.INTER_LINEAR)
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0

        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        img_tensor = img_tensor.unsqueeze(0).to(self.device)  # [1, 3, 518, 518]

        # Forward pass through encoder
        patch_h = 518 // self.patch_size  # 37
        patch_w = 518 // self.patch_size  # 37

        # Get intermediate features (this triggers attention computation)
        _ = self.model.pretrained.get_intermediate_layers(
            img_tensor, self.INTERMEDIATE_LAYER_IDX
        )

        # Extract attention weights from target blocks
        attention_weights_list = []
        for block_idx in self.TARGET_BLOCKS:
            block = self.model.pretrained.blocks[block_idx]
            attn_weights = block.attn.attn_weights  # [1, num_heads, N+1, N+1]
            attention_weights_list.append(attn_weights)

        # Generate importance map
        importance_map = self._compute_importance_map(
            attention_weights_list, patch_h, patch_w
        )  # [1, 1, patch_h, patch_w]

        # Upsample to original resolution
        importance_map_resized = F.interpolate(
            importance_map,
            size=(orig_h, orig_w),
            mode='bilinear',
            align_corners=True
        )  # [1, 1, orig_h, orig_w]

        # Threshold to binary mask
        importance_flat = importance_map_resized.flatten()
        threshold = importance_flat.mean()
        fg_mask = (importance_map_resized > threshold).squeeze().cpu().numpy()

        # Convert to uint8 (0 or 255)
        fg_mask = (fg_mask * 255).astype(np.uint8)

        return fg_mask

    def _compute_importance_map(
        self,
        attention_weights_list: list,
        patch_h: int,
        patch_w: int
    ) -> torch.Tensor:
        """
        Compute importance map from attention weights.
        Reimplements ImportanceMapGenerator logic.
        """
        # Extract CLS-to-patch attention from each layer
        cls_to_patch_list = []
        for attn in attention_weights_list:
            # attn: [B, num_heads, N+1, N+1]
            # CLS row: attn[:, :, 0, 1:] -> [B, num_heads, N]
            cls_to_patch = attn[:, :, 0, 1:]  # [B, num_heads, num_patches]
            cls_to_patch = cls_to_patch.mean(dim=1)  # Average over heads: [B, num_patches]
            cls_to_patch_list.append(cls_to_patch)

        # Average across layers
        cls_attention = torch.stack(cls_to_patch_list, dim=0).mean(dim=0)  # [B, num_patches]

        # Reshape to spatial dimensions
        num_patches = cls_attention.shape[1]
        expected_patches = patch_h * patch_w

        if num_patches != expected_patches:
            # Handle patch mismatch (unlikely for 518x518)
            cls_attention = F.interpolate(
                cls_attention.unsqueeze(1), size=expected_patches,
                mode='linear', align_corners=True
            ).squeeze(1)

        # Reshape to 2D: [B, 1, patch_h, patch_w]
        importance_map = cls_attention.view(-1, patch_h, patch_w).unsqueeze(1)

        # Remove register token (highest attention patch) with 3x3 inpainting
        B = importance_map.shape[0]
        for b in range(B):
            attn_2d = importance_map[b, 0]  # [patch_h, patch_w]

            # Find the patch with maximum attention
            max_val = attn_2d.max()
            outlier_mask = (attn_2d == max_val)

            # Inpaint with local average
            kernel = torch.ones(1, 1, 3, 3, device=importance_map.device) / 9
            attn_smoothed = F.conv2d(importance_map[b:b+1], kernel, padding=1)
            importance_map[b, 0] = torch.where(
                outlier_mask, attn_smoothed[0, 0], importance_map[b, 0]
            )

        # Percentile normalization (1-99 percentile) to [0, 1]
        for b in range(B):
            attn_flat = importance_map[b].flatten()
            attn_p1 = torch.quantile(attn_flat, 0.01)
            attn_p99 = torch.quantile(attn_flat, 0.99)

            importance_map[b] = (importance_map[b] - attn_p1) / (attn_p99 - attn_p1 + 1e-8)
            importance_map[b] = torch.clamp(importance_map[b], 0.0, 1.0)

        return importance_map


def process_eth3d(data_root: Path, generator: FGMaskGenerator, skip_existing: bool = True):
    """Process ETH3D dataset."""
    eth3d_root = data_root / 'eth3d'
    if not eth3d_root.exists():
        logger.warning(f"ETH3D not found at {eth3d_root}")
        return

    logger.info(f"Processing ETH3D from {eth3d_root}")

    # Find all scenes
    scenes = sorted([d for d in eth3d_root.iterdir() if d.is_dir()])

    total_processed = 0
    total_skipped = 0

    for scene_dir in tqdm(scenes, desc="ETH3D scenes"):
        images_dir = scene_dir / 'images' / 'dslr_images'
        if not images_dir.exists():
            continue

        # Create fg_masks directory
        fg_masks_dir = scene_dir / 'fg_masks'
        fg_masks_dir.mkdir(exist_ok=True)

        # Find all images
        image_files = sorted(images_dir.glob('*.JPG')) + sorted(images_dir.glob('*.jpg'))

        for img_path in tqdm(image_files, desc=f"  {scene_dir.name}", leave=False):
            # Output path
            out_path = fg_masks_dir / f"{img_path.stem}.png"

            if skip_existing and out_path.exists():
                total_skipped += 1
                continue

            # Load and process
            image = cv2.imread(str(img_path))
            if image is None:
                logger.warning(f"Failed to load {img_path}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Generate fg_mask
            fg_mask = generator.generate_fg_mask(image)

            # Save
            cv2.imwrite(str(out_path), fg_mask)
            total_processed += 1

    logger.info(f"ETH3D: processed {total_processed}, skipped {total_skipped}")


def process_sintel(data_root: Path, generator: FGMaskGenerator, skip_existing: bool = True):
    """Process Sintel dataset."""
    sintel_root = data_root / 'sintel'
    if not sintel_root.exists():
        logger.warning(f"Sintel not found at {sintel_root}")
        return

    logger.info(f"Processing Sintel from {sintel_root}")

    images_base = sintel_root / 'images' / 'training' / 'clean'
    if not images_base.exists():
        logger.warning(f"Sintel images not found at {images_base}")
        return

    # Create fg_masks directory structure
    fg_masks_base = sintel_root / 'fg_masks' / 'training' / 'clean'

    scenes = sorted([d for d in images_base.iterdir() if d.is_dir()])

    total_processed = 0
    total_skipped = 0

    for scene_dir in tqdm(scenes, desc="Sintel scenes"):
        fg_masks_dir = fg_masks_base / scene_dir.name
        fg_masks_dir.mkdir(parents=True, exist_ok=True)

        image_files = sorted(scene_dir.glob('*.png'))

        for img_path in tqdm(image_files, desc=f"  {scene_dir.name}", leave=False):
            out_path = fg_masks_dir / img_path.name

            if skip_existing and out_path.exists():
                total_skipped += 1
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            fg_mask = generator.generate_fg_mask(image)
            cv2.imwrite(str(out_path), fg_mask)
            total_processed += 1

    logger.info(f"Sintel: processed {total_processed}, skipped {total_skipped}")


def process_waymo_seg(data_root: Path, generator: FGMaskGenerator, skip_existing: bool = True):
    """Process Waymo Segmentation dataset (val sequences only)."""
    waymo_root = data_root / 'waymo_seg' / 'val'
    if not waymo_root.exists():
        logger.warning(f"Waymo not found at {waymo_root}")
        return

    logger.info(f"Processing Waymo from {waymo_root}")

    # Val sequences (8 total)
    val_sequences = [
        'segment-10017090168044687777_6380_000_6400_000',
        'segment-10023947602400723454_1120_000_1140_000',
        'segment-1005081002024129653_5313_150_5333_150',
        'segment-10061305430875486848_1080_000_1100_000',
        'segment-10072140764565668044_4060_000_4080_000',
        'segment-10072231702153043603_5725_000_5745_000',
        'segment-10075870402459732738_1060_000_1080_000',
        'segment-10094743350625019937_3420_000_3440_000',
    ]

    total_processed = 0
    total_skipped = 0

    for seq_name in tqdm(val_sequences, desc="Waymo sequences"):
        seq_dir = waymo_root / seq_name / 'FRONT'
        if not seq_dir.exists():
            logger.warning(f"Sequence not found: {seq_name}")
            continue

        images_dir = seq_dir / 'rgb' / 'original'
        if not images_dir.exists():
            continue

        fg_masks_dir = seq_dir / 'fg_masks'
        fg_masks_dir.mkdir(exist_ok=True)

        image_files = sorted(images_dir.glob('*.jpg'))

        for img_path in tqdm(image_files, desc=f"  {seq_name[:20]}...", leave=False):
            out_path = fg_masks_dir / f"{img_path.stem}.png"

            if skip_existing and out_path.exists():
                total_skipped += 1
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            fg_mask = generator.generate_fg_mask(image)
            cv2.imwrite(str(out_path), fg_mask)
            total_processed += 1

    logger.info(f"Waymo: processed {total_processed}, skipped {total_skipped}")


def process_vkitti(data_root: Path, generator: FGMaskGenerator, skip_existing: bool = True):
    """Process VKITTI2 dataset (clone condition only)."""
    vkitti_root = data_root / 'vkitti'
    if not vkitti_root.exists():
        logger.warning(f"VKITTI not found at {vkitti_root}")
        return

    logger.info(f"Processing VKITTI from {vkitti_root}")

    # Scenes
    scenes = ['Scene01', 'Scene02', 'Scene06', 'Scene18', 'Scene20']
    condition = 'clone'

    total_processed = 0
    total_skipped = 0

    for scene in tqdm(scenes, desc="VKITTI scenes"):
        images_dir = vkitti_root / scene / condition / 'frames' / 'rgb' / 'Camera_0'
        if not images_dir.exists():
            logger.warning(f"Scene not found: {scene}/{condition}")
            continue

        fg_masks_dir = vkitti_root / scene / condition / 'frames' / 'fg_masks' / 'Camera_0'
        fg_masks_dir.mkdir(parents=True, exist_ok=True)

        image_files = sorted(images_dir.glob('rgb_*.jpg'))

        for img_path in tqdm(image_files, desc=f"  {scene}", leave=False):
            # Convert rgb_00000.jpg to fg_00000.png
            frame_num = img_path.stem.replace('rgb_', '')
            out_path = fg_masks_dir / f"fg_{frame_num}.png"

            if skip_existing and out_path.exists():
                total_skipped += 1
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            fg_mask = generator.generate_fg_mask(image)
            cv2.imwrite(str(out_path), fg_mask)
            total_processed += 1

    logger.info(f"VKITTI: processed {total_processed}, skipped {total_skipped}")


def process_unreal4k(data_root: Path, generator: FGMaskGenerator, skip_existing: bool = True):
    """Process UnrealStereo4K dataset."""
    unreal_root = data_root / 'unreal4k'
    if not unreal_root.exists():
        logger.warning(f"Unreal4K not found at {unreal_root}")
        return

    logger.info(f"Processing Unreal4K from {unreal_root}")

    # 9 sequences (0-8)
    sequences = [f"UnrealStereo4K_{i:05d}" for i in range(9)]

    total_processed = 0
    total_skipped = 0

    for seq_name in tqdm(sequences, desc="Unreal4K sequences"):
        seq_dir = unreal_root / seq_name
        if not seq_dir.exists():
            logger.warning(f"Sequence not found: {seq_name}")
            continue

        images_dir = seq_dir / 'Image0'
        if not images_dir.exists():
            continue

        fg_masks_dir = seq_dir / 'fg_masks'
        fg_masks_dir.mkdir(exist_ok=True)

        image_files = sorted(images_dir.glob('*.png'))

        for img_path in tqdm(image_files, desc=f"  {seq_name}", leave=False):
            out_path = fg_masks_dir / img_path.name

            if skip_existing and out_path.exists():
                total_skipped += 1
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            fg_mask = generator.generate_fg_mask(image)
            cv2.imwrite(str(out_path), fg_mask)
            total_processed += 1

    logger.info(f"Unreal4K: processed {total_processed}, skipped {total_skipped}")


def process_bonn(data_root: Path, generator: FGMaskGenerator, skip_existing: bool = True):
    """Process Bonn RGB-D Dynamic Dataset."""
    bonn_root = data_root / 'bonn'
    if not bonn_root.exists():
        logger.warning(f"Bonn not found at {bonn_root}")
        return

    logger.info(f"Processing Bonn from {bonn_root}")

    # Get all sequence directories (rgbd_bonn_*)
    sequences = sorted([d for d in bonn_root.iterdir()
                       if d.is_dir() and d.name.startswith('rgbd_bonn_')])

    if not sequences:
        logger.warning(f"No Bonn sequences found at {bonn_root}")
        return

    total_processed = 0
    total_skipped = 0

    for seq_dir in tqdm(sequences, desc="Bonn sequences"):
        rgb_dir = seq_dir / 'rgb'
        if not rgb_dir.exists():
            logger.warning(f"RGB dir not found: {rgb_dir}")
            continue

        # Create fg_masks directory
        fg_masks_dir = seq_dir / 'fg_masks'
        fg_masks_dir.mkdir(parents=True, exist_ok=True)

        # Get RGB images
        image_files = sorted(rgb_dir.glob('*.png'))

        for img_path in tqdm(image_files, desc=f"  {seq_dir.name}", leave=False):
            out_path = fg_masks_dir / img_path.name

            if skip_existing and out_path.exists():
                total_skipped += 1
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            fg_mask = generator.generate_fg_mask(image)
            cv2.imwrite(str(out_path), fg_mask)
            total_processed += 1

    logger.info(f"Bonn: processed {total_processed}, skipped {total_skipped}")


def main():
    parser = argparse.ArgumentParser(description="Generate FG masks from FlashDepth-L attention")
    parser.add_argument('--data-root', type=str, required=True,
                        help='Root directory containing datasets')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to FlashDepth-L checkpoint')
    parser.add_argument('--datasets', type=str, default='eth3d,sintel,waymo_seg,vkitti,unreal4k',
                        help='Comma-separated list of datasets to process')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                        help='Skip existing fg_mask files')
    parser.add_argument('--no-skip-existing', dest='skip_existing', action='store_false',
                        help='Overwrite existing fg_mask files')

    args = parser.parse_args()

    # Setup device
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Initialize generator
    generator = FGMaskGenerator(args.checkpoint, device)

    # Process datasets
    data_root = Path(args.data_root)
    datasets = [d.strip() for d in args.datasets.split(',')]

    logger.info(f"Processing datasets: {datasets}")
    logger.info(f"Skip existing: {args.skip_existing}")

    dataset_processors = {
        'eth3d': process_eth3d,
        'sintel': process_sintel,
        'waymo_seg': process_waymo_seg,
        'vkitti': process_vkitti,
        'unreal4k': process_unreal4k,
        'bonn': process_bonn,
    }

    for dataset in datasets:
        if dataset in dataset_processors:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing {dataset}")
            logger.info(f"{'='*50}")
            dataset_processors[dataset](data_root, generator, args.skip_existing)
        else:
            logger.warning(f"Unknown dataset: {dataset}")

    logger.info("\nDone!")


if __name__ == '__main__':
    main()
