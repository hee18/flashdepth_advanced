#!/usr/bin/env python3
"""
Test script to verify Waymo sparse depth loading from preprocessed dataset.

Usage:
    CUDA_VISIBLE_DEVICES=2 python scripts/test_waymo_sparse_depth.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataloaders.waymo_segmentation_dataset import WaymoSegmentationDataset
from utils.sparse_depth_visualization import create_dual_sparse_depth_vis
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def visualize_sample_with_depth(sample, save_path):
    """
    Visualize a Waymo sample with sparse depth.

    Args:
        sample: Dataset sample dict
        save_path: Path to save visualization
    """
    images = sample['image']  # (T, 3, H, W)
    depths = sample['depth']  # (T, H, W)
    seg_mask = sample['segmentation']  # (H, W)
    seq_name = sample['sequence_name']

    T = images.shape[0]

    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, T, figure=fig, hspace=0.3, wspace=0.1)

    MAX_DEPTH = 200.0

    for t in range(T):
        # Image
        ax_img = fig.add_subplot(gs[0, t])
        img = images[t].permute(1, 2, 0).numpy()
        ax_img.imshow(img)
        ax_img.set_title(f'Frame {t}' if t < T-1 else f'Frame {t} (LAST)',
                        fontweight='bold' if t == T-1 else 'normal')
        ax_img.axis('off')

        # Sparse depth with inpainting
        ax_depth = fig.add_subplot(gs[1, t])
        depth = depths[t].numpy()

        valid_mask = (depth > 0) & (depth < MAX_DEPTH)
        valid_ratio = valid_mask.sum() / valid_mask.size * 100

        _, depth_vis, depth_info = create_dual_sparse_depth_vis(
            depth, valid_mask, colormap='plasma', percentile_range=(2, 98)
        )
        im_depth = ax_depth.imshow(depth_vis)
        ax_depth.set_title(f'Depth {t} (Sparse + Inpaint)\n{valid_ratio:.1f}% valid')
        ax_depth.axis('off')
        plt.colorbar(im_depth, ax=ax_depth, fraction=0.046, pad=0.04)

        # Segmentation (only for last frame)
        ax_seg = fig.add_subplot(gs[2, t])
        if t == T - 1:
            seg = seg_mask.numpy()
            im_seg = ax_seg.imshow(seg, cmap='tab20')
            ax_seg.set_title(f'Segmentation (Last Frame)', fontweight='bold')
            plt.colorbar(im_seg, ax=ax_seg, fraction=0.046, pad=0.04)
        else:
            ax_seg.text(0.5, 0.5, 'No segmentation\n(not last frame)',
                       ha='center', va='center', transform=ax_seg.transAxes)
            ax_seg.set_title('N/A')
        ax_seg.axis('off')

    # Overall title
    fig.suptitle(f'Waymo - Sequence: {seq_name}\nSparse Depth Verification',
                 fontsize=16, fontweight='bold')

    # Save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved visualization to {save_path}")


def main():
    logger.info("="*80)
    logger.info("Testing Waymo Sparse Depth Loading")
    logger.info("="*80)

    # Initialize dataset WITH depth loading
    data_root = '/home/cvlab/hsy/Datasets/waymo_seg'
    depth_root = '/home/cvlab/hsy/Datasets/waymo/val'

    try:
        dataset = WaymoSegmentationDataset(
            data_root=data_root,
            split='val',
            video_length=5,
            resolution=518,
            camera_name=1,  # FRONT camera
            use_depth=True,
            depth_root=depth_root
        )

        logger.info(f"✓ Dataset initialized with depth loading")
        logger.info(f"  Total sequences: {len(dataset)}")
        logger.info(f"  Depth root: {depth_root}")

        if len(dataset) == 0:
            logger.error("✗ No sequences found!")
            return False

        # Test 3 sequences
        num_to_test = min(3, len(dataset))
        logger.info(f"\nTesting {num_to_test} sequences...")

        for i in range(num_to_test):
            logger.info(f"\n{'='*60}")
            logger.info(f"Sequence {i+1}/{num_to_test}")
            logger.info(f"{'='*60}")

            sample = dataset[i]

            if sample is None:
                logger.warning(f"  Sequence {i} returned None, skipping...")
                continue

            images = sample['image']
            depths = sample['depth']
            seg_mask = sample['segmentation']
            valid_mask = sample['valid_mask']

            logger.info(f"  Images shape: {images.shape}")
            logger.info(f"  Depth shape: {depths.shape}")
            logger.info(f"  Segmentation shape: {seg_mask.shape}")
            logger.info(f"  Sequence name: {sample['sequence_name']}")

            # Check depth statistics
            for t in range(depths.shape[0]):
                depth = depths[t]
                valid_points = (depth > 0).sum().item()
                if valid_points > 0:
                    min_d = depth[depth > 0].min().item()
                    max_d = depth[depth > 0].max().item()
                    mean_d = depth[depth > 0].mean().item()
                    logger.info(f"  Frame {t} depth: {valid_points} points, "
                              f"range [{min_d:.2f}, {max_d:.2f}]m, mean {mean_d:.2f}m")
                else:
                    logger.warning(f"  Frame {t} depth: 0 valid points!")

            # Valid mask check
            valid_ratio = valid_mask.sum().item() / valid_mask.numel() * 100
            logger.info(f"  Valid mask: {valid_ratio:.1f}% valid pixels")

            # Visualize
            vis_path = f'test_results/sparse_depth/waymo_seq_{i}.png'
            visualize_sample_with_depth(sample, vis_path)

        logger.info(f"\n✓ Waymo sparse depth test PASSED")
        logger.info(f"  Visualizations saved to test_results/sparse_depth/")
        return True

    except Exception as e:
        logger.error(f"✗ Waymo sparse depth test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
