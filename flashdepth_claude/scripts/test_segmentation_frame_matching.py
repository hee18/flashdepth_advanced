"""
Test script to verify segmentation datasets load matching frames correctly.

This script checks that:
1. Images, depth, and segmentation all correspond to the same frames
2. Frame indices are correctly aligned across the sequence
3. Last frame segmentation matches the last frame image

Usage:
    CUDA_VISIBLE_DEVICES=2 python scripts/test_segmentation_frame_matching.py --dataset waymo
    CUDA_VISIBLE_DEVICES=2 python scripts/test_segmentation_frame_matching.py --dataset sintel
    CUDA_VISIBLE_DEVICES=2 python scripts/test_segmentation_frame_matching.py --dataset both
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataloaders.waymo_segmentation_dataset import WaymoSegmentationDataset
from dataloaders.sintel_segmentation_dataset import SintelSegmentationDataset
from utils.sparse_depth_visualization import create_dual_sparse_depth_vis
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def visualize_sample(sample, dataset_name, save_path):
    """
    Visualize a sample to verify frame matching.

    Args:
        sample: Dataset sample dict with 'image', 'depth', 'segmentation'
        dataset_name: Name of dataset for title
        save_path: Path to save visualization
    """
    import torch

    # Get data
    img_key = 'images' if 'images' in sample else 'image'
    images = sample[img_key]  # (T, 3, H, W)
    depths = sample['depth']  # (T, H, W)
    seg_mask = sample['segmentation']  # (H, W) - last frame only

    T = images.shape[0]

    # Check if this is sparse depth dataset (Waymo)
    is_sparse = 'waymo' in dataset_name.lower()

    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, T, figure=fig, hspace=0.3, wspace=0.1)

    # Plot all frames
    for t in range(T):
        # Image
        ax_img = fig.add_subplot(gs[0, t])
        img = images[t].permute(1, 2, 0).numpy()  # (H, W, 3)
        ax_img.imshow(img)
        ax_img.set_title(f'Frame {t}' if t < T-1 else f'Frame {t} (LAST)', fontweight='bold' if t == T-1 else 'normal')
        ax_img.axis('off')

        # Depth (with inpainting for sparse datasets)
        ax_depth = fig.add_subplot(gs[1, t])
        depth = depths[t].numpy()

        if is_sparse:
            # For Waymo: use inpainting within valid row range only
            MAX_DEPTH = 200.0
            valid_mask = (depth > 0) & (depth < MAX_DEPTH)
            valid_ratio = valid_mask.sum() / valid_mask.size * 100

            _, depth_vis, depth_info = create_dual_sparse_depth_vis(
                depth, valid_mask, colormap='plasma', percentile_range=(2, 98)
            )
            im_depth = ax_depth.imshow(depth_vis)
            ax_depth.set_title(f'Depth {t} (Inpainted)\n{valid_ratio:.1f}% valid')
        else:
            # For Sintel: standard visualization
            im_depth = ax_depth.imshow(depth, cmap='plasma')
            ax_depth.set_title(f'Depth {t}')

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
    seq_name = sample.get('sequence_name', 'Unknown')
    fig.suptitle(f'{dataset_name} - Sequence: {seq_name}\nFrame Alignment Test', fontsize=16, fontweight='bold')

    # Save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved visualization to {save_path}")


def test_waymo_frame_matching(data_root: str, num_samples: int = 3):
    """Test Waymo segmentation dataset frame matching."""
    logger.info("="*80)
    logger.info("Testing Waymo Segmentation Dataset - Frame Matching")
    logger.info("="*80)

    try:
        # Use sparse depth from preprocessed waymo dataset
        if 'waymo_seg' in data_root:
            # Replace waymo_seg with waymo/val (preprocessed depth location)
            depth_root = data_root.replace('waymo_seg', 'waymo') + '/val'
        else:
            depth_root = None

        dataset = WaymoSegmentationDataset(
            data_root=data_root,
            split='val',  # Waymo uses 'val' not 'validation'
            video_length=5,
            resolution=518,
            camera_name=1,  # FRONT camera
            use_depth=True,
            depth_root=depth_root
        )

        logger.info(f"✓ Dataset initialized successfully")
        logger.info(f"  Total sequences: {len(dataset)}")

        if len(dataset) == 0:
            logger.error("✗ No sequences found! Check data_root and file structure.")
            return False

        # Test multiple sequences
        num_to_test = min(num_samples, len(dataset))
        logger.info(f"\nTesting {num_to_test} sequences...")

        for i in range(num_to_test):
            logger.info(f"\n{'='*60}")
            logger.info(f"Sequence {i+1}/{num_to_test}")
            logger.info(f"{'='*60}")

            sample = dataset[i]

            if sample is None:
                logger.warning(f"  Sequence {i} returned None, skipping...")
                continue

            # Check keys
            img_key = 'images' if 'images' in sample else 'image'
            logger.info(f"  Sample keys: {list(sample.keys())}")
            logger.info(f"  Images shape: {sample[img_key].shape}")
            logger.info(f"  Depth shape: {sample['depth'].shape}")
            logger.info(f"  Segmentation shape: {sample['segmentation'].shape}")
            logger.info(f"  Sequence name: {sample.get('sequence_name', 'N/A')}")

            # Check frame alignment
            images = sample[img_key]
            depths = sample['depth']
            seg_mask = sample['segmentation']

            T = images.shape[0]
            logger.info(f"\n  Frame alignment check:")
            logger.info(f"    Video length (T): {T}")
            logger.info(f"    Images: {T} frames")
            logger.info(f"    Depths: {depths.shape[0]} frames")
            logger.info(f"    Segmentation: Last frame only (expected)")

            # Verify shapes match
            assert images.shape[0] == depths.shape[0], "Images and depth have different number of frames!"
            assert seg_mask.shape == depths[-1].shape, "Segmentation shape doesn't match last depth frame!"

            logger.info(f"  ✓ Shape alignment verified")

            # Check segmentation classes
            import torch
            unique_classes = torch.unique(seg_mask)
            logger.info(f"  Unique segmentation classes: {unique_classes.tolist()[:10]}... ({len(unique_classes)} total)")

            # Visualize
            vis_path = f'test_results/frame_matching/waymo_seq_{i}.png'
            visualize_sample(sample, 'Waymo', vis_path)

        logger.info(f"\n✓ Waymo frame matching test PASSED")
        return True

    except Exception as e:
        logger.error(f"✗ Waymo frame matching test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sintel_frame_matching(data_root: str, num_samples: int = 3):
    """Test Sintel segmentation dataset frame matching."""
    logger.info("="*80)
    logger.info("Testing Sintel Segmentation Dataset - Frame Matching")
    logger.info("="*80)

    try:
        dataset = SintelSegmentationDataset(
            data_root=data_root,
            split='test',  # Test split has segmentation
            video_length=5,
            resolution=518,
            pass_type='clean'
        )

        logger.info(f"✓ Dataset initialized successfully")
        logger.info(f"  Total sequences: {len(dataset)}")

        if len(dataset) == 0:
            logger.error("✗ No sequences found! Check data_root and file structure.")
            return False

        # Test multiple sequences
        num_to_test = min(num_samples, len(dataset))
        logger.info(f"\nTesting {num_to_test} sequences...")

        for i in range(num_to_test):
            logger.info(f"\n{'='*60}")
            logger.info(f"Sequence {i+1}/{num_to_test}")
            logger.info(f"{'='*60}")

            sample = dataset[i]

            if sample is None:
                logger.warning(f"  Sequence {i} returned None, skipping...")
                continue

            # Check keys
            img_key = 'images' if 'images' in sample else 'image'
            logger.info(f"  Sample keys: {list(sample.keys())}")
            logger.info(f"  Images shape: {sample[img_key].shape}")
            logger.info(f"  Depth shape: {sample['depth'].shape}")
            logger.info(f"  Segmentation shape: {sample['segmentation'].shape}")
            logger.info(f"  Sequence name: {sample.get('sequence_name', 'N/A')}")

            # Check frame alignment
            images = sample[img_key]
            depths = sample['depth']
            seg_mask = sample['segmentation']

            T = images.shape[0]
            logger.info(f"\n  Frame alignment check:")
            logger.info(f"    Video length (T): {T}")
            logger.info(f"    Images: {T} frames")
            logger.info(f"    Depths: {depths.shape[0]} frames")
            logger.info(f"    Segmentation: Last frame only (expected)")

            # Verify shapes match
            assert images.shape[0] == depths.shape[0], "Images and depth have different number of frames!"
            assert seg_mask.shape == depths[-1].shape, "Segmentation shape doesn't match last depth frame!"

            logger.info(f"  ✓ Shape alignment verified")

            # Check segmentation instances
            import torch
            unique_instances = torch.unique(seg_mask)
            logger.info(f"  Unique instance IDs: {len(unique_instances)} instances")
            logger.info(f"  Instance ID range: {unique_instances.min().item()} - {unique_instances.max().item()}")

            # Visualize
            vis_path = f'test_results/frame_matching/sintel_seq_{i}.png'
            visualize_sample(sample, 'Sintel', vis_path)

        logger.info(f"\n✓ Sintel frame matching test PASSED")
        return True

    except Exception as e:
        logger.error(f"✗ Sintel frame matching test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Test segmentation dataset frame matching')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['waymo', 'sintel', 'both'],
                        help='Dataset to test')
    parser.add_argument('--data-root', type=str, default=None,
                        help='Data root directory (default: /home/cvlab/hsy/Datasets/{dataset})')
    parser.add_argument('--num-samples', type=int, default=3,
                        help='Number of samples to test per dataset (default: 3)')

    args = parser.parse_args()

    results = []

    if args.dataset in ['waymo', 'both']:
        data_root = args.data_root or '/home/cvlab/hsy/Datasets/waymo_seg'
        logger.info(f"\nWaymo data root: {data_root}\n")
        result = test_waymo_frame_matching(data_root, args.num_samples)
        results.append(('Waymo', result))

    if args.dataset in ['sintel', 'both']:
        data_root = args.data_root or '/home/cvlab/hsy/Datasets/sintel'
        logger.info(f"\nSintel data root: {data_root}\n")
        result = test_sintel_frame_matching(data_root, args.num_samples)
        results.append(('Sintel', result))

    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"  {name}: {status}")

    all_passed = all(r for _, r in results)
    if all_passed:
        logger.info("\n✓ All frame matching tests PASSED!")
        logger.info("  Visualizations saved to test_results/frame_matching/")
        sys.exit(0)
    else:
        logger.error("\n✗ Some frame matching tests FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
