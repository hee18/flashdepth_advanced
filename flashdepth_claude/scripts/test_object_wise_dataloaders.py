"""
Test script to verify object-wise dataloaders work correctly.

Usage:
    python scripts/test_object_wise_dataloaders.py --dataset waymo
    python scripts/test_object_wise_dataloaders.py --dataset sintel
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataloaders.waymo_segmentation_dataset import WaymoSegmentationDataset
from dataloaders.sintel_segmentation_dataset import SintelSegmentationDataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_waymo(data_root: str):
    """Test Waymo segmentation dataset."""
    logger.info("="*60)
    logger.info("Testing Waymo Segmentation Dataset")
    logger.info("="*60)

    try:
        dataset = WaymoSegmentationDataset(
            data_root=data_root,
            split='val',
            video_length=5,
            resolution=518,
            camera_name=1  # FRONT camera
        )

        logger.info(f"✓ Dataset initialized successfully")
        logger.info(f"  Total sequences: {len(dataset)}")

        if len(dataset) == 0:
            logger.error("✗ No sequences found! Check data_root and file structure.")
            return False

        # Test loading first sequence
        logger.info(f"\nTesting first sequence...")
        sample = dataset[0]

        logger.info(f"✓ Sample loaded successfully")
        logger.info(f"  Sample keys: {list(sample.keys())}")

        # Check which key exists
        img_key = 'images' if 'images' in sample else 'image'
        logger.info(f"  Images shape: {sample[img_key].shape}")
        logger.info(f"  Depth shape: {sample['depth'].shape}")
        logger.info(f"  Segmentation shape: {sample['segmentation'].shape}")
        logger.info(f"  Sequence name: {sample.get('sequence_name', 'N/A')}")

        # Check unique segmentation classes
        import torch
        seg = sample['segmentation']
        unique_classes = torch.unique(seg)
        logger.info(f"  Unique segmentation classes: {unique_classes.tolist()}")

        logger.info(f"\n✓ Waymo dataset test PASSED")
        return True

    except Exception as e:
        logger.error(f"✗ Waymo dataset test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sintel(data_root: str):
    """Test Sintel segmentation dataset."""
    logger.info("="*60)
    logger.info("Testing Sintel Segmentation Dataset")
    logger.info("="*60)

    try:
        dataset = SintelSegmentationDataset(
            data_root=data_root,
            split='val',
            video_length=5,
            resolution=518,
            pass_type='clean'
        )

        logger.info(f"✓ Dataset initialized successfully")
        logger.info(f"  Total sequences: {len(dataset)}")

        if len(dataset) == 0:
            logger.error("✗ No sequences found! Check data_root and file structure.")
            return False

        # Test loading first sequence
        logger.info(f"\nTesting first sequence...")
        sample = dataset[0]

        logger.info(f"✓ Sample loaded successfully")
        logger.info(f"  Sample keys: {list(sample.keys())}")

        # Check which key exists
        img_key = 'images' if 'images' in sample else 'image'
        logger.info(f"  Images shape: {sample[img_key].shape}")
        logger.info(f"  Depth shape: {sample['depth'].shape}")
        logger.info(f"  Segmentation shape: {sample['segmentation'].shape}")
        logger.info(f"  Sequence name: {sample.get('sequence_name', 'N/A')}")

        # Check unique segmentation instances
        import torch
        seg = sample['segmentation']
        unique_instances = torch.unique(seg)
        logger.info(f"  Unique instance IDs: {len(unique_instances)} instances")

        logger.info(f"\n✓ Sintel dataset test PASSED")
        return True

    except Exception as e:
        logger.error(f"✗ Sintel dataset test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Test object-wise dataloaders')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['waymo', 'sintel', 'both'],
                        help='Dataset to test')
    parser.add_argument('--data-root', type=str, default=None,
                        help='Data root directory (default: /home/cvlab/hsy/Datasets/{dataset}_seg)')

    args = parser.parse_args()

    results = []

    if args.dataset in ['waymo', 'both']:
        data_root = args.data_root or '/home/cvlab/hsy/Datasets/waymo_seg'
        logger.info(f"\nData root: {data_root}\n")
        result = test_waymo(data_root)
        results.append(('Waymo', result))

    if args.dataset in ['sintel', 'both']:
        data_root = args.data_root or '/home/cvlab/hsy/Datasets/sintel_seg'
        logger.info(f"\nData root: {data_root}\n")
        result = test_sintel(data_root)
        results.append(('Sintel', result))

    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"  {name}: {status}")

    all_passed = all(r for _, r in results)
    if all_passed:
        logger.info("\n✓ All tests PASSED!")
        sys.exit(0)
    else:
        logger.error("\n✗ Some tests FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
