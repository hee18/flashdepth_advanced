"""
Debug script to trace best_frame extraction in test_gear3_upgrade.py.
Runs a single sequence and saves detailed frame-by-frame visualizations.
"""

import numpy as np
import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import sys
import logging

# Add path for imports
sys.path.append('/home/cvlab/hsy/flashdepth_advanced/flashdepth_claude')

from dataloaders.waymo_segmentation_dataset import WaymoSegmentationDataset, collate_fn
from torch.utils.data import DataLoader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def visualize_batch_structure(batch, best_frame_idx=5):
    """
    Visualize the batch structure and show what gets extracted for best_frame.
    """
    print(f"\n{'='*80}")
    print("BATCH STRUCTURE ANALYSIS")
    print(f"{'='*80}\n")

    # Batch info
    images = batch['images']  # [B, T, 3, H, W]
    depths = batch['depth']   # [B, T, H, W]
    segmentations = batch['segmentations']  # [B, T, H, W]
    frame_indices = batch['frame_indices']  # List[List[int]]
    sequence_name = batch['sequence_name']  # List[str]

    B, T = images.shape[:2]
    H, W = images.shape[3:]

    print(f"Batch shapes:")
    print(f"  images: {images.shape}")
    print(f"  depths: {depths.shape}")
    print(f"  segmentations: {segmentations.shape}")
    print(f"  frame_indices: {frame_indices}")
    print(f"  sequence_name: {sequence_name}")
    print(f"\nAssuming best_frame_idx = {best_frame_idx}")
    print(f"{'='*80}\n")

    # Simulate extraction like test_gear3_upgrade.py
    # From line 559-562
    images_extracted = images  # [1, T, 3, H, W]
    if images_extracted.ndim == 4:
        images_extracted = images_extracted.unsqueeze(0)

    # From line 874
    seg_masks = segmentations[0]  # [T, H, W] - batch 0
    seg_masks_np = seg_masks.cpu().numpy() if isinstance(seg_masks, torch.Tensor) else seg_masks

    # From line 977 and 995
    seg_mask_for_viz = seg_masks_np[best_frame_idx]  # [H, W]
    image_for_viz = images_extracted[0, best_frame_idx]  # [3, H, W]

    # Get actual frame number from frame_indices
    actual_frame_number = frame_indices[0][best_frame_idx]

    print(f"EXTRACTION FOR BEST_FRAME:")
    print(f"  best_frame_idx (batch index): {best_frame_idx}")
    print(f"  actual_frame_number (from frame_indices): {actual_frame_number}")
    print(f"  image_for_viz shape: {image_for_viz.shape}")
    print(f"  seg_mask_for_viz shape: {seg_mask_for_viz.shape}")
    print(f"\n{'-'*80}\n")

    # Denormalize image
    img_np = image_for_viz.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = img_np * std + mean
    img_np = np.clip(img_np, 0, 1)

    # Get original files for verification
    seq_dir = Path('/home/cvlab/hsy/Datasets/waymo_seg/val') / sequence_name[0]
    camera_dir = seq_dir / 'FRONT'
    rgb_dir = camera_dir / 'rgb' / 'original'
    seg_dir = camera_dir / 'segmentation'

    rgb_path = rgb_dir / f'{actual_frame_number:04d}.jpg'
    seg_path = seg_dir / f'{actual_frame_number:04d}.png'

    print(f"VERIFICATION - Loading original files:")
    print(f"  RGB: {rgb_path}")
    print(f"  Seg: {seg_path}")

    if not rgb_path.exists():
        print(f"  ERROR: RGB file does not exist!")
        return

    if not seg_path.exists():
        print(f"  ERROR: Seg file does not exist!")
        return

    img_orig = np.array(Image.open(rgb_path).convert('RGB'))
    seg_orig = np.array(Image.open(seg_path))

    print(f"  RGB original shape: {img_orig.shape}")
    print(f"  Seg original shape: {seg_orig.shape}")
    print(f"\n{'-'*80}\n")

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Row 1: Processed (resized) versions
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title(f'Processed Image\n(batch_idx={best_frame_idx}, frame={actual_frame_number})\n{img_np.shape[1]}x{img_np.shape[0]}', fontsize=10)
    axes[0, 0].axis('off')

    seg_viz = seg_mask_for_viz.copy().astype(np.float32)
    seg_viz[seg_viz == 0] = np.nan
    axes[0, 1].imshow(img_np, alpha=0.5)
    axes[0, 1].imshow(seg_viz, cmap='tab20', alpha=0.7, vmin=0, vmax=19)
    axes[0, 1].set_title(f'Processed Seg Overlay\n(batch_idx={best_frame_idx}, frame={actual_frame_number})\n{seg_mask_for_viz.shape[1]}x{seg_mask_for_viz.shape[0]}', fontsize=10)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(seg_mask_for_viz, cmap='tab20', vmin=0, vmax=19)
    axes[0, 2].set_title(f'Processed Seg Mask Only\nUnique classes: {np.unique(seg_mask_for_viz)}', fontsize=10)
    axes[0, 2].axis('off')

    # Row 2: Original files
    axes[1, 0].imshow(img_orig)
    axes[1, 0].set_title(f'Original RGB File\n{rgb_path.name}\n{img_orig.shape[1]}x{img_orig.shape[0]}', fontsize=10)
    axes[1, 0].axis('off')

    seg_orig_viz = seg_orig.copy().astype(np.float32)
    seg_orig_viz[seg_orig_viz == 0] = np.nan
    axes[1, 1].imshow(img_orig, alpha=0.5)
    axes[1, 1].imshow(seg_orig_viz, cmap='tab20', alpha=0.7, vmin=0, vmax=19)
    axes[1, 1].set_title(f'Original Seg Overlay\n{seg_path.name}\n{seg_orig.shape[1]}x{seg_orig.shape[0]}', fontsize=10)
    axes[1, 1].axis('off')

    axes[1, 2].imshow(seg_orig, cmap='tab20', vmin=0, vmax=19)
    axes[1, 2].set_title(f'Original Seg Mask Only\nUnique classes: {np.unique(seg_orig)}', fontsize=10)
    axes[1, 2].axis('off')

    plt.tight_layout()
    output_path = f'/tmp/debug_best_frame_idx{best_frame_idx}_frame{actual_frame_number}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")
    print(f"{'='*80}\n")
    plt.close()

    # Check if scenes match
    print(f"VISUAL VERIFICATION:")
    print(f"  Do the processed and original images show THE SAME SCENE?")
    print(f"  Do the processed and original segmentations show THE SAME OBJECTS?")
    print(f"  If not, there's a frame mismatch bug!")
    print(f"{'='*80}\n")


def test_all_frames():
    """
    Test all frames to find mismatches.
    """
    dataset = WaymoSegmentationDataset(
        data_root='/home/cvlab/hsy/Datasets/waymo_seg',
        split='val',
        video_length=20,
        resolution=518,
        objwise_mode=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    batch = next(iter(dataloader))

    if batch is None:
        print("ERROR: Batch is None!")
        return

    # Test frames 0, 5, 10, 14, 19
    test_indices = [0, 5, 10, 14, 19]

    for idx in test_indices:
        visualize_batch_structure(batch, best_frame_idx=idx)


if __name__ == '__main__':
    test_all_frames()
