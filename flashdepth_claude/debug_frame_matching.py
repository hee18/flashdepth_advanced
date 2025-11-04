"""
Debug script to verify frame matching between images and segmentations.
Checks if images[i] and segmentations[i] actually come from the same frame_idx.
"""

import numpy as np
import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import sys

# Add path for imports
sys.path.append('/home/cvlab/hsy/flashdepth_advanced/flashdepth_claude')

from dataloaders.waymo_segmentation_dataset import WaymoSegmentationDataset, collate_fn
from torch.utils.data import DataLoader

def visualize_frame_matching(dataset, sequence_idx=0):
    """
    Load a sequence and visualize first 5 frames to verify matching.
    """
    print(f"\n{'='*80}")
    print(f"Testing sequence {sequence_idx}")
    print(f"{'='*80}\n")

    # Get raw data
    sample = dataset[sequence_idx]

    if sample is None:
        print("ERROR: Sample is None!")
        return

    images = sample['images']  # [T, 3, H, W]
    segmentations = sample['segmentations']  # [T, H, W]
    frame_indices = sample['frame_indices']  # List of actual frame numbers
    sequence_name = sample['sequence_name']

    T = images.shape[0]
    print(f"Sequence: {sequence_name}")
    print(f"Number of frames loaded: {T}")
    print(f"Frame indices: {frame_indices}")
    print(f"Images shape: {images.shape}")
    print(f"Segmentations shape: {segmentations.shape}")

    # Get the actual file paths for verification
    seq_dir, num_frames, orig_frame_indices = dataset.sequences[sequence_idx]
    camera_dir = seq_dir / dataset.camera_name
    rgb_dir = camera_dir / 'rgb' / 'original'
    seg_dir = camera_dir / 'segmentation'

    print(f"\nVerifying file loading for first {min(5, T)} frames:")
    print(f"{'-'*80}")

    for i in range(min(5, T)):
        frame_idx = frame_indices[i]
        rgb_path = rgb_dir / f'{frame_idx:04d}.jpg'
        seg_path = seg_dir / f'{frame_idx:04d}.png'

        print(f"\nBatch index {i} → Frame {frame_idx}:")
        print(f"  RGB: {rgb_path.name} (exists: {rgb_path.exists()})")
        print(f"  Seg: {seg_path.name} (exists: {seg_path.exists()})")

        # Load original files directly to compare
        if rgb_path.exists() and seg_path.exists():
            img_orig = Image.open(rgb_path).convert('RGB')
            seg_orig = Image.open(seg_path)
            seg_orig_np = np.array(seg_orig)

            # Check segmentation annotation
            annotated_pixels = (seg_orig_np > 0).sum()
            total_pixels = seg_orig_np.size
            annotation_pct = 100.0 * annotated_pixels / total_pixels

            print(f"  Original image size: {img_orig.size}")
            print(f"  Original seg size: {seg_orig.size}")
            print(f"  Annotation: {annotated_pixels}/{total_pixels} pixels ({annotation_pct:.1f}%)")
            print(f"  Unique classes: {np.unique(seg_orig_np)}")

    # Create visualization
    num_viz_frames = min(5, T)
    fig, axes = plt.subplots(num_viz_frames, 3, figsize=(15, 5*num_viz_frames))

    if num_viz_frames == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_viz_frames):
        frame_idx = frame_indices[i]

        # Denormalize image
        img = images[i].permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = img * std + mean
        img = np.clip(img, 0, 1)

        # Get segmentation
        seg = segmentations[i].cpu().numpy()

        # Load original files for comparison
        rgb_path = rgb_dir / f'{frame_idx:04d}.jpg'
        seg_path = seg_dir / f'{frame_idx:04d}.png'

        img_orig = np.array(Image.open(rgb_path).convert('RGB'))
        seg_orig = np.array(Image.open(seg_path))

        # Show processed (resized) image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Batch idx {i}, Frame {frame_idx}\n(Processed, {img.shape[1]}x{img.shape[0]})')
        axes[i, 0].axis('off')

        # Show original image (downsampled for display)
        axes[i, 1].imshow(img_orig)
        axes[i, 1].set_title(f'Original RGB\n({img_orig.shape[1]}x{img_orig.shape[0]})')
        axes[i, 1].axis('off')

        # Show segmentation with colormap
        seg_viz = seg.copy().astype(np.float32)
        seg_viz[seg_viz == 0] = np.nan  # Make background transparent
        axes[i, 2].imshow(img, alpha=0.5)  # Faded background
        axes[i, 2].imshow(seg_viz, cmap='tab20', alpha=0.7, vmin=0, vmax=19)
        axes[i, 2].set_title(f'Segmentation Overlay\n({seg.shape[1]}x{seg.shape[0]})')
        axes[i, 2].axis('off')

    plt.tight_layout()
    output_path = f'/tmp/debug_frame_matching_seq{sequence_idx}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n{'='*80}")
    print(f"Saved visualization to: {output_path}")
    print(f"{'='*80}\n")
    plt.close()


def check_dataloader_batch():
    """
    Test with DataLoader to verify batch collation.
    """
    print("\n" + "="*80)
    print("Testing DataLoader batch collation")
    print("="*80 + "\n")

    dataset = WaymoSegmentationDataset(
        data_root='/home/cvlab/hsy/Datasets/waymo_seg',
        split='val',
        video_length=20,
        resolution=518,
        objwise_mode=True
    )

    print(f"Dataset size: {len(dataset)} sequences\n")

    # Test first sequence
    visualize_frame_matching(dataset, sequence_idx=0)

    # Test with dataloader
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

    print("\nDataLoader batch info:")
    print(f"  images: {batch['images'].shape}")
    print(f"  segmentations: {batch['segmentations'].shape}")
    print(f"  frame_indices: {batch['frame_indices']}")
    print(f"  sequence_name: {batch['sequence_name']}")


if __name__ == '__main__':
    check_dataloader_batch()
