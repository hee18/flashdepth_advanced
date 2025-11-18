#!/usr/bin/env python3
"""
Visualize Waymo segmentation classes with different colors.
Shows RGB image alongside class-colored segmentation to verify labels.
"""

import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Waymo class definitions
WAYMO_CLASSES = {
    0: 'undefined',
    1: 'car',
    2: 'truck',
    3: 'bus',
    4: 'other_vehicle',
    5: 'motorcyclist',
    6: 'bicyclist',
    7: 'pedestrian',
    8: 'sign',
    9: 'traffic_light',
    10: 'pole',
    11: 'construction_cone',
    12: 'bicycle',
    13: 'motorcycle',
    14: 'building',
    15: 'vegetation',
    16: 'tree_trunk',
    17: 'curb',
    18: 'road',
    19: 'lane_marker',
    20: 'other_ground',
    21: 'walkable',
    22: 'sidewalk',
    23: 'unknown_23',
    24: 'unknown_24',
    25: 'unknown_25',
    26: 'unknown_26',
    27: 'unknown_27',
    28: 'unknown_28',
}

# Color palette (distinct colors for each class)
CLASS_COLORS = {
    0: [255, 0, 0],      # undefined - Red
    1: [255, 127, 0],    # car - Orange
    2: [255, 255, 0],    # truck - Yellow
    3: [127, 255, 0],    # bus - Yellow-Green
    4: [0, 255, 0],      # other_vehicle - Green
    5: [0, 255, 127],    # motorcyclist - Cyan-Green
    6: [0, 255, 255],    # bicyclist - Cyan
    7: [0, 127, 255],    # pedestrian - Sky Blue
    8: [0, 0, 255],      # sign - Blue
    9: [127, 0, 255],    # traffic_light - Purple
    10: [255, 0, 255],   # pole - Magenta
    11: [255, 0, 127],   # construction_cone - Pink
    12: [127, 127, 0],   # bicycle - Olive
    13: [0, 127, 127],   # motorcycle - Teal
    14: [127, 0, 127],   # building - Purple-Red
    15: [64, 128, 0],    # vegetation - Dark Green
    16: [128, 64, 0],    # tree_trunk - Brown
    17: [128, 128, 128], # curb - Gray
    18: [64, 64, 64],    # road - Dark Gray
    19: [192, 192, 192], # lane_marker - Light Gray
    20: [160, 82, 45],   # other_ground - Sienna
    21: [210, 180, 140], # walkable - Tan
    22: [176, 196, 222], # sidewalk - Light Steel Blue
    23: [255, 192, 203], # unknown_23 - Pink
    24: [255, 218, 185], # unknown_24 - Peach
    25: [240, 230, 140], # unknown_25 - Khaki
    26: [221, 160, 221], # unknown_26 - Plum
    27: [238, 130, 238], # unknown_27 - Violet
    28: [147, 112, 219], # unknown_28 - Medium Purple
}


def seg_mask_to_color(seg_mask, class_colors):
    """Convert segmentation mask to RGB color image"""
    h, w = seg_mask.shape
    color_seg = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in class_colors.items():
        mask = (seg_mask == class_id)
        color_seg[mask] = color

    return color_seg


def visualize_waymo_sequence(dataset_root, sequence_name=None, frame_idx=0, output_dir=None):
    """
    Visualize Waymo segmentation classes with colors.

    Args:
        dataset_root: Path to waymo_seg/val
        sequence_name: Sequence to visualize (None = first sequence)
        frame_idx: Frame index to visualize
        output_dir: Where to save visualization
    """
    dataset_root = Path(dataset_root)

    # Find sequences
    seq_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir() and d.name.startswith('segment-')])

    if len(seq_dirs) == 0:
        print(f"No sequences found in {dataset_root}")
        return

    # Select sequence
    if sequence_name:
        seq_dir = dataset_root / sequence_name
        if not seq_dir.exists():
            print(f"Sequence {sequence_name} not found")
            return
    else:
        seq_dir = seq_dirs[0]

    print(f"Visualizing sequence: {seq_dir.name}")

    # Load RGB image
    rgb_dir = seq_dir / 'FRONT' / 'rgb' / 'original'
    rgb_files = sorted(rgb_dir.glob('*.jpg'))

    if frame_idx >= len(rgb_files):
        print(f"Frame {frame_idx} not found (only {len(rgb_files)} frames)")
        frame_idx = 0

    rgb_path = rgb_files[frame_idx]
    rgb_image = Image.open(rgb_path)

    # Load segmentation
    seg_dir = seq_dir / 'FRONT' / 'segmentation'
    seg_path = seg_dir / f'{frame_idx:04d}.png'

    if not seg_path.exists():
        print(f"Segmentation not found: {seg_path}")
        return

    seg_mask = np.array(Image.open(seg_path))

    # Get unique classes in this frame
    unique_classes = np.unique(seg_mask)
    print(f"\nFrame {frame_idx}: {rgb_path.name}")
    print(f"Unique class IDs: {unique_classes}")

    # Count pixels per class
    class_counts = {}
    for class_id in unique_classes:
        count = (seg_mask == class_id).sum()
        class_counts[class_id] = count
        class_name = WAYMO_CLASSES.get(class_id, f'unknown_{class_id}')
        print(f"  Class {class_id:2d} ({class_name:20s}): {count:8,} pixels ({count/seg_mask.size*100:5.2f}%)")

    # Convert segmentation to color
    color_seg = seg_mask_to_color(seg_mask, CLASS_COLORS)

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # 1. RGB Image
    axes[0].imshow(rgb_image)
    axes[0].set_title('RGB Image', fontsize=16, fontweight='bold')
    axes[0].axis('off')

    # 2. Color-coded Segmentation
    axes[1].imshow(color_seg)
    axes[1].set_title('Segmentation (Class Colors)', fontsize=16, fontweight='bold')
    axes[1].axis('off')

    # 3. Overlay (50% transparency)
    axes[2].imshow(rgb_image)
    axes[2].imshow(color_seg, alpha=0.5)
    axes[2].set_title('RGB + Segmentation Overlay', fontsize=16, fontweight='bold')
    axes[2].axis('off')

    # Create legend (only for classes present in this frame)
    legend_elements = []
    sorted_classes = sorted(unique_classes, key=lambda x: class_counts[x], reverse=True)

    for class_id in sorted_classes[:15]:  # Top 15 classes
        class_name = WAYMO_CLASSES.get(class_id, f'unknown_{class_id}')
        color = np.array(CLASS_COLORS[class_id]) / 255.0
        count = class_counts[class_id]
        label = f'ID {class_id}: {class_name} ({count:,} px)'
        legend_elements.append(mpatches.Patch(facecolor=color, label=label))

    # Add legend below the plots
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               fontsize=10, frameon=True, fancybox=True, shadow=True)

    plt.suptitle(f'Waymo Segmentation Visualization\n{seq_dir.name} - Frame {frame_idx}',
                 fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.12, 1, 0.96])

    # Save
    if output_dir is None:
        output_dir = Path('waymo_class_visualization')
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{seq_dir.name}_frame{frame_idx:04d}_classes.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {output_path}")

    plt.close()

    # Also create a class ID map (grayscale showing actual IDs)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    axes[0].imshow(rgb_image)
    axes[0].set_title('RGB Image', fontsize=16, fontweight='bold')
    axes[0].axis('off')

    im = axes[1].imshow(seg_mask, cmap='tab20', vmin=0, vmax=28)
    axes[1].set_title('Class ID Map (Grayscale)', fontsize=16, fontweight='bold')
    axes[1].axis('off')

    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Class ID', fontsize=12)

    plt.suptitle(f'Waymo Class ID Map\n{seq_dir.name} - Frame {frame_idx}',
                 fontsize=18, fontweight='bold')
    plt.tight_layout()

    id_map_path = output_dir / f'{seq_dir.name}_frame{frame_idx:04d}_id_map.png'
    plt.savefig(id_map_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved ID map to: {id_map_path}")

    plt.close()

    return color_seg, class_counts


if __name__ == "__main__":
    import sys

    # Default path
    dataset_root = Path("/home/cvlab/hsy/Datasets/waymo_seg/val")

    # Parse arguments
    sequence_name = None
    frame_idx = 0

    if len(sys.argv) > 1:
        sequence_name = sys.argv[1]
    if len(sys.argv) > 2:
        frame_idx = int(sys.argv[2])

    print("=" * 80)
    print("WAYMO SEGMENTATION CLASS VISUALIZATION")
    print("=" * 80)
    print(f"Dataset: {dataset_root}")
    print(f"Sequence: {sequence_name or 'first available'}")
    print(f"Frame: {frame_idx}")
    print()

    # Visualize first sequence, first frame
    visualize_waymo_sequence(dataset_root, sequence_name, frame_idx)

    print("\n" + "=" * 80)
    print("USAGE:")
    print("  python visualize_waymo_classes.py [sequence_name] [frame_idx]")
    print()
    print("EXAMPLES:")
    print("  python visualize_waymo_classes.py")
    print("  python visualize_waymo_classes.py segment-10017090168044687777_6380_000_6400_000 5")
    print("=" * 80)
