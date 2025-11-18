#!/usr/bin/env python3
"""
Verify actual class IDs in waymo_seg, urbansyn, and vkitti datasets.
Run this script to check what class IDs are actually present in the data.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from collections import Counter
import cv2

def check_waymo_classes(dataset_root):
    """Check Waymo segmentation class IDs"""
    print("=" * 80)
    print("WAYMO SEGMENTATION DATASET")
    print("=" * 80)

    waymo_paths = [
        dataset_root / "waymo_seg" / "val",
        dataset_root / "Waymo_Seg" / "val",
        dataset_root / "waymo_seg",
    ]

    waymo_root = None
    for path in waymo_paths:
        if path.exists():
            waymo_root = path
            break

    if not waymo_root or not waymo_root.exists():
        print(f"❌ Waymo dataset not found at {dataset_root}")
        return None

    print(f"✓ Found: {waymo_root}")

    seq_dirs = sorted([d for d in waymo_root.iterdir() if d.is_dir() and d.name.startswith('segment-')])

    if len(seq_dirs) == 0:
        print("❌ No sequences found")
        return None

    all_class_ids = set()
    class_pixel_counts = Counter()

    # Check first 3 sequences
    for seq_dir in seq_dirs[:3]:
        seg_dir = seq_dir / 'FRONT' / 'segmentation'
        if not seg_dir.exists():
            continue

        seg_files = sorted(seg_dir.glob('*.png'))
        print(f"\nSequence: {seq_dir.name}")
        print(f"  Segmentation files: {len(seg_files)}")

        # Check first 5 segmentation files
        for seg_file in seg_files[:5]:
            seg_mask = np.array(Image.open(seg_file))
            unique_classes = np.unique(seg_mask)
            all_class_ids.update(unique_classes)

            for class_id in unique_classes:
                count = (seg_mask == class_id).sum()
                class_pixel_counts[class_id] += count

    print(f"\n{'All unique class IDs found:'}")
    print(f"  {sorted(all_class_ids)}")

    print(f"\n{'Class pixel counts (top 20):'}")
    for class_id, count in class_pixel_counts.most_common(20):
        print(f"  Class {class_id:3d}: {count:,} pixels")

    return sorted(all_class_ids)


def check_urbansyn_classes(dataset_root):
    """Check UrbanSyn segmentation class IDs"""
    print("\n" + "=" * 80)
    print("URBANSYN DATASET")
    print("=" * 80)

    urbansyn_paths = [
        dataset_root / "urbansyn",
        dataset_root / "UrbanSyn",
        dataset_root / "urban_syn",
    ]

    urbansyn_root = None
    for path in urbansyn_paths:
        if path.exists():
            urbansyn_root = path
            break

    if not urbansyn_root or not urbansyn_root.exists():
        print(f"❌ UrbanSyn dataset not found at {dataset_root}")
        return None

    print(f"✓ Found: {urbansyn_root}")

    # Find segmentation directory
    seg_dirs = list(urbansyn_root.rglob('ss_trainid'))
    if len(seg_dirs) == 0:
        print("❌ No ss_trainid directory found")
        return None

    all_class_ids = set()
    class_pixel_counts = Counter()

    # Check first directory
    seg_dir = seg_dirs[0]
    print(f"✓ Found segmentation: {seg_dir}")

    seg_files = sorted(seg_dir.glob('ss_color_*.png'))[:10]
    print(f"  Checking {len(seg_files)} files...")

    for seg_file in seg_files:
        seg_mask = np.array(Image.open(seg_file))
        unique_classes = np.unique(seg_mask)
        all_class_ids.update(unique_classes)

        for class_id in unique_classes:
            count = (seg_mask == class_id).sum()
            class_pixel_counts[class_id] += count

    print(f"\n{'All unique class IDs found:'}")
    print(f"  {sorted(all_class_ids)}")

    print(f"\n{'Class pixel counts (top 20):'}")
    for class_id, count in class_pixel_counts.most_common(20):
        print(f"  Class {class_id:3d}: {count:,} pixels")

    return sorted(all_class_ids)


def check_vkitti_classes(dataset_root):
    """Check VKITTI segmentation class IDs"""
    print("\n" + "=" * 80)
    print("VKITTI DATASET")
    print("=" * 80)

    vkitti_paths = [
        dataset_root / "vkitti",
        dataset_root / "VKITTI",
        dataset_root / "vkitti2",
        dataset_root / "VKITTI2",
    ]

    vkitti_root = None
    for path in vkitti_paths:
        if path.exists():
            vkitti_root = path
            break

    if not vkitti_root or not vkitti_root.exists():
        print(f"❌ VKITTI dataset not found at {dataset_root}")
        return None

    print(f"✓ Found: {vkitti_root}")

    # Find classSegmentation directory
    seg_dirs = list(vkitti_root.rglob('classSegmentation'))
    if len(seg_dirs) == 0:
        print("❌ No classSegmentation directory found")
        return None

    # RGB to Class ID mapping (from colors.txt)
    rgb_to_class = {
        (210, 0, 200): 0,      # Terrain
        (90, 200, 255): 1,     # Sky
        (0, 199, 0): 2,        # Tree
        (90, 240, 0): 3,       # Vegetation
        (140, 140, 140): 4,    # Building
        (100, 60, 100): 5,     # Road
        (250, 100, 255): 6,    # GuardRail
        (255, 255, 0): 7,      # TrafficSign
        (200, 200, 0): 8,      # TrafficLight
        (255, 130, 0): 9,      # Pole
        (80, 80, 80): 10,      # Misc
        (160, 60, 60): 11,     # Truck
        (255, 127, 80): 12,    # Car
        (0, 139, 139): 13,     # Van
        (0, 0, 0): 14,         # Undefined
    }

    all_rgb_colors = set()
    all_class_ids = set()
    class_pixel_counts = Counter()

    # Check first directory
    for seg_dir in seg_dirs[:1]:
        print(f"✓ Found segmentation: {seg_dir}")

        camera_dir = seg_dir / 'Camera_0'
        if not camera_dir.exists():
            camera_dir = list(seg_dir.glob('Camera_*'))[0]

        seg_files = sorted(camera_dir.glob('classgt_*.png'))[:10]
        print(f"  Checking {len(seg_files)} files...")

        for seg_file in seg_files:
            seg_rgb = np.array(Image.open(seg_file))

            # Extract unique RGB colors
            pixels = seg_rgb.reshape(-1, 3)
            unique_colors = np.unique(pixels, axis=0)

            for color in unique_colors:
                rgb_tuple = tuple(color)
                all_rgb_colors.add(rgb_tuple)

                if rgb_tuple in rgb_to_class:
                    class_id = rgb_to_class[rgb_tuple]
                    all_class_ids.add(class_id)
                    count = np.all(seg_rgb == color, axis=2).sum()
                    class_pixel_counts[class_id] += count

    print(f"\n{'All unique RGB colors found:'}")
    for color in sorted(all_rgb_colors):
        class_id = rgb_to_class.get(color, -1)
        print(f"  RGB{color}: Class {class_id}")

    print(f"\n{'All unique class IDs found:'}")
    print(f"  {sorted(all_class_ids)}")

    print(f"\n{'Class pixel counts:'}")
    for class_id, count in class_pixel_counts.most_common():
        print(f"  Class {class_id:3d}: {count:,} pixels")

    return sorted(all_class_ids)


if __name__ == "__main__":
    dataset_root = Path("/home/cvlab/hsy/Datasets")

    print(f"Searching for datasets in: {dataset_root}\n")

    waymo_classes = check_waymo_classes(dataset_root)
    urbansyn_classes = check_urbansyn_classes(dataset_root)
    vkitti_classes = check_vkitti_classes(dataset_root)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Waymo class IDs: {waymo_classes}")
    print(f"UrbanSyn class IDs: {urbansyn_classes}")
    print(f"VKITTI class IDs: {vkitti_classes}")
