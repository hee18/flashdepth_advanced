#!/usr/bin/env python3
"""
Reorganize ETH3D dataset structure.

Before:
  /eth3d/multi_view_training_dslr_undistorted/{scene}/images/dslr_images_undistorted/*.JPG
  /eth3d/{scene}/ground_truth_depth/dslr_images/*.JPG (depth files)

After:
  /eth3d/{scene}/images/dslr_images/*.JPG (moved RGB images)
  /eth3d/{scene}/ground_truth_depth/dslr_images/*.JPG (depth files, unchanged)
"""

import os
import shutil
from pathlib import Path

def reorganize_eth3d(eth3d_root):
    """Reorganize ETH3D directory structure."""
    eth3d_root = Path(eth3d_root)

    # Source directory
    multi_view_dir = eth3d_root / 'multi_view_training_dslr_undistorted'

    if not multi_view_dir.exists():
        print(f"Error: {multi_view_dir} does not exist!")
        return False

    # Get all scene directories
    scenes = [d for d in multi_view_dir.iterdir() if d.is_dir()]

    print(f"Found {len(scenes)} scenes to process")
    print("=" * 60)

    for scene_src in sorted(scenes):
        scene_name = scene_src.name
        print(f"\nProcessing: {scene_name}")

        # Source: multi_view_training_dslr_undistorted/{scene}/images/dslr_images_undistorted/
        rgb_src = scene_src / 'images' / 'dslr_images_undistorted'

        # Destination: {scene}/images/dslr_images/
        scene_dst = eth3d_root / scene_name
        rgb_dst_parent = scene_dst / 'images'
        rgb_dst = rgb_dst_parent / 'dslr_images'

        # Check if source exists
        if not rgb_src.exists():
            print(f"  ⚠ RGB source not found: {rgb_src}")
            continue

        # Check if destination already has images (not a symlink)
        if rgb_dst.exists() and not rgb_dst.is_symlink():
            print(f"  ℹ Destination already exists (real directory): {rgb_dst}")
            # Count files
            num_files = len(list(rgb_dst.glob('*.JPG')))
            print(f"    Contains {num_files} JPG files")
            continue

        # Remove symlink if exists
        if rgb_dst.is_symlink():
            print(f"  🗑 Removing symlink: {rgb_dst}")
            rgb_dst.unlink()

        # Create parent directory
        rgb_dst_parent.mkdir(parents=True, exist_ok=True)

        # Move RGB images
        print(f"  📁 Source: {rgb_src}")
        print(f"  📁 Destination: {rgb_dst}")

        # Count files before move
        num_files = len(list(rgb_src.glob('*.JPG')))
        print(f"  📄 Moving {num_files} JPG files...")

        try:
            # Move the entire directory
            shutil.move(str(rgb_src), str(rgb_dst))
            print(f"  ✅ Success!")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue

    print("\n" + "=" * 60)
    print("Reorganization complete!")

    # Check if multi_view_training_dslr_undistorted is now empty (except calibration)
    remaining_items = []
    for scene_dir in multi_view_dir.iterdir():
        if scene_dir.is_dir():
            images_dir = scene_dir / 'images'
            if images_dir.exists():
                remaining = list(images_dir.iterdir())
                if remaining:
                    # Check if only dslr_calibration_undistorted remains
                    non_calib = [r for r in remaining if r.name != 'dslr_calibration_undistorted']
                    if non_calib:
                        remaining_items.append((scene_dir.name, non_calib))

    if remaining_items:
        print(f"\n⚠ Warning: Some items remain in multi_view_training_dslr_undistorted:")
        for scene, items in remaining_items:
            print(f"  {scene}: {[i.name for i in items]}")
    else:
        print(f"\n✅ multi_view_training_dslr_undistorted/*/images/ directories are clean!")
        print(f"   (Only calibration data remains, which we can keep or remove)")

    return True

if __name__ == '__main__':
    eth3d_root = '/home/cvlab/hsy/Datasets/eth3d'

    print("ETH3D Dataset Reorganization")
    print("=" * 60)
    print(f"Root directory: {eth3d_root}")
    print()

    # Confirm before proceeding
    response = input("This will MOVE RGB images from multi_view_training_dslr_undistorted.\nProceed? (yes/no): ")

    if response.lower() in ['yes', 'y']:
        reorganize_eth3d(eth3d_root)
    else:
        print("Cancelled.")
