#!/usr/bin/env python3
"""
Verify ETH3D dataset structure after reorganization.
"""
import os
from pathlib import Path

def verify_eth3d(eth3d_root):
    eth3d_root = Path(eth3d_root)

    # Get all scene directories
    scenes = sorted([d for d in eth3d_root.iterdir() if d.is_dir()])

    print("ETH3D Dataset Structure Verification")
    print("=" * 60)
    print(f"Root: {eth3d_root}")
    print(f"Total scenes: {len(scenes)}\n")

    all_valid = True
    total_rgb = 0
    total_depth = 0

    for scene in scenes:
        scene_name = scene.name
        rgb_dir = scene / 'images' / 'dslr_images'
        depth_dir = scene / 'ground_truth_depth' / 'dslr_images'

        rgb_exists = rgb_dir.exists()
        depth_exists = depth_dir.exists()

        if rgb_exists:
            rgb_files = list(rgb_dir.glob('*.JPG'))
            rgb_count = len(rgb_files)
            # Check if any are symlinks
            has_symlinks = any(f.is_symlink() for f in rgb_files[:5])
            total_rgb += rgb_count
        else:
            rgb_count = 0
            has_symlinks = False

        if depth_exists:
            depth_count = len(list(depth_dir.glob('*.JPG')))
            total_depth += depth_count
        else:
            depth_count = 0

        status = "✅" if rgb_exists and depth_exists else "⚠"
        symlink_status = " [SYMLINK]" if has_symlinks else ""

        print(f"{status} {scene_name:20s} RGB: {rgb_count:3d}, Depth: {depth_count:3d}{symlink_status}")

        if not rgb_exists or not depth_exists:
            all_valid = False
            if not rgb_exists:
                print(f"    Missing: {rgb_dir}")
            if not depth_exists:
                print(f"    Missing: {depth_dir}")

    print("\n" + "=" * 60)
    print(f"Summary:")
    print(f"  Total scenes:      {len(scenes)}")
    print(f"  Total RGB files:   {total_rgb}")
    print(f"  Total depth files: {total_depth}")
    print(f"  Status:            {'✅ All valid' if all_valid else '⚠ Some issues found'}")

    # Check for old directory
    old_dir = eth3d_root / 'multi_view_training_dslr_undistorted'
    if old_dir.exists():
        print(f"\n⚠ Warning: {old_dir.name} still exists!")
    else:
        print(f"\n✅ Old directory (multi_view_training_dslr_undistorted) removed")

    return all_valid

if __name__ == '__main__':
    eth3d_root = '/home/cvlab/hsy/Datasets/eth3d'
    verify_eth3d(eth3d_root)
