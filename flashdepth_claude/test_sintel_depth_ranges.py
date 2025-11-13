#!/usr/bin/env python3
"""
Test script to check Sintel GT depth ranges and canonical transformation
"""
import os
import sys
import numpy as np
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dataloaders.sintel_dataset import SintelDepth

# Constants from combined_dataset.py
CANONICAL_FOCAL_LENGTH = 500.0
ACTUAL_MAX_DEPTH = 70.0

def apply_canonical_transform(inverse_depth_actual, fx_actual,
                               original_h, original_w,
                               target_resolution, resize_factor=1.0):
    """
    Apply Metric3D-style canonical transformation (same as combined_dataset.py)
    """
    # Convert to numpy
    if isinstance(inverse_depth_actual, torch.Tensor):
        inverse_np = inverse_depth_actual.cpu().numpy()
    else:
        inverse_np = inverse_depth_actual

    # Step 1: Apply dataset-specific pre-resize
    pre_h = int(original_h * resize_factor)
    pre_w = int(original_w * resize_factor)

    # Step 2: Compute small_resize_ratio (shorter side matches target)
    target_h, target_w = target_resolution
    small_resize_ratio = max(target_h / pre_h, target_w / pre_w)

    # Step 3: Focal length ratio
    fx_ratio = CANONICAL_FOCAL_LENGTH / fx_actual  # 500 / fx_actual

    # Step 4: Total resize ratio
    total_resize_ratio = resize_factor * small_resize_ratio

    # Step 5: Depth correction ratio
    depth_correction_ratio = total_resize_ratio / fx_ratio
    inverse_canonical_np = inverse_np * depth_correction_ratio

    print(f"\n=== Canonical Transform Calculation ===")
    print(f"Original resolution: {original_w}×{original_h}")
    print(f"Target resolution: {target_w}×{target_h}")
    print(f"fx_actual: {fx_actual:.2f} pixels")
    print(f"resize_factor: {resize_factor}")
    print(f"pre_h: {pre_h}, pre_w: {pre_w}")
    print(f"small_resize_ratio: {small_resize_ratio:.6f}")
    print(f"fx_ratio (500/fx_actual): {fx_ratio:.6f}")
    print(f"total_resize_ratio: {total_resize_ratio:.6f}")
    print(f"depth_correction_ratio (total/fx_ratio): {depth_correction_ratio:.6f}")
    print(f"======================================\n")

    return inverse_canonical_np, fx_ratio, total_resize_ratio, depth_correction_ratio


def main():
    # Dataset path
    data_root = "/home/cvlab/hsy/Datasets"
    sintel_root = os.path.join(data_root, "sintel/images/training/clean")

    if not os.path.exists(sintel_root):
        print(f"ERROR: Sintel dataset not found at {sintel_root}")
        return

    # Load Sintel validation dataset
    dataset = SintelDepth(root_dir=data_root, split='val', load_cache=None)

    print(f"Loaded Sintel validation dataset: {len(dataset.pairs)} sequences")
    print(f"Sequences to check: seq 0, 4, 7")

    # Target sequences
    target_seqs = [0, 4, 7]

    for seq_idx in target_seqs:
        if seq_idx >= len(dataset.pairs):
            print(f"\nWARNING: Sequence {seq_idx} not found (only {len(dataset.pairs)} sequences)")
            continue

        seq_pairs = dataset.pairs[seq_idx]
        if len(seq_pairs) == 0:
            print(f"\nWARNING: Sequence {seq_idx} is empty")
            continue

        # Get first frame
        first_pair = seq_pairs[0]
        scene_name = first_pair['scene_name']
        img_path = first_pair['image']
        depth_path = first_pair['depth']

        print(f"\n{'='*80}")
        print(f"Sequence {seq_idx}: {scene_name}")
        print(f"First frame: {os.path.basename(img_path)}")
        print(f"{'='*80}")

        # Read depth (returns inverse depth in 1/m)
        inverse_depth = dataset.depth_read(depth_path)

        # Get focal length
        from PIL import Image
        img = Image.open(img_path)
        original_h, original_w = img.size[1], img.size[0]  # PIL: (W, H)

        # Get focal length from camera file
        fx_actual = dataset.get_focal_length(first_pair, (original_h, original_w))

        print(f"\nOriginal image size: {original_w}×{original_h}")
        print(f"Focal length (fx_actual): {fx_actual:.2f} pixels")

        # Calculate normal depth (metric depth in meters)
        with np.errstate(divide='ignore', invalid='ignore'):
            normal_depth = np.where(inverse_depth > 0, 1.0 / inverse_depth, -1)

        # Valid pixels (exclude invalid=-1, sky=0)
        valid_mask = normal_depth > 0

        if valid_mask.sum() == 0:
            print("WARNING: No valid depth pixels found!")
            continue

        valid_depths = normal_depth[valid_mask]

        print(f"\n--- Normal GT Depth (Metric Depth in meters) ---")
        print(f"Valid pixels: {valid_mask.sum()} / {valid_mask.size} ({100*valid_mask.sum()/valid_mask.size:.1f}%)")
        print(f"Min depth: {valid_depths.min():.3f} m")
        print(f"Max depth: {valid_depths.max():.3f} m")
        print(f"Mean depth: {valid_depths.mean():.3f} m")
        print(f"Median depth: {np.median(valid_depths):.3f} m")

        # Depth distribution
        depth_ranges = [
            (0, 10, "0-10m"),
            (10, 20, "10-20m"),
            (20, 50, "20-50m"),
            (50, 70, "50-70m"),
            (70, 1000, "70-1000m"),
            (1000, float('inf'), ">1000m")
        ]

        print(f"\nDepth distribution:")
        for min_d, max_d, label in depth_ranges:
            count = ((valid_depths >= min_d) & (valid_depths < max_d)).sum()
            pct = 100 * count / valid_mask.sum()
            print(f"  {label:12s}: {count:8d} pixels ({pct:5.1f}%)")

        # Apply canonical transformation
        target_resolution = (1022, 434)  # Sintel validation resolution from combined_dataset.py
        resize_factor = 1.0  # Default

        inverse_canonical, fx_ratio, total_resize_ratio, depth_correction_ratio = apply_canonical_transform(
            inverse_depth, fx_actual, original_h, original_w, target_resolution, resize_factor
        )

        # Convert canonical inverse to metric depth
        with np.errstate(divide='ignore', invalid='ignore'):
            depth_canonical_metric = np.where(inverse_canonical > 0, 1.0 / inverse_canonical, -1)

        valid_canonical = depth_canonical_metric > 0
        valid_canonical_depths = depth_canonical_metric[valid_canonical]

        print(f"\n--- After Canonical Transform (Metric Depth) ---")
        print(f"Min depth: {valid_canonical_depths.min():.3f} m")
        print(f"Max depth: {valid_canonical_depths.max():.3f} m")
        print(f"Mean depth: {valid_canonical_depths.mean():.3f} m")

        # Show inverse depth values (what's actually stored)
        valid_inverse = inverse_depth[valid_mask]
        valid_inverse_canonical = inverse_canonical[valid_canonical]

        print(f"\n--- Inverse Depth (1/m) ---")
        print(f"Actual space - Min: {valid_inverse.min():.6f}, Max: {valid_inverse.max():.6f}")
        print(f"Canonical space - Min: {valid_inverse_canonical.min():.6f}, Max: {valid_inverse_canonical.max():.6f}")

        # What would be logged in training (inverse_100)
        inverse_100_canonical = inverse_canonical * 100.0
        valid_inverse_100 = inverse_100_canonical[valid_canonical]

        print(f"\n--- Inverse Depth × 100 (what's logged in training) ---")
        print(f"Min: {valid_inverse_100.min():.4f}")
        print(f"Max: {valid_inverse_100.max():.4f}")
        print(f"Mean: {valid_inverse_100.mean():.4f}")
        print(f"\nTo convert back to metric depth: 100 / inverse_100")
        print(f"Example: 100 / {valid_inverse_100.min():.4f} = {100/valid_inverse_100.min():.3f} m")
        print(f"Example: 100 / {valid_inverse_100.max():.4f} = {100/valid_inverse_100.max():.3f} m")


if __name__ == "__main__":
    main()
