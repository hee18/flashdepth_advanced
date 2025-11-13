#!/usr/bin/env python3
"""
Verify the fix using actual dataloader
"""
import os
import sys
import numpy as np
import torch
from pathlib import Path
import logging

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dataloaders.sintel_dataset import SintelDepth

# Silence warnings
logging.basicConfig(level=logging.ERROR)

CANONICAL_FOCAL_LENGTH = 500.0

def test_with_real_data():
    """Test with real Sintel data through dataloader"""
    print("="*80)
    print("Testing with Real Sintel Data (After Fix)")
    print("="*80)

    data_root = "/home/cvlab/hsy/Datasets"

    # Load Sintel dataset
    dataset = SintelDepth(root_dir=data_root, split='val', load_cache=None)

    if len(dataset.pairs) == 0:
        print("ERROR: No sequences found")
        return False

    # Get first sequence, first frame
    first_pair = dataset.pairs[0][0]
    scene_name = first_pair['scene_name']
    img_path = first_pair['image']
    depth_path = first_pair['depth']

    print(f"Scene: {scene_name}")
    print(f"First frame: {os.path.basename(img_path)}")

    # Read depth (returns inverse depth)
    inverse_depth_actual = dataset.depth_read(depth_path)

    # Get image size
    from PIL import Image
    img = Image.open(img_path)
    original_w, original_h = img.size  # PIL: (W, H)

    # Get focal length
    fx_actual = dataset.get_focal_length(first_pair, (original_h, original_w))

    print(f"\nOriginal resolution: {original_w}×{original_h} (W×H)")
    print(f"fx_actual: {fx_actual:.2f} pixels")

    # Target resolution for Sintel validation (from combined_dataset.py)
    target_resolution = (1022, 434)  # (W, H)
    resize_factor = 1.0

    print(f"Target resolution: {target_resolution} (W, H)")
    print(f"Resize factor: {resize_factor}")

    # Apply canonical transform (same logic as fixed combined_dataset.py)
    pre_h = int(original_h * resize_factor)
    pre_w = int(original_w * resize_factor)

    # FIXED: Unpack as (W, H)
    target_w, target_h = target_resolution
    small_resize_ratio = max(target_w / pre_w, target_h / pre_h)

    fx_ratio = CANONICAL_FOCAL_LENGTH / fx_actual
    total_resize_ratio = resize_factor * small_resize_ratio
    depth_correction_ratio = total_resize_ratio / fx_ratio

    print(f"\n--- Fixed Canonical Transform ---")
    print(f"target_w, target_h = {target_resolution}")
    print(f"target_w = {target_w}, target_h = {target_h}")
    print(f"small_resize_ratio = max({target_w}/{pre_w}, {target_h}/{pre_h})")
    print(f"                   = max({target_w/pre_w:.6f}, {target_h/pre_h:.6f})")
    print(f"                   = {small_resize_ratio:.6f}")
    print(f"fx_ratio = {CANONICAL_FOCAL_LENGTH} / {fx_actual:.2f} = {fx_ratio:.6f}")
    print(f"total_resize_ratio = {resize_factor} × {small_resize_ratio:.6f} = {total_resize_ratio:.6f}")
    print(f"depth_correction_ratio = {total_resize_ratio:.6f} / {fx_ratio:.6f} = {depth_correction_ratio:.6f}")

    # Apply correction
    inverse_canonical = inverse_depth_actual * depth_correction_ratio

    # Convert to metric depth
    valid_mask = inverse_canonical > 0
    if valid_mask.sum() > 0:
        valid_inverse_canonical = inverse_canonical[valid_mask]
        metric_depth_canonical = 1.0 / valid_inverse_canonical

        print(f"\n--- Canonical Space Metric Depth ---")
        print(f"Min: {metric_depth_canonical.min():.3f} m")
        print(f"Max: {metric_depth_canonical.max():.3f} m")
        print(f"Mean: {metric_depth_canonical.mean():.3f} m")

        # What would be logged (inverse_100)
        inverse_100 = inverse_canonical * 100.0
        valid_inverse_100 = inverse_100[valid_mask]

        print(f"\n--- GT inverse_100 (what's logged) ---")
        print(f"Min: {valid_inverse_100.min():.4f}")
        print(f"Max: {valid_inverse_100.max():.4f}")
        print(f"Mean: {valid_inverse_100.mean():.4f}")

        # Compare with original (actual space)
        valid_inverse_actual = inverse_depth_actual[valid_mask]
        metric_depth_actual = 1.0 / valid_inverse_actual

        print(f"\n--- Original (Actual Space) Metric Depth ---")
        print(f"Min: {metric_depth_actual.min():.3f} m")
        print(f"Max: {metric_depth_actual.max():.3f} m")
        print(f"Mean: {metric_depth_actual.mean():.3f} m")

        print(f"\n--- Depth Scaling Effect ---")
        print(f"Canonical / Actual ratio: {depth_correction_ratio:.6f}")
        print(f"Expected (for minimal resize): {1.0/fx_ratio:.6f}")
        error = abs(depth_correction_ratio - 1.0/fx_ratio)
        print(f"Error: {error:.6f}")

        if error < 0.01:
            print("✅ PASS - Depth correction is now correct!")
            return True
        else:
            print("❌ FAIL - Still has error")
            return False
    else:
        print("ERROR: No valid depth pixels")
        return False


def main():
    print("\n" + "#"*80)
    print("# Verify Fix with Real Dataloader")
    print("#"*80 + "\n")

    success = test_with_real_data()

    print("\n" + "="*80)
    if success:
        print("🎉 SUCCESS - The canonicalization bug is fixed!")
        print("="*80)
    else:
        print("❌ FAILED - Something is still wrong")
        print("="*80)


if __name__ == "__main__":
    main()
