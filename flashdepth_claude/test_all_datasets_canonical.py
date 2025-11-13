#!/usr/bin/env python3
"""
Test canonical transform calculation for all validation datasets
"""
import os
import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Constants
CANONICAL_FOCAL_LENGTH = 500.0

def calculate_canonical_transform(dataset_name, original_w, original_h, fx_actual,
                                  target_resolution, resize_factor=1.0):
    """
    Calculate canonical transform exactly as in combined_dataset.py
    """
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*80}")
    print(f"Original resolution: {original_w}×{original_h} (W×H)")
    print(f"fx_actual: {fx_actual:.2f} pixels")
    print(f"target_resolution tuple: {target_resolution}")
    print(f"resize_factor: {resize_factor}")

    # Step 1: Pre-resize
    pre_h = int(original_h * resize_factor)
    pre_w = int(original_w * resize_factor)
    print(f"\nStep 1 - Pre-resize:")
    print(f"  pre_h = {original_h} × {resize_factor} = {pre_h}")
    print(f"  pre_w = {original_w} × {resize_factor} = {pre_w}")

    # Step 2: Unpack target_resolution (AS IN CODE)
    target_h, target_w = target_resolution
    print(f"\nStep 2 - Unpack target_resolution (AS IN CODE):")
    print(f"  target_h, target_w = {target_resolution}")
    print(f"  target_h = {target_h}")
    print(f"  target_w = {target_w}")

    # Compute small_resize_ratio (AS IN CODE - potentially wrong!)
    small_resize_ratio_code = max(target_h / pre_h, target_w / pre_w)
    print(f"\nStep 3 - small_resize_ratio (AS IN CODE):")
    print(f"  max(target_h / pre_h, target_w / pre_w)")
    print(f"  = max({target_h}/{pre_h}, {target_w}/{pre_w})")
    print(f"  = max({target_h/pre_h:.6f}, {target_w/pre_w:.6f})")
    print(f"  = {small_resize_ratio_code:.6f}")

    # CORRECT calculation (W→W, H→H)
    print(f"\nCORRECT calculation (assuming target_resolution is (W, H)):")
    print(f"  If target_resolution = (target_w_correct, target_h_correct)")
    print(f"  Then: target_w_correct = {target_resolution[0]}, target_h_correct = {target_resolution[1]}")
    print(f"  small_resize_ratio_correct = max(W'/W, H'/H)")
    print(f"                             = max({target_resolution[0]}/{pre_w}, {target_resolution[1]}/{pre_h})")
    print(f"                             = max({target_resolution[0]/pre_w:.6f}, {target_resolution[1]/pre_h:.6f})")
    small_resize_ratio_correct = max(target_resolution[0]/pre_w, target_resolution[1]/pre_h)
    print(f"                             = {small_resize_ratio_correct:.6f}")

    # Step 4: fx_ratio
    fx_ratio = CANONICAL_FOCAL_LENGTH / fx_actual
    print(f"\nStep 4 - fx_ratio:")
    print(f"  fx_ratio = {CANONICAL_FOCAL_LENGTH} / {fx_actual} = {fx_ratio:.6f}")

    # Step 5: total_resize_ratio (both versions)
    total_resize_ratio_code = resize_factor * small_resize_ratio_code
    total_resize_ratio_correct = resize_factor * small_resize_ratio_correct
    print(f"\nStep 5 - total_resize_ratio:")
    print(f"  CODE version: {resize_factor} × {small_resize_ratio_code:.6f} = {total_resize_ratio_code:.6f}")
    print(f"  CORRECT version: {resize_factor} × {small_resize_ratio_correct:.6f} = {total_resize_ratio_correct:.6f}")

    # Step 6: depth_correction_ratio (both versions)
    depth_correction_ratio_code = total_resize_ratio_code / fx_ratio
    depth_correction_ratio_correct = total_resize_ratio_correct / fx_ratio
    print(f"\nStep 6 - depth_correction_ratio:")
    print(f"  CODE version: {total_resize_ratio_code:.6f} / {fx_ratio:.6f} = {depth_correction_ratio_code:.6f}")
    print(f"  CORRECT version: {total_resize_ratio_correct:.6f} / {fx_ratio:.6f} = {depth_correction_ratio_correct:.6f}")
    print(f"\n  If CODE is correct: inverse_canonical = inverse_actual × {depth_correction_ratio_code:.6f}")
    print(f"  If CORRECT logic: inverse_canonical = inverse_actual × {depth_correction_ratio_correct:.6f}")

    # Compare: should be close to 1/fx_ratio if resize is minimal
    expected_ratio = 1.0 / fx_ratio
    print(f"\n  Expected (1/fx_ratio): {expected_ratio:.6f}")
    print(f"  CODE error: {abs(depth_correction_ratio_code - expected_ratio):.6f}")
    print(f"  CORRECT error: {abs(depth_correction_ratio_correct - expected_ratio):.6f}")

    return depth_correction_ratio_code, depth_correction_ratio_correct


def main():
    print("="*80)
    print("Testing Canonical Transform for All Validation Datasets")
    print("="*80)

    # Dataset configurations
    # Format: (name, original_w, original_h, fx_actual, target_resolution, resize_factor)
    datasets = [
        # Sintel (from combined_dataset.py line 109)
        ("sintel", 1024, 436, 688.0, (1022, 434), 1.0),

        # Waymo_seg (from combined_dataset.py line 107)
        # Assuming typical Waymo resolution 1920×1280
        ("waymo_seg (base)", 1920, 1280, 1000.0, (784, 518), 1.0),

        # Waymo_seg 2K (from combined_dataset.py line 129)
        ("waymo_seg (2K)", 1920, 1280, 1000.0, (1918, 1274), 1.0),

        # ETH3D (from combined_dataset.py line 107)
        # Assuming typical ETH3D resolution ~4032×3024
        ("eth3d (base)", 4032, 3024, 2800.0, (784, 518), 1.0),

        # ETH3D 2K (from combined_dataset.py line 129)
        ("eth3d (2K)", 4032, 3024, 2800.0, (1918, 1274), 1.0),

        # TartanAir (from combined_dataset.py line 115)
        # Original: 640×480, typical fx~320
        ("tartanair", 640, 480, 320.0, (518, 518), 1.0),
    ]

    for dataset_name, orig_w, orig_h, fx, target_res, resize_f in datasets:
        calculate_canonical_transform(
            dataset_name, orig_w, orig_h, fx, target_res, resize_f
        )

    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    print("\nKEY OBSERVATION:")
    print("If target_resolution is stored as (W, H) but code unpacks as (h, w),")
    print("then the calculation is WRONG for non-square resolutions!")
    print("\nFor validation datasets with minimal resize (like Sintel):")
    print("  - CORRECT version should give depth_correction ≈ 1/fx_ratio")
    print("  - CODE version gives completely different values!")
    print("\nNeed to check:")
    print("  1. How is target_resolution actually stored? (W, H) or (H, W)?")
    print("  2. Should code be: target_w, target_h = target_resolution?")


if __name__ == "__main__":
    main()
