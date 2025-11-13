#!/usr/bin/env python3
"""
Verify the canonicalization fix
"""
import os
import sys
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

CANONICAL_FOCAL_LENGTH = 500.0

def test_sintel():
    """Test Sintel (val-base) - should now be close to 1/fx_ratio"""
    print("="*80)
    print("TEST: Sintel (val-base)")
    print("="*80)

    original_w, original_h = 1024, 436
    fx_actual = 688.0
    target_resolution = (1022, 434)  # (W, H)
    resize_factor = 1.0

    print(f"Original: {original_w}×{original_h} (W×H)")
    print(f"Target resolution: {target_resolution} (W, H)")
    print(f"fx_actual: {fx_actual}")

    # Step 1: Pre-resize
    pre_h = int(original_h * resize_factor)
    pre_w = int(original_w * resize_factor)

    # Step 2: FIXED - Unpack correctly as (W, H)
    target_w, target_h = target_resolution
    small_resize_ratio = max(target_w / pre_w, target_h / pre_h)

    print(f"\nFIXED Calculation:")
    print(f"  target_w, target_h = {target_resolution}")
    print(f"  target_w = {target_w}, target_h = {target_h}")
    print(f"  small_resize_ratio = max({target_w}/{pre_w}, {target_h}/{pre_h})")
    print(f"                     = max({target_w/pre_w:.6f}, {target_h/pre_h:.6f})")
    print(f"                     = {small_resize_ratio:.6f}")

    # Step 3-6
    fx_ratio = CANONICAL_FOCAL_LENGTH / fx_actual
    total_resize_ratio = resize_factor * small_resize_ratio
    depth_correction = total_resize_ratio / fx_ratio
    expected = 1.0 / fx_ratio
    error = abs(depth_correction - expected)

    print(f"\nResults:")
    print(f"  fx_ratio = 500.0 / {fx_actual} = {fx_ratio:.6f}")
    print(f"  total_resize_ratio = {resize_factor} × {small_resize_ratio:.6f} = {total_resize_ratio:.6f}")
    print(f"  depth_correction = {total_resize_ratio:.6f} / {fx_ratio:.6f} = {depth_correction:.6f}")
    print(f"\nVerification:")
    print(f"  Expected (1/fx_ratio): {expected:.6f}")
    print(f"  Actual: {depth_correction:.6f}")
    print(f"  Error: {error:.6f}")

    if error < 0.01:
        print(f"  ✅ PASS - Error is minimal (resize is almost 1:1)")
    else:
        print(f"  ❌ FAIL - Error is still significant")

    return error < 0.01


def test_waymo_2k():
    """Test Waymo_seg (val-2k) - should now be close to 1/fx_ratio"""
    print("\n" + "="*80)
    print("TEST: Waymo_seg (val-2k)")
    print("="*80)

    original_w, original_h = 1920, 1280
    fx_actual = 2059.61
    target_resolution = (1918, 1274)  # (W, H)
    resize_factor = 1.0

    print(f"Original: {original_w}×{original_h} (W×H)")
    print(f"Target resolution: {target_resolution} (W, H)")
    print(f"fx_actual: {fx_actual}")

    # Step 1: Pre-resize
    pre_h = int(original_h * resize_factor)
    pre_w = int(original_w * resize_factor)

    # Step 2: FIXED - Unpack correctly as (W, H)
    target_w, target_h = target_resolution
    small_resize_ratio = max(target_w / pre_w, target_h / pre_h)

    print(f"\nFIXED Calculation:")
    print(f"  target_w, target_h = {target_resolution}")
    print(f"  target_w = {target_w}, target_h = {target_h}")
    print(f"  small_resize_ratio = max({target_w}/{pre_w}, {target_h}/{pre_h})")
    print(f"                     = max({target_w/pre_w:.6f}, {target_h/pre_h:.6f})")
    print(f"                     = {small_resize_ratio:.6f}")

    # Step 3-6
    fx_ratio = CANONICAL_FOCAL_LENGTH / fx_actual
    total_resize_ratio = resize_factor * small_resize_ratio
    depth_correction = total_resize_ratio / fx_ratio
    expected = 1.0 / fx_ratio
    error = abs(depth_correction - expected)

    print(f"\nResults:")
    print(f"  fx_ratio = 500.0 / {fx_actual:.2f} = {fx_ratio:.6f}")
    print(f"  total_resize_ratio = {resize_factor} × {small_resize_ratio:.6f} = {total_resize_ratio:.6f}")
    print(f"  depth_correction = {total_resize_ratio:.6f} / {fx_ratio:.6f} = {depth_correction:.6f}")
    print(f"\nVerification:")
    print(f"  Expected (1/fx_ratio): {expected:.6f}")
    print(f"  Actual: {depth_correction:.6f}")
    print(f"  Error: {error:.6f}")

    if error < 0.01:
        print(f"  ✅ PASS - Error is minimal (resize is almost 1:1)")
    else:
        print(f"  ❌ FAIL - Error is still significant")

    return error < 0.01


def main():
    print("\n" + "#"*80)
    print("# Canonicalization Fix Verification")
    print("#"*80 + "\n")

    test1 = test_sintel()
    test2 = test_waymo_2k()

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Sintel (val-base): {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"Waymo_seg (val-2k): {'✅ PASS' if test2 else '❌ FAIL'}")

    if test1 and test2:
        print("\n🎉 All tests passed! The bug is fixed.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")


if __name__ == "__main__":
    main()
