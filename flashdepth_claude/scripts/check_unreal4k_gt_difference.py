#!/usr/bin/env python3
"""
Check GT difference between test_comparison and test_gear5 for Unreal4K

This script verifies if the 200m clipping in test_gear5's Unreal4kDepth dataset
affects the GT values when evaluated at <70m range.
"""

import numpy as np
import sys
from pathlib import Path

def check_gt_difference():
    """Compare GT processing between two methods"""

    # Load first frame of first sequence
    gt_path = Path("/home/cvlab/hsy/Datasets/unreal4k/UnrealStereo4K_00000/Disp0/00000.npy")

    if not gt_path.exists():
        print(f"ERROR: GT file not found: {gt_path}")
        return False

    print("="*80)
    print("UNREAL4K GT COMPARISON: test_comparison vs test_gear5")
    print("="*80)
    print(f"Loading: {gt_path}")
    print()

    # Load original depth
    original_depth = np.load(gt_path)
    print(f"Original depth shape: {original_depth.shape}")
    print(f"Original depth range: [{original_depth.min():.2f}, {original_depth.max():.2f}] meters")
    print()

    # ========================================
    # Method 1: test_comparison (ComparisonDataset)
    # ========================================
    print("-"*80)
    print("Method 1: test_comparison (ComparisonDataset)")
    print("-"*80)

    # ComparisonDataset: no 200m clipping, only remove invalid
    comparison_depth = original_depth.copy()
    invalid_mask_comparison = np.logical_or.reduce((
        np.isinf(comparison_depth),
        np.isnan(comparison_depth),
        comparison_depth <= 0
    ))
    comparison_depth[invalid_mask_comparison] = 0

    valid_comparison = comparison_depth > 0
    print(f"Valid pixels: {valid_comparison.sum():,} / {valid_comparison.size:,} ({100*valid_comparison.sum()/valid_comparison.size:.1f}%)")
    print(f"Valid depth range: [{comparison_depth[valid_comparison].min():.2f}, {comparison_depth[valid_comparison].max():.2f}] meters")

    # Pixels in <70m range
    mask_70_comparison = (comparison_depth > 0) & (comparison_depth < 70)
    print(f"Pixels <70m: {mask_70_comparison.sum():,} ({100*mask_70_comparison.sum()/valid_comparison.sum():.1f}% of valid)")
    if mask_70_comparison.sum() > 0:
        print(f"  Mean depth: {comparison_depth[mask_70_comparison].mean():.2f}m")
        print(f"  Median depth: {np.median(comparison_depth[mask_70_comparison]):.2f}m")
        print(f"  Min/Max: [{comparison_depth[mask_70_comparison].min():.2f}, {comparison_depth[mask_70_comparison].max():.2f}]m")
    print()

    # ========================================
    # Method 2: test_gear5 (Unreal4kDepth)
    # ========================================
    print("-"*80)
    print("Method 2: test_gear5 (Unreal4kDepth with 200m clipping)")
    print("-"*80)

    # Unreal4kDepth: 200m clipping
    gear5_depth = original_depth.copy()
    MAX_VALID_DEPTH = 200.0
    invalid_mask_gear5 = np.logical_or.reduce((
        np.isinf(gear5_depth),
        np.isnan(gear5_depth),
        gear5_depth <= 0,
        gear5_depth > MAX_VALID_DEPTH  # 200m clipping
    ))

    # Simulate inverse depth conversion
    inverse_depth = np.zeros_like(gear5_depth)
    valid_mask_gear5 = ~invalid_mask_gear5
    inverse_depth[valid_mask_gear5] = 1.0 / gear5_depth[valid_mask_gear5]
    inverse_depth[invalid_mask_gear5] = -1

    # Convert back to metric depth (simulating test_gear5's reconstruction)
    # inverse_depth is in 1/m, but test_gear5 uses 100/m (inverse_100)
    inverse_100 = inverse_depth * 100.0  # Convert to 100/m

    # Reconstruct metric depth: depth = 100 / inverse_100
    reconstructed_depth = np.zeros_like(inverse_100)
    valid_inverse = inverse_100 > 0
    reconstructed_depth[valid_inverse] = 100.0 / inverse_100[valid_inverse]
    reconstructed_depth[~valid_inverse] = 0

    valid_gear5 = reconstructed_depth > 0
    print(f"Valid pixels: {valid_gear5.sum():,} / {valid_gear5.size:,} ({100*valid_gear5.sum()/valid_gear5.size:.1f}%)")
    print(f"Valid depth range: [{reconstructed_depth[valid_gear5].min():.2f}, {reconstructed_depth[valid_gear5].max():.2f}] meters")

    # Check 200m clipping impact
    pixels_clipped = invalid_mask_gear5.sum() - invalid_mask_comparison.sum()
    if pixels_clipped > 0:
        print(f"⚠️  Pixels removed by 200m clipping: {pixels_clipped:,}")

        # Find which pixels were clipped
        clipped_pixels = (original_depth > 200) & (original_depth < np.inf)
        if clipped_pixels.sum() > 0:
            print(f"    Depth range of clipped pixels: [{original_depth[clipped_pixels].min():.2f}, {original_depth[clipped_pixels].max():.2f}]m")

    # Pixels in <70m range
    mask_70_gear5 = (reconstructed_depth > 0) & (reconstructed_depth < 70)
    print(f"Pixels <70m: {mask_70_gear5.sum():,} ({100*mask_70_gear5.sum()/valid_gear5.sum():.1f}% of valid)")
    if mask_70_gear5.sum() > 0:
        print(f"  Mean depth: {reconstructed_depth[mask_70_gear5].mean():.2f}m")
        print(f"  Median depth: {np.median(reconstructed_depth[mask_70_gear5]):.2f}m")
        print(f"  Min/Max: [{reconstructed_depth[mask_70_gear5].min():.2f}, {reconstructed_depth[mask_70_gear5].max():.2f}]m")
    print()

    # ========================================
    # Comparison in <70m range
    # ========================================
    print("="*80)
    print("DIFFERENCE ANALYSIS (<70m range)")
    print("="*80)

    # Find common valid pixels in <70m range
    common_mask = mask_70_comparison & mask_70_gear5
    print(f"Common valid pixels (<70m): {common_mask.sum():,}")

    if common_mask.sum() > 0:
        comparison_vals = comparison_depth[common_mask]
        gear5_vals = reconstructed_depth[common_mask]

        # Compute differences
        abs_diff = np.abs(comparison_vals - gear5_vals)
        rel_diff = abs_diff / comparison_vals

        print(f"\nAbsolute difference (meters):")
        print(f"  Mean: {abs_diff.mean():.6f}m")
        print(f"  Median: {np.median(abs_diff):.6f}m")
        print(f"  Max: {abs_diff.max():.6f}m")
        print(f"  Std: {abs_diff.std():.6f}m")

        print(f"\nRelative difference (%):")
        print(f"  Mean: {100*rel_diff.mean():.4f}%")
        print(f"  Median: {100*np.median(rel_diff):.4f}%")
        print(f"  Max: {100*rel_diff.max():.4f}%")

        # Check if difference is significant
        threshold = 1e-3  # 1mm
        significant_diff = abs_diff > threshold
        if significant_diff.sum() > 0:
            print(f"\n⚠️  Pixels with significant difference (>1mm): {significant_diff.sum():,} ({100*significant_diff.sum()/common_mask.sum():.2f}%)")
        else:
            print(f"\n✓ No significant differences (all <1mm)")

        # Precision loss analysis
        print(f"\nPrecision loss from inverse depth conversion:")
        max_precision_loss = abs_diff.max()
        if max_precision_loss < 0.01:
            print(f"  ✓ Negligible (<1cm): max={max_precision_loss*1000:.2f}mm")
        elif max_precision_loss < 0.1:
            print(f"  ⚠️  Minor (<10cm): max={max_precision_loss*100:.2f}cm")
        else:
            print(f"  ❌ Significant (>10cm): max={max_precision_loss:.2f}m")

    # ========================================
    # Impact on delta_1 metric
    # ========================================
    print("\n" + "="*80)
    print("IMPACT ON DELTA_1 METRIC")
    print("="*80)

    # Simulate delta_1 calculation (threshold=1.25)
    # delta_1 = percentage of pixels where max(pred/gt, gt/pred) < 1.25

    # Assume a perfect prediction (pred = gt) for comparison
    if common_mask.sum() > 0:
        # Comparison method
        comparison_vals = comparison_depth[common_mask]
        ratio_comparison = np.maximum(
            comparison_vals / comparison_vals,
            comparison_vals / comparison_vals
        )  # Should be 1.0 everywhere
        delta1_comparison = (ratio_comparison < 1.25).mean()

        # Gear5 method (with precision loss)
        gear5_vals = reconstructed_depth[common_mask]
        ratio_gear5 = np.maximum(
            gear5_vals / comparison_vals,
            comparison_vals / gear5_vals
        )
        delta1_gear5 = (ratio_gear5 < 1.25).mean()

        print(f"Perfect prediction scenario (pred = gt_comparison):")
        print(f"  delta_1 (comparison GT): {delta1_comparison:.4f}")
        print(f"  delta_1 (gear5 GT):      {delta1_gear5:.4f}")
        print(f"  Difference: {abs(delta1_comparison - delta1_gear5):.6f}")

        if abs(delta1_comparison - delta1_gear5) < 1e-4:
            print(f"\n✓ No impact on delta_1 from GT precision loss")
        else:
            print(f"\n⚠️  GT precision loss may affect delta_1")

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    # Check for pixels >70m and <200m (these should NOT affect metrics)
    pixels_70_200 = (comparison_depth > 70) & (comparison_depth < 200)
    print(f"Pixels in 70-200m range: {pixels_70_200.sum():,}")
    if pixels_70_200.sum() > 0:
        print(f"  → These are excluded from metrics (MAX_DEPTH=70m)")
        print(f"  → 200m clipping should NOT affect them")

    # Check for pixels >200m (these are clipped in gear5)
    pixels_over_200 = comparison_depth > 200
    print(f"Pixels >200m: {pixels_over_200.sum():,}")
    if pixels_over_200.sum() > 0:
        print(f"  → These are clipped in gear5 (set to invalid)")
        print(f"  → But also excluded from metrics (MAX_DEPTH=70m)")
        print(f"  → 200m clipping should NOT affect metrics")

    print(f"\nFinal verdict:")
    if common_mask.sum() > 0 and abs_diff.max() < 0.01:
        print("✓ GT processing is effectively identical for <70m range")
        print("✓ 200m clipping does NOT affect metric computation")
        print("✓ Delta_1 difference must be due to MODEL PERFORMANCE, not GT")
    else:
        print("⚠️  GT processing differs in <70m range")
        print("⚠️  This may contribute to metric differences")

    return True


if __name__ == '__main__':
    success = check_gt_difference()
    sys.exit(0 if success else 1)
