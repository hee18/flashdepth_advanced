#!/usr/bin/env python3
"""
Test script to verify Unreal4K disparity-to-depth conversion is correct.
"""

import sys
import numpy as np
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataloaders.comparison_dataset import ComparisonDataset

def test_baseline_conversion():
    """Test that indoor/outdoor baselines are correctly applied."""

    # Sequence paths
    data_root = "/home/cvlab/hsy/Datasets"

    # Test outdoor sequence (seq 0, baseline=0.5m)
    outdoor_seq = "unreal4k/UnrealStereo4K_00000/Disp0/00000.npy"
    outdoor_path = str(Path(data_root) / outdoor_seq)

    # Test indoor sequence (seq 4, baseline=0.2m)
    indoor_seq = "unreal4k/UnrealStereo4K_00004/Disp0/00000.npy"
    indoor_path = str(Path(data_root) / indoor_seq)

    # Load raw disparity
    outdoor_disp = np.load(outdoor_path)
    indoor_disp = np.load(indoor_path)

    # Expected parameters
    fx = 1056.0  # Downsampled focal length
    baseline_outdoor = 0.5  # 50cm
    baseline_indoor = 0.2   # 20cm

    # Manual calculation
    outdoor_depth_manual = (baseline_outdoor * fx) / outdoor_disp
    indoor_depth_manual = (baseline_indoor * fx) / indoor_disp

    # Create dataset instance
    dataset = ComparisonDataset(
        dataset_name='unreal4k',
        data_root=data_root,
        split='test',
        video_length=50
    )

    # Load through dataset method
    outdoor_depth_dataset = dataset._load_unreal4k_depth(outdoor_path).numpy()
    indoor_depth_dataset = dataset._load_unreal4k_depth(indoor_path).numpy()

    # Compare (ignoring invalid values)
    outdoor_valid = (outdoor_disp > 0) & np.isfinite(outdoor_depth_manual)
    indoor_valid = (indoor_disp > 0) & np.isfinite(indoor_depth_manual)

    outdoor_diff = np.abs(outdoor_depth_manual[outdoor_valid] - outdoor_depth_dataset[outdoor_valid])
    indoor_diff = np.abs(indoor_depth_manual[indoor_valid] - indoor_depth_dataset[indoor_valid])

    print("="*80)
    print("Unreal4K Disparity-to-Depth Conversion Test")
    print("="*80)

    print("\nOutdoor Sequence (seq 0, baseline=0.5m):")
    print(f"  Disparity range: {outdoor_disp[outdoor_valid].min():.2f} - {outdoor_disp[outdoor_valid].max():.2f}")
    print(f"  Manual depth range: {outdoor_depth_manual[outdoor_valid].min():.2f}m - {outdoor_depth_manual[outdoor_valid].max():.2f}m")
    print(f"  Dataset depth range: {outdoor_depth_dataset[outdoor_valid].min():.2f}m - {outdoor_depth_dataset[outdoor_valid].max():.2f}m")
    print(f"  Max difference: {outdoor_diff.max():.6f}m")
    print(f"  Mean difference: {outdoor_diff.mean():.6f}m")

    print("\nIndoor Sequence (seq 4, baseline=0.2m):")
    print(f"  Disparity range: {indoor_disp[indoor_valid].min():.2f} - {indoor_disp[indoor_valid].max():.2f}")
    print(f"  Manual depth range: {indoor_depth_manual[indoor_valid].min():.2f}m - {indoor_depth_manual[indoor_valid].max():.2f}m")
    print(f"  Dataset depth range: {indoor_depth_dataset[indoor_valid].min():.2f}m - {indoor_depth_dataset[indoor_valid].max():.2f}m")
    print(f"  Max difference: {indoor_diff.max():.6f}m")
    print(f"  Mean difference: {indoor_diff.mean():.6f}m")

    # Verify baseline ratio
    # At same disparity value, outdoor depth should be 2.5x indoor depth (0.5/0.2 = 2.5)
    sample_disp = 100.0
    depth_outdoor_expected = (baseline_outdoor * fx) / sample_disp
    depth_indoor_expected = (baseline_indoor * fx) / sample_disp
    ratio = depth_outdoor_expected / depth_indoor_expected

    print(f"\nBaseline Ratio Verification:")
    print(f"  At disparity=100:")
    print(f"    Outdoor depth (baseline=0.5m): {depth_outdoor_expected:.2f}m")
    print(f"    Indoor depth (baseline=0.2m): {depth_indoor_expected:.2f}m")
    print(f"    Ratio (outdoor/indoor): {ratio:.2f}x (expected: 2.5x)")

    # Check if conversion is correct
    outdoor_pass = outdoor_diff.max() < 0.01 and outdoor_diff.mean() < 0.001
    indoor_pass = indoor_diff.max() < 0.01 and indoor_diff.mean() < 0.001
    ratio_pass = abs(ratio - 2.5) < 0.01

    print("\n" + "="*80)
    print("Test Results:")
    print("="*80)
    print(f"  Outdoor conversion: {'✓ PASS' if outdoor_pass else '✗ FAIL'}")
    print(f"  Indoor conversion: {'✓ PASS' if indoor_pass else '✗ FAIL'}")
    print(f"  Baseline ratio: {'✓ PASS' if ratio_pass else '✗ FAIL'}")
    print(f"\n  Overall: {'✓ ALL TESTS PASSED' if all([outdoor_pass, indoor_pass, ratio_pass]) else '✗ SOME TESTS FAILED'}")
    print("="*80)

    return all([outdoor_pass, indoor_pass, ratio_pass])

if __name__ == "__main__":
    success = test_baseline_conversion()
    sys.exit(0 if success else 1)
