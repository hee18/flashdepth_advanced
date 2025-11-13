#!/usr/bin/env python3
"""
Verify fx_actual discrepancy between dataloader and _get_actual_focal_length
"""
import os
import sys
import torch
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dataloaders.combined_dataset import CombinedDataset
from utils.dataset_intrinsics import get_intrinsics_info, get_fallback_fx

def _get_actual_focal_length_replica(dataset_name, image_shape):
    """
    Replica of the function in test_gear5.py
    """
    if isinstance(dataset_name, str):
        dataset_name = dataset_name.lower().replace('-', '_')

    intrinsics_info = get_intrinsics_info(dataset_name)

    if intrinsics_info is None:
        width = image_shape[-1]
        fx = get_fallback_fx(width)
        print(f"  WARNING: No intrinsics for {dataset_name}, using fallback fx={fx:.1f}")
        return fx

    if intrinsics_info['type'] == 'fixed':
        return intrinsics_info['fx']

    if intrinsics_info['type'] == 'computed':
        if dataset_name in ['dynamicreplica', 'replica']:
            width = image_shape[-1]
            return width / 2.0
        else:
            width = image_shape[-1]
            return get_fallback_fx(width)

    if 'typical_fx' in intrinsics_info:
        return intrinsics_info['typical_fx']

    width = image_shape[-1]
    fx = get_fallback_fx(width)
    print(f"  WARNING: Could not determine fx for {dataset_name}, using fallback fx={fx:.1f}")
    return fx


def main():
    data_root = "/home/cvlab/hsy/Datasets"

    print("="*80)
    print("Testing fx_actual Discrepancy")
    print("="*80)

    # Test sintel and waymo_seg
    datasets_to_test = ['sintel', 'waymo_seg']

    for dataset_name in datasets_to_test:
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*80}")

        # Load dataset
        try:
            dataset = CombinedDataset(
                root_dir=data_root,
                enable_dataset_flags=[dataset_name],
                split='val',
                resolution='base',
                video_length=5
            )
        except Exception as e:
            print(f"ERROR loading dataset: {e}")
            continue

        if len(dataset) == 0:
            print("No data found")
            continue

        # Get first sequence
        sample = dataset[0]
        if sample is None:
            print("Sample is None")
            continue

        # Unpack
        images, gt_depth, fx_canonical, fx_actual_from_dataloader, actual_valid_mask, fx_ratio, resize_ratio, name = sample

        print(f"\n--- From Dataloader ---")
        print(f"fx_canonical: {fx_canonical[0].item():.2f}")
        print(f"fx_actual (frame 0): {fx_actual_from_dataloader[0].item():.2f}")
        print(f"fx_ratio (frame 0): {fx_ratio[0].item():.6f}")
        print(f"resize_ratio (frame 0): {resize_ratio[0].item():.6f}")

        # Compute fx_actual from fx_ratio
        fx_actual_from_ratio = 500.0 / fx_ratio[0].item()
        print(f"\nfx_actual from ratio (500/fx_ratio): {fx_actual_from_ratio:.2f}")

        # Get fx_actual using test_gear5.py's method
        image_shape = images.shape  # [T, C, H, W]
        fx_actual_from_typical = _get_actual_focal_length_replica(dataset_name, image_shape)

        print(f"\n--- From test_gear5.py method ---")
        print(f"fx_actual (typical): {fx_actual_from_typical:.2f}")

        # Compare
        print(f"\n--- Comparison ---")
        print(f"Dataloader fx_actual:     {fx_actual_from_dataloader[0].item():.2f}")
        print(f"From fx_ratio (correct):  {fx_actual_from_ratio:.2f}")
        print(f"From typical (test code): {fx_actual_from_typical:.2f}")

        diff_dataloader = abs(fx_actual_from_dataloader[0].item() - fx_actual_from_typical)
        diff_ratio = abs(fx_actual_from_ratio - fx_actual_from_typical)

        print(f"\nDiscrepancy (dataloader vs typical): {diff_dataloader:.2f}")
        print(f"Discrepancy (ratio vs typical):      {diff_ratio:.2f}")

        if diff_dataloader > 1.0 or diff_ratio > 1.0:
            print(f"⚠️  SIGNIFICANT DISCREPANCY DETECTED!")
            print(f"    This will cause incorrect de-canonicalization!")

            # Show impact on de_canonical_ratio
            CANONICAL_FX = 500.0
            de_canon_correct = CANONICAL_FX / fx_actual_from_ratio
            de_canon_wrong = CANONICAL_FX / fx_actual_from_typical

            print(f"\n--- Impact on de_canonical_ratio_inverse ---")
            print(f"Correct (from batch):  500 / {fx_actual_from_ratio:.2f} = {de_canon_correct:.6f}")
            print(f"Wrong (from typical):  500 / {fx_actual_from_typical:.2f} = {de_canon_wrong:.6f}")
            print(f"Ratio error: {abs(de_canon_correct - de_canon_wrong) / de_canon_correct * 100:.2f}%")

            # Show impact on predicted depth
            print(f"\n--- Example: If canonical pred_inverse = 10.0 (100/m) ---")
            pred_inv_canonical = 10.0
            pred_inv_correct = pred_inv_canonical * de_canon_correct
            pred_inv_wrong = pred_inv_canonical * de_canon_wrong
            pred_depth_correct = 100.0 / pred_inv_correct
            pred_depth_wrong = 100.0 / pred_inv_wrong

            print(f"Correct prediction: {pred_depth_correct:.2f} m")
            print(f"Wrong prediction:   {pred_depth_wrong:.2f} m")
            print(f"Error: {abs(pred_depth_correct - pred_depth_wrong):.2f} m ({abs(pred_depth_correct - pred_depth_wrong) / pred_depth_correct * 100:.1f}%)")
        else:
            print(f"✅ No significant discrepancy")

        # Test multiple frames
        print(f"\n--- Per-frame fx_actual (first 3 frames) ---")
        for i in range(min(3, len(fx_actual_from_dataloader))):
            print(f"Frame {i}: {fx_actual_from_dataloader[i].item():.2f}")

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("If significant discrepancy detected:")
    print("  → test_gear5.py should use batch['focal_lengths_actual']")
    print("  → OR compute from batch['fx_ratio']: fx_actual = 500 / fx_ratio")
    print("  → NOT use _get_actual_focal_length(dataset_name)")


if __name__ == "__main__":
    main()
