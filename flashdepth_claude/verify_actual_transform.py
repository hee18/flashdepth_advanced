#!/usr/bin/env python3
"""
Verify actual canonical transform by loading data through combined_dataset
"""
import os
import sys
import numpy as np
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dataloaders.combined_dataset import CombinedDataset

def main():
    data_root = "/home/cvlab/hsy/Datasets"

    # Create validation dataset (base resolution)
    print("Loading validation datasets through CombinedDataset...")
    dataset = CombinedDataset(
        root_dir=data_root,
        enable_dataset_flags=['sintel', 'waymo_seg'],
        split='val',
        resolution='base',
        video_length=5
    )

    print(f"Total validation sequences: {len(dataset)}")

    # Get first sintel sequence
    print("\n" + "="*80)
    print("Testing SINTEL sequence")
    print("="*80)

    # Find first sintel sequence
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if sample is None:
            continue

        dataset_idx = sample[-1].item()  # Last element is dataset_idx
        if dataset_idx == 0:  # Assuming sintel is first in val_datasets
            print(f"\nFound Sintel sequence at index {idx}")

            # Unpack sample (format from combined_dataset.py __getitem__)
            images, gt_depth, fx_canonical, fx_actual, actual_valid_mask, fx_ratio, resize_ratio, dataset_idx_tensor = sample

            print(f"Images shape: {images.shape}")
            print(f"GT depth (inverse canonical) shape: {gt_depth.shape}")
            print(f"fx_canonical: {fx_canonical[0].item():.2f}")
            print(f"fx_actual: {fx_actual[0].item():.2f}")
            print(f"fx_ratio (500/fx_actual): {fx_ratio[0].item():.6f}")
            print(f"resize_ratio (total): {resize_ratio[0].item():.6f}")

            # Calculate depth_correction_ratio as code does
            depth_correction_from_data = resize_ratio[0].item() / fx_ratio[0].item()
            print(f"\ndepth_correction_ratio (from data): {depth_correction_from_data:.6f}")
            print(f"Expected (1/fx_ratio): {1.0/fx_ratio[0].item():.6f}")
            print(f"Difference: {abs(depth_correction_from_data - 1.0/fx_ratio[0].item()):.6f}")

            # Check GT depth values (inverse canonical)
            gt_inverse_canonical = gt_depth[0].numpy()  # First frame
            valid_mask = gt_inverse_canonical > 0

            if valid_mask.sum() > 0:
                valid_inverse = gt_inverse_canonical[valid_mask]
                print(f"\nGT inverse depth (canonical space) statistics:")
                print(f"  Min: {valid_inverse.min():.6f}")
                print(f"  Max: {valid_inverse.max():.6f}")
                print(f"  Mean: {valid_inverse.mean():.6f}")

                # Convert to metric depth (canonical space)
                metric_depth_canonical = 1.0 / valid_inverse
                print(f"\nMetric depth (canonical space) statistics:")
                print(f"  Min: {metric_depth_canonical.min():.3f} m")
                print(f"  Max: {metric_depth_canonical.max():.3f} m")
                print(f"  Mean: {metric_depth_canonical.mean():.3f} m")

                # What would be logged in training (× 100)
                inverse_100 = gt_inverse_canonical * 100.0
                valid_inverse_100 = inverse_100[valid_mask]
                print(f"\nGT inverse_100 (what's logged in training):")
                print(f"  Min: {valid_inverse_100.min():.4f}")
                print(f"  Max: {valid_inverse_100.max():.4f}")
                print(f"  Mean: {valid_inverse_100.mean():.4f}")

            break

    # Get first waymo_seg sequence
    print("\n" + "="*80)
    print("Testing WAYMO_SEG sequence")
    print("="*80)

    for idx in range(len(dataset)):
        sample = dataset[idx]
        if sample is None:
            continue

        dataset_idx = sample[-1].item()
        if dataset_idx == 1:  # Assuming waymo_seg is second in val_datasets
            print(f"\nFound Waymo_seg sequence at index {idx}")

            images, gt_depth, fx_canonical, fx_actual, actual_valid_mask, fx_ratio, resize_ratio, dataset_idx_tensor = sample

            print(f"Images shape: {images.shape}")
            print(f"GT depth (inverse canonical) shape: {gt_depth.shape}")
            print(f"fx_canonical: {fx_canonical[0].item():.2f}")
            print(f"fx_actual: {fx_actual[0].item():.2f}")
            print(f"fx_ratio (500/fx_actual): {fx_ratio[0].item():.6f}")
            print(f"resize_ratio (total): {resize_ratio[0].item():.6f}")

            depth_correction_from_data = resize_ratio[0].item() / fx_ratio[0].item()
            print(f"\ndepth_correction_ratio (from data): {depth_correction_from_data:.6f}")
            print(f"Expected (1/fx_ratio): {1.0/fx_ratio[0].item():.6f}")
            print(f"Difference: {abs(depth_correction_from_data - 1.0/fx_ratio[0].item()):.6f}")

            # Check GT depth values
            gt_inverse_canonical = gt_depth[0].numpy()
            valid_mask = gt_inverse_canonical > 0

            if valid_mask.sum() > 0:
                valid_inverse = gt_inverse_canonical[valid_mask]
                print(f"\nGT inverse depth (canonical space) statistics:")
                print(f"  Min: {valid_inverse.min():.6f}")
                print(f"  Max: {valid_inverse.max():.6f}")
                print(f"  Mean: {valid_inverse.mean():.6f}")

                metric_depth_canonical = 1.0 / valid_inverse
                print(f"\nMetric depth (canonical space) statistics:")
                print(f"  Min: {metric_depth_canonical.min():.3f} m")
                print(f"  Max: {metric_depth_canonical.max():.3f} m")
                print(f"  Mean: {metric_depth_canonical.mean():.3f} m")

            break


if __name__ == "__main__":
    main()
