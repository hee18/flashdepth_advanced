#!/usr/bin/env python3
"""
Recreate unreal4k dataset with proper downsampling.

This script properly downsamples both images and disparity maps from unreal4k_original.
The disparity values are scaled proportionally to the image resolution change.

Original resolution: 3840×2160
Target resolution: 2112×1188
Scale factor: 0.55
"""

import os
import sys
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
from tqdm import tqdm

def main():
    # Paths
    original_root = Path("/home/cvlab/hsy/Datasets/unreal4k_original")
    target_root = Path("/home/cvlab/hsy/Datasets/unreal4k_fixed")

    # Target resolution (same as current incorrect dataset)
    target_height = 1188
    target_width = 2112
    original_height = 2160
    original_width = 3840

    scale_factor = target_width / original_width

    print("="*80)
    print("Recreating Unreal4K Dataset with Proper Downsampling")
    print("="*80)
    print(f"Original resolution: {original_width}×{original_height}")
    print(f"Target resolution: {target_width}×{target_height}")
    print(f"Scale factor: {scale_factor:.4f}")
    print(f"Original path: {original_root}")
    print(f"Target path: {target_root}")
    print("="*80)

    # Create target root directory
    target_root.mkdir(parents=True, exist_ok=True)

    # Process each sequence
    for seq_id in range(9):
        seq_name = f"UnrealStereo4K_0000{seq_id}"
        orig_seq_dir = original_root / seq_name
        target_seq_dir = target_root / seq_name

        if not orig_seq_dir.exists():
            print(f"\n⚠ Skipping {seq_name}: not found")
            continue

        print(f"\n{'='*80}")
        print(f"Processing {seq_name}")
        print(f"{'='*80}")

        # Create target sequence directories
        (target_seq_dir / "Image0").mkdir(parents=True, exist_ok=True)
        (target_seq_dir / "Disp0").mkdir(parents=True, exist_ok=True)

        # Process images
        image_dir = orig_seq_dir / "Image0"
        target_image_dir = target_seq_dir / "Image0"

        if image_dir.exists():
            image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
            print(f"\nProcessing {len(image_files)} images...")

            for img_file in tqdm(image_files, desc="Images"):
                # Load original image
                img_path = image_dir / img_file
                img = Image.open(img_path)

                # Resize with high-quality bicubic interpolation
                img_resized = img.resize((target_width, target_height), Image.BICUBIC)

                # Save
                target_img_path = target_image_dir / img_file
                img_resized.save(target_img_path)

        # Process disparity maps
        disp_dir = orig_seq_dir / "Disp0"
        target_disp_dir = target_seq_dir / "Disp0"

        if disp_dir.exists():
            disp_files = sorted([f for f in os.listdir(disp_dir) if f.endswith('.npy')])
            print(f"\nProcessing {len(disp_files)} disparity maps...")

            for disp_file in tqdm(disp_files, desc="Disparities"):
                # Load original disparity
                disp_path = disp_dir / disp_file
                disparity = np.load(disp_path)

                # Resize disparity using bilinear interpolation
                disparity_resized = cv2.resize(
                    disparity,
                    (target_width, target_height),
                    interpolation=cv2.INTER_LINEAR
                )

                # IMPORTANT: Scale disparity values proportionally
                disparity_resized = disparity_resized * scale_factor

                # Save
                target_disp_path = target_disp_dir / disp_file
                np.save(target_disp_path, disparity_resized.astype(np.float32))

    print("\n" + "="*80)
    print("Dataset creation complete!")
    print("="*80)
    print(f"\nNew dataset saved to: {target_root}")
    print("\nVerifying a sample...")

    # Verify one sample
    sample_seq = "UnrealStereo4K_00000"
    sample_file = "00000.npy"

    orig_disp = np.load(original_root / sample_seq / "Disp0" / sample_file)
    new_disp = np.load(target_root / sample_seq / "Disp0" / sample_file)

    print(f"\nOriginal disparity:")
    print(f"  Shape: {orig_disp.shape}")
    print(f"  Sample value at (1080, 1920): {orig_disp[1080, 1920]:.4f}")

    print(f"\nNew disparity:")
    print(f"  Shape: {new_disp.shape}")
    print(f"  Sample value at (594, 1056): {new_disp[594, 1056]:.4f}")
    print(f"  Expected (scaled): {orig_disp[1080, 1920] * scale_factor:.4f}")

    # Verify depth calculation
    fx_orig = 1920.0
    fx_new = fx_orig * scale_factor
    baseline = 0.5  # outdoor

    depth_orig = (baseline * fx_orig) / orig_disp[1080, 1920]
    depth_new = (baseline * fx_new) / new_disp[594, 1056]

    print(f"\nDepth verification (outdoor baseline=0.5m):")
    print(f"  Original: {depth_orig:.2f}m")
    print(f"  New: {depth_new:.2f}m")
    print(f"  Match: {abs(depth_orig - depth_new) < 0.01} ✓" if abs(depth_orig - depth_new) < 0.01 else f"  Match: False ✗")

    print("\n" + "="*80)
    print("Next steps:")
    print("  1. Backup old unreal4k: mv /home/cvlab/hsy/Datasets/unreal4k /home/cvlab/hsy/Datasets/unreal4k_old")
    print("  2. Use new dataset: mv /home/cvlab/hsy/Datasets/unreal4k_fixed /home/cvlab/hsy/Datasets/unreal4k")
    print("="*80)

if __name__ == "__main__":
    main()
