#!/usr/bin/env python3
"""
Recreate unreal4k dataset with proper downsampling (FAST VERSION with multiprocessing).

This script properly downsamples both images and disparity maps from unreal4k_original.
The disparity values are scaled proportionally to the image resolution change.

Original resolution: 3840×2160
Target resolution: 2112×1188
Scale factor: 0.55
"""

import os
import sys
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

# Target resolution
TARGET_HEIGHT = 1188
TARGET_WIDTH = 2112
SCALE_FACTOR = TARGET_WIDTH / 3840

def process_image(args):
    """Process a single image file."""
    img_path, target_img_path = args

    try:
        # Load with OpenCV (faster than PIL)
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)

        # Resize with high-quality bicubic interpolation
        img_resized = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_CUBIC)

        # Save
        cv2.imwrite(str(target_img_path), img_resized)
        return True
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return False

def process_disparity(args):
    """Process a single disparity file."""
    disp_path, target_disp_path = args

    try:
        # Load disparity
        disparity = np.load(disp_path)

        # Resize using bilinear interpolation
        disparity_resized = cv2.resize(
            disparity,
            (TARGET_WIDTH, TARGET_HEIGHT),
            interpolation=cv2.INTER_LINEAR
        )

        # Scale disparity values proportionally
        disparity_resized = disparity_resized * SCALE_FACTOR

        # Save
        np.save(target_disp_path, disparity_resized.astype(np.float32))
        return True
    except Exception as e:
        print(f"Error processing {disp_path}: {e}")
        return False

def main():
    # Paths
    original_root = Path("/home/cvlab/hsy/Datasets/unreal4k_original")
    target_root = Path("/home/cvlab/hsy/Datasets/unreal4k_fixed")

    print("="*80)
    print("Recreating Unreal4K Dataset with Proper Downsampling (FAST)")
    print("="*80)
    print(f"Original resolution: 3840×2160")
    print(f"Target resolution: {TARGET_WIDTH}×{TARGET_HEIGHT}")
    print(f"Scale factor: {SCALE_FACTOR:.4f}")
    print(f"Using {cpu_count()} CPU cores")
    print(f"Original path: {original_root}")
    print(f"Target path: {target_root}")
    print("="*80)

    # Create target root directory
    target_root.mkdir(parents=True, exist_ok=True)

    # Collect all tasks
    image_tasks = []
    disparity_tasks = []

    for seq_id in range(9):
        seq_name = f"UnrealStereo4K_0000{seq_id}"
        orig_seq_dir = original_root / seq_name
        target_seq_dir = target_root / seq_name

        if not orig_seq_dir.exists():
            print(f"⚠ Skipping {seq_name}: not found")
            continue

        # Create target directories
        (target_seq_dir / "Image0").mkdir(parents=True, exist_ok=True)
        (target_seq_dir / "Disp0").mkdir(parents=True, exist_ok=True)

        # Collect image tasks
        image_dir = orig_seq_dir / "Image0"
        target_image_dir = target_seq_dir / "Image0"

        if image_dir.exists():
            for img_file in sorted(os.listdir(image_dir)):
                if img_file.endswith('.png'):
                    image_tasks.append((
                        image_dir / img_file,
                        target_image_dir / img_file
                    ))

        # Collect disparity tasks
        disp_dir = orig_seq_dir / "Disp0"
        target_disp_dir = target_seq_dir / "Disp0"

        if disp_dir.exists():
            for disp_file in sorted(os.listdir(disp_dir)):
                if disp_file.endswith('.npy'):
                    disparity_tasks.append((
                        disp_dir / disp_file,
                        target_disp_dir / disp_file
                    ))

    print(f"\nTotal tasks:")
    print(f"  Images: {len(image_tasks)}")
    print(f"  Disparities: {len(disparity_tasks)}")

    # Process images with multiprocessing
    print(f"\nProcessing {len(image_tasks)} images...")
    with Pool(cpu_count()) as pool:
        list(tqdm(
            pool.imap(process_image, image_tasks),
            total=len(image_tasks),
            desc="Images"
        ))

    # Process disparities with multiprocessing
    print(f"\nProcessing {len(disparity_tasks)} disparity maps...")
    with Pool(cpu_count()) as pool:
        list(tqdm(
            pool.imap(process_disparity, disparity_tasks),
            total=len(disparity_tasks),
            desc="Disparities"
        ))

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
    print(f"  Expected (scaled): {orig_disp[1080, 1920] * SCALE_FACTOR:.4f}")

    # Verify depth calculation
    fx_orig = 1920.0
    fx_new = fx_orig * SCALE_FACTOR
    baseline = 0.5  # outdoor

    depth_orig = (baseline * fx_orig) / orig_disp[1080, 1920]
    depth_new = (baseline * fx_new) / new_disp[594, 1056]

    print(f"\nDepth verification (outdoor baseline=0.5m):")
    print(f"  Original: {depth_orig:.2f}m")
    print(f"  New: {depth_new:.2f}m")
    print(f"  Match: {'✓ PASS' if abs(depth_orig - depth_new) < 0.01 else '✗ FAIL'}")

    print("\n" + "="*80)
    print("Next steps:")
    print("  1. Backup old unreal4k: mv /home/cvlab/hsy/Datasets/unreal4k /home/cvlab/hsy/Datasets/unreal4k_old")
    print("  2. Use new dataset: mv /home/cvlab/hsy/Datasets/unreal4k_fixed /home/cvlab/hsy/Datasets/unreal4k")
    print("="*80)

if __name__ == "__main__":
    main()
