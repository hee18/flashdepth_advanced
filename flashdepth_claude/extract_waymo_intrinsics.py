#!/usr/bin/env python3
"""
Extract Waymo intrinsics from parquet files and save to each sequence directory.

This script:
1. Reads camera_calibration/*.parquet files
2. Extracts fx, fy, cx, cy for FRONT camera
3. Saves intrinsics.npy to each sequence directory

This eliminates the need for pyarrow/fastparquet dependencies in inference code.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def extract_and_save_intrinsics(waymo_seg_root):
    """
    Extract intrinsics from parquet files and save to sequence directories.

    Args:
        waymo_seg_root: Path to waymo_seg root (e.g., /home/cvlab/hsy/Datasets/waymo_seg)
    """
    waymo_seg_root = Path(waymo_seg_root)

    # Path to calibration files
    calib_dir = waymo_seg_root / 'waymo_seg' / 'camera_calibration'

    if not calib_dir.exists():
        print(f"Error: Calibration directory not found: {calib_dir}")
        return

    # Get all parquet files
    parquet_files = sorted(calib_dir.glob('*.parquet'))
    print(f"Found {len(parquet_files)} calibration files")

    if len(parquet_files) == 0:
        print("No parquet files found!")
        return

    # Process val split
    val_dir = waymo_seg_root / 'val'
    if not val_dir.exists():
        print(f"Error: Val directory not found: {val_dir}")
        return

    # Statistics
    success_count = 0
    missing_count = 0
    error_count = 0

    for parquet_file in tqdm(parquet_files, desc="Processing calibration files"):
        # Extract sequence ID from filename (without .parquet)
        seq_id = parquet_file.stem

        # Find corresponding sequence directory in val/
        seq_name = f'segment-{seq_id}'
        seq_dir = val_dir / seq_name / 'FRONT'

        if not seq_dir.exists():
            missing_count += 1
            continue

        try:
            # Read parquet file
            calib_df = pd.read_parquet(parquet_file)

            # Try to extract intrinsics (try multiple field name formats)
            try:
                # New format with component prefix
                fx = float(calib_df['[CameraCalibrationComponent].intrinsic.f_u'].iloc[0])
                fy = float(calib_df['[CameraCalibrationComponent].intrinsic.f_v'].iloc[0])
                cx = float(calib_df['[CameraCalibrationComponent].intrinsic.c_u'].iloc[0])
                cy = float(calib_df['[CameraCalibrationComponent].intrinsic.c_v'].iloc[0])
            except KeyError:
                # Old format without prefix
                fx = float(calib_df['f_u'].iloc[0])
                fy = float(calib_df['f_v'].iloc[0])
                cx = float(calib_df['c_u'].iloc[0])
                cy = float(calib_df['c_v'].iloc[0])

            # Save to sequence directory as intrinsics.npy
            # Format: [fx, fy, cx, cy] for original 1920x1280 resolution
            intrinsics = np.array([fx, fy, cx, cy], dtype=np.float32)
            intrinsics_path = seq_dir / 'intrinsics.npy'
            np.save(intrinsics_path, intrinsics)

            success_count += 1

        except Exception as e:
            print(f"\nError processing {parquet_file.name}: {e}")
            error_count += 1
            continue

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total parquet files: {len(parquet_files)}")
    print(f"  Successfully saved: {success_count}")
    print(f"  Sequence not found: {missing_count}")
    print(f"  Errors: {error_count}")
    print(f"{'='*60}")

    if success_count > 0:
        print(f"\nIntrinsics saved to: {val_dir}/segment-*/FRONT/intrinsics.npy")
        print(f"Format: [fx, fy, cx, cy] for original 1920×1280 resolution")


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        waymo_seg_root = sys.argv[1]
    else:
        waymo_seg_root = '/home/cvlab/hsy/Datasets/waymo_seg'

    print(f"Waymo Segmentation Root: {waymo_seg_root}")
    extract_and_save_intrinsics(waymo_seg_root)
