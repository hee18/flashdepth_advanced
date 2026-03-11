#!/usr/bin/env python3
"""
Extract Waymo camera data from parquet files and save as .npz per sequence.

Extracts (for FRONT camera):
  - K: [3,3] intrinsic matrix (for original 1920x1280)
  - camera_to_vehicle: [4,4] extrinsic
  - poses: [N,4,4] camera-to-world poses (sorted by timestamp)

Saves to: waymo_seg/val/segment-{id}/FRONT/camera_data.npz

After extraction, parquet files can be deleted to save disk space.
This eliminates the pyarrow dependency for TAE computation.

Usage:
    python extract_waymo_cameras.py [waymo_seg_root]
    python extract_waymo_cameras.py /home/cvlab/hsy/Datasets/waymo_seg
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def extract_waymo_cameras(waymo_seg_root):
    waymo_seg_root = Path(waymo_seg_root)

    calib_dir = waymo_seg_root / 'waymo_seg' / 'camera_calibration'
    image_dir = waymo_seg_root / 'waymo_seg' / 'camera_image'
    val_dir = waymo_seg_root / 'val'

    for d, name in [(calib_dir, 'camera_calibration'), (image_dir, 'camera_image'), (val_dir, 'val')]:
        if not d.exists():
            print(f"Error: {name} directory not found: {d}")
            return

    parquet_files = sorted(calib_dir.glob('*.parquet'))
    print(f"Found {len(parquet_files)} calibration parquet files")

    if not parquet_files:
        print("No parquet files found!")
        return

    CAMERA_ID = 1  # FRONT

    success = 0
    missing = 0
    errors = 0

    for pf in tqdm(parquet_files, desc="Extracting camera data"):
        seq_id = pf.stem
        seq_dir = val_dir / f'segment-{seq_id}' / 'FRONT'

        if not seq_dir.exists():
            missing += 1
            continue

        try:
            # --- Calibration ---
            calib_df = pd.read_parquet(pf)
            cam_row = calib_df[calib_df['key.camera_name'] == CAMERA_ID].iloc[0]

            try:
                fx = float(cam_row['[CameraCalibrationComponent].intrinsic.f_u'])
                fy = float(cam_row['[CameraCalibrationComponent].intrinsic.f_v'])
                cx = float(cam_row['[CameraCalibrationComponent].intrinsic.c_u'])
                cy = float(cam_row['[CameraCalibrationComponent].intrinsic.c_v'])
                cam2veh = np.array(cam_row['[CameraCalibrationComponent].extrinsic.transform']).reshape(4, 4)
            except KeyError:
                fx = float(cam_row['f_u'])
                fy = float(cam_row['f_v'])
                cx = float(cam_row['c_u'])
                cy = float(cam_row['c_v'])
                cam2veh = np.array(cam_row['extrinsic.transform']).reshape(4, 4)

            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

            # --- Per-frame poses ---
            img_parquet = image_dir / f'{seq_id}.parquet'
            if not img_parquet.exists():
                print(f"\nWarning: camera_image parquet not found for {seq_id}")
                errors += 1
                continue

            img_df = pd.read_parquet(img_parquet)
            cam_df = img_df[img_df['key.camera_name'] == CAMERA_ID].sort_values('key.frame_timestamp_micros')

            num_frames = len(cam_df)
            poses = np.zeros((num_frames, 4, 4), dtype=np.float64)

            for frame_idx, (_, row) in enumerate(cam_df.iterrows()):
                veh2world = np.array(row['[CameraImageComponent].pose.transform']).reshape(4, 4)
                poses[frame_idx] = veh2world @ cam2veh  # camera-to-world

            # --- Save ---
            out_path = seq_dir / 'camera_data.npz'
            np.savez_compressed(out_path, K=K, camera_to_vehicle=cam2veh, poses=poses)

            # Also save intrinsics.npy for backward compatibility
            intrinsics_path = seq_dir / 'intrinsics.npy'
            np.save(intrinsics_path, np.array([fx, fy, cx, cy], dtype=np.float32))

            success += 1

        except Exception as e:
            print(f"\nError processing {seq_id}: {e}")
            errors += 1

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total parquet files: {len(parquet_files)}")
    print(f"  Successfully saved:  {success}")
    print(f"  Sequence not found:  {missing}")
    print(f"  Errors:              {errors}")
    print(f"{'='*60}")

    if success > 0:
        print(f"\nSaved to: {val_dir}/segment-*/FRONT/camera_data.npz")
        print(f"Contents: K [3,3], camera_to_vehicle [4,4], poses [N,4,4]")
        print(f"\nYou can now safely delete the parquet directories:")
        print(f"  rm -rf {calib_dir}")
        print(f"  rm -rf {image_dir}")


if __name__ == '__main__':
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else '/home/cvlab/hsy/Datasets/waymo_seg'
    print(f"Waymo Segmentation Root: {root}")
    extract_waymo_cameras(root)
