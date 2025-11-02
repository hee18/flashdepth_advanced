#!/usr/bin/env python3
"""
Waymo Open Dataset v2.0 Preprocessing Script

Converts Waymo parquet files to FlashDepth-compatible format:
- Extracts RGB images from camera_image parquet
- Extracts semantic segmentation from camera_segmentation parquet
- Copies depth from existing waymo/val dataset

Output structure:
    waymo_seg/val/
        segment-{context_name}/
            FRONT/
                rgb/original/*.jpg
                depth/*.npy (copied from waymo/val)
                segmentation/*.png
"""

import os
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
from pathlib import Path
import io
import shutil
from multiprocessing import Pool
import logging

# Check for required packages
try:
    import pyarrow.parquet as pq
    import pandas as pd
except ImportError:
    print("ERROR: pyarrow not installed!")
    print("Install with: pip install pyarrow pandas")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Camera name mapping (Waymo v2.0)
CAMERA_NAMES = {
    1: 'FRONT',
    2: 'FRONT_LEFT',
    3: 'FRONT_RIGHT',
    4: 'SIDE_LEFT',
    5: 'SIDE_RIGHT'
}


def process_sequence(context_name, parquet_root, output_root, camera_id=1):
    """
    Process a single Waymo sequence: extract RGB, segmentation, and depth from parquet.

    Args:
        context_name: Sequence context name (without .parquet extension)
        parquet_root: Root directory with parquet files
        output_root: Output directory (waymo_seg/val)
        camera_id: Camera to process (1=FRONT)

    Returns:
        True if successful, False otherwise
    """
    try:
        camera_str = CAMERA_NAMES.get(camera_id, 'FRONT')

        # Paths to parquet files
        image_file = parquet_root / 'camera_image' / f'{context_name}.parquet'
        seg_file = parquet_root / 'camera_segmentation' / f'{context_name}.parquet'

        if not image_file.exists():
            logger.error(f"Image file not found: {image_file}")
            return False

        if not seg_file.exists():
            logger.warning(f"Segmentation file not found: {seg_file}")
            return False

        # Load parquet files
        logger.info(f"Processing {context_name}...")

        img_table = pq.read_table(image_file)
        img_df = img_table.to_pandas()

        seg_table = pq.read_table(seg_file)
        seg_df = seg_table.to_pandas()

        # Filter by camera
        img_camera_df = img_df[img_df['key.camera_name'] == camera_id].reset_index(drop=True)
        seg_camera_df = seg_df[seg_df['key.camera_name'] == camera_id].reset_index(drop=True)

        num_frames = len(img_camera_df)

        if num_frames == 0:
            logger.warning(f"No frames found for camera {camera_id} in {context_name}")
            return False

        # Create output directories
        output_seq_dir = output_root / f'segment-{context_name}' / camera_str
        output_rgb_dir = output_seq_dir / 'rgb' / 'original'
        output_seg_dir = output_seq_dir / 'segmentation'
        output_depth_dir = output_seq_dir / 'depth'

        output_rgb_dir.mkdir(parents=True, exist_ok=True)
        output_seg_dir.mkdir(parents=True, exist_ok=True)
        output_depth_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Found {num_frames} frames for {camera_str} camera")

        # Process each frame
        for frame_idx in tqdm(range(num_frames), desc=f"{context_name[:20]}.../{camera_str}"):
            # Extract RGB
            img_row = img_camera_df.iloc[frame_idx]
            img_bytes = img_row['[CameraImageComponent].image']
            image = Image.open(io.BytesIO(img_bytes))

            # Save as JPEG
            output_path = output_rgb_dir / f'{frame_idx:04d}.jpg'
            image.save(output_path, 'JPEG', quality=95)

            # Extract segmentation
            if frame_idx < len(seg_camera_df):
                seg_row = seg_camera_df.iloc[frame_idx]
                seg_bytes = seg_row['[CameraSegmentationLabelComponent].panoptic_label']
                divisor = seg_row['[CameraSegmentationLabelComponent].panoptic_label_divisor']

                # Load panoptic label
                seg_img = Image.open(io.BytesIO(seg_bytes))
                panoptic_label = np.array(seg_img).astype(np.int64)

                # Extract semantic class: semantic_class = panoptic_label // divisor
                semantic_class = panoptic_label // divisor

                # Convert to uint8 (0-18 range for Waymo)
                semantic_class = semantic_class.astype(np.uint8)

                # Save as PNG
                output_path = output_seg_dir / f'{frame_idx:04d}.png'
                Image.fromarray(semantic_class).save(output_path)
            else:
                # Missing segmentation: create zero-filled
                logger.warning(f"Missing segmentation for frame {frame_idx}, using zeros")
                h, w = image.size[1], image.size[0]  # PIL uses (width, height)
                zero_seg = np.zeros((h, w), dtype=np.uint8)
                output_path = output_seg_dir / f'{frame_idx:04d}.png'
                Image.fromarray(zero_seg).save(output_path)

        # Extract depth from lidar_camera_projection parquet
        lidar_proj_file = parquet_root / 'lidar_camera_projection' / f'{context_name}.parquet'

        if not lidar_proj_file.exists():
            logger.warning(f"LiDAR projection file not found: {lidar_proj_file}")
            logger.warning("Creating zero-filled depth files")
            for frame_idx in range(num_frames):
                empty_depth = np.zeros((0, 3), dtype=np.float32)
                output_path = output_depth_dir / f'{frame_idx:04d}.npy'
                np.save(output_path, empty_depth)
        else:
            logger.info(f"Extracting depth from LiDAR camera projections...")

            # Load lidar_camera_projection parquet
            lidar_proj_table = pq.read_table(lidar_proj_file)
            lidar_proj_df = lidar_proj_table.to_pandas()

            # Filter by TOP laser (name=1)
            # Note: Each row contains projections for all cameras in 6-channel range image
            # Channels 0-2 are for FRONT camera (camera_id=1)
            lidar_top_df = lidar_proj_df[lidar_proj_df['key.laser_name'] == 1].reset_index(drop=True)

            if len(lidar_top_df) == 0:
                logger.warning(f"No LiDAR projections found for TOP laser")
                for frame_idx in range(num_frames):
                    empty_depth = np.zeros((0, 3), dtype=np.float32)
                    output_path = output_depth_dir / f'{frame_idx:04d}.npy'
                    np.save(output_path, empty_depth)
            else:
                # Determine channel offset for this camera
                # For FRONT (camera_id=1): use channels 0-2
                # For other cameras: would need different channel mapping
                if camera_id == 1:  # FRONT camera
                    channel_offset = 0
                else:
                    logger.warning(f"Camera {camera_id} channel mapping not implemented, using FRONT channels")
                    channel_offset = 0

                # Extract depth for each frame
                total_points = 0
                for frame_idx in range(num_frames):
                    if frame_idx >= len(lidar_top_df):
                        # No LiDAR data for this frame
                        empty_depth = np.zeros((0, 3), dtype=np.float32)
                        output_path = output_depth_dir / f'{frame_idx:04d}.npy'
                        np.save(output_path, empty_depth)
                        continue

                    row = lidar_top_df.iloc[frame_idx]

                    # Extract range image (6 channels)
                    ri_values = row['[LiDARCameraProjectionComponent].range_image_return1.values']
                    ri_shape = row['[LiDARCameraProjectionComponent].range_image_return1.shape']

                    # Reshape to (height, width, 6)
                    ri_array = np.array(ri_values).reshape(ri_shape)

                    # Extract channels for this camera
                    # Channel offset+0: range (scaled 1-5, multiply by 15 to get meters)
                    # Channel offset+1: x pixel coordinate (1-1919 for 1920x1280 image)
                    # Channel offset+2: y pixel coordinate (64-1279 for 1920x1280 image)
                    range_data = ri_array[:, :, channel_offset + 0] * 15.0  # Scale to meters (max ~75m)
                    x_pixels = ri_array[:, :, channel_offset + 1]
                    y_pixels = ri_array[:, :, channel_offset + 2]

                    # Filter valid points (all channels > 0)
                    valid_mask = (range_data > 0) & (x_pixels > 0) & (y_pixels > 0)

                    # Extract valid points
                    x_valid = x_pixels[valid_mask]
                    y_valid = y_pixels[valid_mask]
                    depth_valid = range_data[valid_mask]

                    # Stack to (N, 3) format: [x_pixel, y_pixel, depth_meters]
                    sparse_depth = np.stack([x_valid, y_valid, depth_valid], axis=1).astype(np.float32)

                    # Save sparse depth
                    output_path = output_depth_dir / f'{frame_idx:04d}.npy'
                    np.save(output_path, sparse_depth)

                    total_points += len(sparse_depth)

                avg_points = total_points / num_frames if num_frames > 0 else 0
                logger.info(f"Extracted depth for {num_frames} frames (avg {avg_points:.0f} points/frame)")

        logger.info(f"✓ Successfully processed {context_name} ({num_frames} frames)")
        return True

    except Exception as e:
        logger.error(f"✗ Error processing {context_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_sequence_wrapper(args):
    """Wrapper for multiprocessing."""
    return process_sequence(*args)


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess Waymo Open Dataset v2.0 (parquet format) for FlashDepth'
    )
    parser.add_argument(
        '--parquet-root',
        type=str,
        required=True,
        help='Root directory with parquet files (e.g., /Datasets/waymo_seg_parquet_backup)'
    )
    parser.add_argument(
        '--output-root',
        type=str,
        required=True,
        help='Output root directory (e.g., /Datasets/waymo_seg/val)'
    )
    parser.add_argument(
        '--camera',
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help='Camera to process (1=FRONT, 2=FRONT_LEFT, etc.)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of parallel workers'
    )
    parser.add_argument(
        '--max-sequences',
        type=int,
        default=None,
        help='Maximum number of sequences to process (for testing)'
    )

    args = parser.parse_args()

    # Set paths
    parquet_root = Path(args.parquet_root)
    output_root = Path(args.output_root)

    if not parquet_root.exists():
        logger.error(f"Parquet root not found: {parquet_root}")
        return

    # Create output directory
    output_root.mkdir(parents=True, exist_ok=True)

    # Get list of sequences from camera_image parquet files
    image_dir = parquet_root / 'camera_image'
    if not image_dir.exists():
        logger.error(f"camera_image directory not found: {image_dir}")
        return

    parquet_files = sorted(image_dir.glob('*.parquet'))
    context_names = [f.stem for f in parquet_files]

    logger.info(f"Found {len(context_names)} sequences")

    # Limit sequences if requested
    if args.max_sequences is not None:
        context_names = context_names[:args.max_sequences]
        logger.info(f"Processing first {len(context_names)} sequences")

    # Process sequences
    camera_str = CAMERA_NAMES.get(args.camera, 'FRONT')
    logger.info(f"Processing camera: {camera_str}")
    logger.info(f"Using {args.num_workers} workers")

    if args.num_workers > 1:
        # Parallel processing
        task_args = [
            (name, parquet_root, output_root, args.camera)
            for name in context_names
        ]

        with Pool(args.num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_sequence_wrapper, task_args),
                total=len(task_args),
                desc="Overall progress"
            ))
    else:
        # Sequential processing
        results = []
        for name in context_names:
            result = process_sequence(name, parquet_root, output_root, args.camera)
            results.append(result)

    # Summary
    success_count = sum(results)
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing complete!")
    logger.info(f"Successfully processed: {success_count}/{len(context_names)} sequences")
    logger.info(f"Output directory: {output_root}")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()
