#!/usr/bin/env python3
"""
Waymo Open Dataset Preprocessing Script

Converts Waymo tfrecord files to FlashDepth-compatible format:
- RGB images: JPEG format
- Depth: Sparse LiDAR projection in .npy format (N, 3) - [x, y, depth]

Based on official Waymo Open Dataset tutorials:
https://github.com/waymo-research/waymo-open-dataset
"""

import os
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
import argparse
from pathlib import Path

# Waymo Open Dataset imports
try:
    from waymo_open_dataset.utils import range_image_utils
    from waymo_open_dataset.utils import transform_utils
    from waymo_open_dataset.utils import frame_utils
    from waymo_open_dataset import dataset_pb2 as open_dataset
except ImportError:
    print("ERROR: waymo-open-dataset not installed!")
    print("Install with: pip install waymo-open-dataset-tf-2-12-0")
    exit(1)


def parse_camera_name(camera_name):
    """Convert camera enum to readable name"""
    camera_names = {
        open_dataset.CameraName.FRONT: 'FRONT',
        open_dataset.CameraName.FRONT_LEFT: 'FRONT_LEFT',
        open_dataset.CameraName.FRONT_RIGHT: 'FRONT_RIGHT',
        open_dataset.CameraName.SIDE_LEFT: 'SIDE_LEFT',
        open_dataset.CameraName.SIDE_RIGHT: 'SIDE_RIGHT',
    }
    return camera_names.get(camera_name, 'UNKNOWN')


def parse_range_image_and_camera_projection_fixed(frame):
    """
    Fixed version of frame_utils.parse_range_image_and_camera_projection
    that handles bytes/bytearray issue
    """
    range_images = {}
    camera_projections = {}
    seg_labels = {}
    range_image_top_pose = None

    for laser in frame.lasers:
        if len(laser.ri_return1.range_image_compressed) > 0:
            range_image_str_tensor = tf.io.decode_compressed(
                laser.ri_return1.range_image_compressed, 'ZLIB')
            ri = open_dataset.MatrixFloat()
            ri.ParseFromString(range_image_str_tensor.numpy())  # Fixed: removed bytearray()
            range_images[laser.name] = [ri]

            if laser.name == open_dataset.LaserName.TOP:
                range_image_top_pose_str_tensor = tf.io.decode_compressed(
                    laser.ri_return1.range_image_pose_compressed, 'ZLIB')
                range_image_top_pose = open_dataset.MatrixFloat()
                range_image_top_pose.ParseFromString(
                    range_image_top_pose_str_tensor.numpy())  # Fixed: removed bytearray()

            camera_projection_str_tensor = tf.io.decode_compressed(
                laser.ri_return1.camera_projection_compressed, 'ZLIB')
            cp = open_dataset.MatrixInt32()
            cp.ParseFromString(camera_projection_str_tensor.numpy())  # Fixed: removed bytearray()
            camera_projections[laser.name] = [cp]

        if len(laser.ri_return2.range_image_compressed) > 0:
            range_image_str_tensor = tf.io.decode_compressed(
                laser.ri_return2.range_image_compressed, 'ZLIB')
            ri = open_dataset.MatrixFloat()
            ri.ParseFromString(range_image_str_tensor.numpy())  # Fixed: removed bytearray()
            range_images[laser.name].append(ri)

            camera_projection_str_tensor = tf.io.decode_compressed(
                laser.ri_return2.camera_projection_compressed, 'ZLIB')
            cp = open_dataset.MatrixInt32()
            cp.ParseFromString(camera_projection_str_tensor.numpy())  # Fixed: removed bytearray()
            camera_projections[laser.name].append(cp)

    return range_images, camera_projections, seg_labels, range_image_top_pose


def build_camera_depth_from_lidar(frame, camera_name=open_dataset.CameraName.FRONT):
    """
    Project LiDAR points to camera image plane (Official Waymo method)

    Returns:
        depth_points: (N, 3) array of [x_pixel, y_pixel, depth_meters]
    """
    # Parse range images and camera projections (using fixed version)
    (range_images, camera_projections, _, range_image_top_pose) = \
        parse_range_image_and_camera_projection_fixed(frame)

    # Get the camera calibration for dimensions
    camera_calib = None
    for calib in frame.context.camera_calibrations:
        if calib.name == camera_name:
            camera_calib = calib
            break

    if camera_calib is None:
        return None

    h, w = camera_calib.height, camera_calib.width

    # Convert range images to point cloud
    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose,
        keep_polar_features=True
    )

    # Get points projected to this camera
    # cp_points contains [x_pixel, y_pixel, ...] for each range image
    points_all = []
    for i, (pts, cp_pts) in enumerate(zip(points, cp_points)):
        if cp_pts is None:
            continue

        # Filter points that project to this camera
        # cp_pts shape: [num_points, 6] where [..., 0] is camera index
        camera_mask = cp_pts[:, 0] == camera_name

        if camera_mask.sum() == 0:
            continue

        # Get pixel coordinates and depth
        x_pixels = cp_pts[camera_mask, 1]  # x coordinate
        y_pixels = cp_pts[camera_mask, 2]  # y coordinate

        # Get depth from point cloud (distance from camera)
        pts_filtered = pts[camera_mask]
        # Distance from origin (camera center)
        depths = np.linalg.norm(pts_filtered[:, :3], axis=1)

        # Stack into (N, 3) format
        points_filtered = np.stack([x_pixels, y_pixels, depths], axis=1)
        points_all.append(points_filtered)

    if len(points_all) == 0:
        return None

    # Concatenate all points
    depth_points = np.concatenate(points_all, axis=0)

    # Filter points within image bounds
    valid_mask = (
        (depth_points[:, 0] >= 0) & (depth_points[:, 0] < w) &
        (depth_points[:, 1] >= 0) & (depth_points[:, 1] < h) &
        (depth_points[:, 2] > 0)
    )
    depth_points = depth_points[valid_mask]

    return depth_points


def process_tfrecord(tfrecord_path, output_dir, camera_name=open_dataset.CameraName.FRONT):
    """
    Process a single tfrecord file

    Args:
        tfrecord_path: Path to .tfrecord file
        output_dir: Output directory
        camera_name: Which camera to extract (default: FRONT)
    """
    # Extract segment name from filename
    segment_name = Path(tfrecord_path).stem.replace('_with_camera_labels', '')

    camera_str = parse_camera_name(camera_name)

    # Create output directories
    rgb_dir = output_dir / segment_name / camera_str / 'rgb' / 'original'
    depth_dir = output_dir / segment_name / camera_str / 'depth'
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    # Read tfrecord
    dataset = tf.data.TFRecordDataset(str(tfrecord_path), compression_type='')

    frame_idx = 0
    for data in tqdm(dataset, desc=f"Processing {segment_name}"):
        frame = open_dataset.Frame()
        frame.ParseFromString(data.numpy())

        # Extract camera image
        camera_image = None
        for image in frame.images:
            if image.name == camera_name:
                camera_image = image
                break

        if camera_image is None:
            print(f"Warning: Camera {camera_str} not found in frame {frame_idx}")
            frame_idx += 1
            continue

        # Decode and save RGB image
        img = tf.image.decode_jpeg(camera_image.image).numpy()
        img_pil = Image.fromarray(img)
        img_path = rgb_dir / f"{frame_idx:04d}.jpg"
        img_pil.save(img_path, quality=95)

        # Build and save depth
        depth_points = build_camera_depth_from_lidar(frame, camera_name)

        if depth_points is not None and len(depth_points) > 0:
            depth_path = depth_dir / f"{frame_idx:04d}.npy"
            np.save(depth_path, depth_points.astype(np.float32))
        else:
            print(f"Warning: No depth points for frame {frame_idx}")

        frame_idx += 1

    print(f"Processed {frame_idx} frames for {segment_name}")
    return frame_idx


def process_single_file(args_tuple):
    """Wrapper for multiprocessing"""
    tfrecord_path, output_dir, camera_name = args_tuple
    try:
        return process_tfrecord(tfrecord_path, output_dir, camera_name)
    except Exception as e:
        print(f"ERROR processing {tfrecord_path}: {e}")
        import traceback
        traceback.print_exc()
        return 0


def main():
    parser = argparse.ArgumentParser(description='Preprocess Waymo tfrecords to FlashDepth format')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing .tfrecord files')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for processed data')
    parser.add_argument('--camera', type=str, default='FRONT',
                        choices=['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT'],
                        help='Which camera to extract (default: FRONT)')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum number of tfrecord files to process')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of parallel workers (default: 8)')

    args = parser.parse_args()

    # Convert camera name to enum
    camera_map = {
        'FRONT': open_dataset.CameraName.FRONT,
        'FRONT_LEFT': open_dataset.CameraName.FRONT_LEFT,
        'FRONT_RIGHT': open_dataset.CameraName.FRONT_RIGHT,
        'SIDE_LEFT': open_dataset.CameraName.SIDE_LEFT,
        'SIDE_RIGHT': open_dataset.CameraName.SIDE_RIGHT,
    }
    camera_name = camera_map[args.camera]

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Find all tfrecord files
    tfrecord_files = sorted(input_dir.glob('*.tfrecord'))

    if args.max_files:
        tfrecord_files = tfrecord_files[:args.max_files]

    print(f"Found {len(tfrecord_files)} tfrecord files")
    print(f"Output directory: {output_dir}")
    print(f"Camera: {args.camera}")
    print(f"Parallel workers: {args.num_workers}")
    print()

    # Process with multiprocessing if multiple files
    if len(tfrecord_files) > 1 and args.num_workers > 1:
        from multiprocessing import Pool

        # Prepare arguments
        process_args = [(f, output_dir, camera_name) for f in tfrecord_files]

        # Process in parallel
        with Pool(processes=args.num_workers) as pool:
            results = pool.map(process_single_file, process_args)

        total_frames = sum(results)
    else:
        # Single file or single worker - process sequentially
        total_frames = 0
        for tfrecord_path in tfrecord_files:
            try:
                num_frames = process_tfrecord(tfrecord_path, output_dir, camera_name)
                total_frames += num_frames
            except Exception as e:
                print(f"ERROR processing {tfrecord_path}: {e}")
                import traceback
                traceback.print_exc()

    print()
    print("="*60)
    print(f"Preprocessing complete!")
    print(f"Total files processed: {len(tfrecord_files)}")
    print(f"Total frames extracted: {total_frames}")
    print(f"Output directory: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
