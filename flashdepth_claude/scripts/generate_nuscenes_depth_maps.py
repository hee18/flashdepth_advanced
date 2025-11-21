import os
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import cv2

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

def generate_depth_maps(
    nuscenes_root: Path,
    version: str,
    camera_name: str,
    output_dir: Path,
    min_distance: float = 1.0,  # Minimum valid depth in meters
    max_distance: float = 70.0, # Maximum valid depth in meters
    lidar_scan_window: float = 0.01 # time window to consider LiDAR points (for single sweep)
):
    """
    Generates depth maps for NuScenes camera images by projecting LiDAR points.

    Args:
        nuscenes_root: Path to the NuScenes dataset root directory.
        version: NuScenes dataset version (e.g., 'v1.0-test').
        camera_name: Name of the camera to generate depth maps for (e.g., 'CAM_FRONT').
        output_dir: Directory to save the generated depth maps.
        min_distance: Minimum valid depth distance in meters.
        max_distance: Maximum valid depth distance in meters.
        lidar_scan_window: Time window in seconds around camera timestamp to consider LiDAR points.
                           For single sweep, keep this small (e.g., 0.01s).
    """
    print(f"Initializing NuScenes with version {version} from {nuscenes_root}...")
    nusc = NuScenes(version=version, dataroot=nuscenes_root, verbose=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory for depth maps: {output_dir}")

    # Get sensor token for the specified camera
    cam_token = None
    for sd_record in nusc.sample_data: # Iterate through records
        if sd_record['channel'] == camera_name:
            cam_token = sd_record['token'] # Get the token from the record
            break
    if not cam_token:
        print(f"Error: Camera '{camera_name}' not found in NuScenes metadata.")
        return

    # Filter samples for the specified version and camera
    sample_tokens_to_process = []
    for sample in nusc.sample:
        # Ensure it has data for the specified camera
        if sample['data'].get(camera_name):
            sample_tokens_to_process.append(sample['token'])
    
    print(f"Found {len(sample_tokens_to_process)} samples for {camera_name} in {version} split.")

    # Process each sample
    for sample_token in tqdm(sample_tokens_to_process, desc=f"Generating depth for {camera_name}"):
        sample = nusc.get('sample', sample_token)

        # Get camera data
        cam_data_token = sample['data'][camera_name]
        cam_data = nusc.get('sample_data', cam_data_token)
        cam_path = nusc.get_sample_data_path(cam_data_token)
        cam_timestamp = cam_data['timestamp']

        # Get calibrated sensor and ego pose for the camera
        calibrated_cam = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        cam_ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])

        # Get image dimensions
        img = Image.open(cam_path)
        img_w, img_h = img.size
        
        # Initialize depth map with zeros (invalid depth)
        depth_map = np.zeros((img_h, img_w), dtype=np.float32)

        # Get LIDAR data (FRONT_LIDAR is typically used)
        # We need the LIDAR token that is closest in time to the camera image
        # NuScenes sample['data'] provides tokens that are closest synchronized
        lidar_data_token = sample['data']['LIDAR_TOP'] # Assuming LIDAR_TOP is the primary LIDAR
        lidar_data = nusc.get('sample_data', lidar_data_token)
        
        # Load LIDAR points
        # LidarPointCloud.from_file handles decompression automatically
        pointsensor = nusc.get('sample_data', lidar_data_token)
        lidar_path = nusc.get_sample_data_path(lidar_data_token)
        
        cs_record_lidar = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        pc = LidarPointCloud.from_file(lidar_path)

        print(f"DEBUG: pc.points shape before transformations: {pc.points.shape}") # DEBUG
        
        # For precision, only consider the points from a single sweep synchronized with the camera
        # No accumulation from other sweeps, as per user's request.
        
        # 1. Transform lidar points from lidar frame to ego frame
        pc.rotate(Quaternion(cs_record_lidar['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record_lidar['translation']))
        
        # 2. Transform lidar points from ego frame to global frame
        ego_pose_lidar = nusc.get('ego_pose', pointsensor['ego_pose_token'])
        pc.rotate(Quaternion(ego_pose_lidar['rotation']).rotation_matrix)
        pc.translate(np.array(ego_pose_lidar['translation']))

        # 3. Transform lidar points from global frame to camera ego frame
        # (inverse of cam_ego_pose to get from global to camera's ego-pose frame)
        global_from_cam_ego = transform_matrix(cam_ego_pose['translation'], Quaternion(cam_ego_pose['rotation']),
                                                 inverse=False)
        cam_ego_from_global = np.linalg.inv(global_from_cam_ego)
        pc.transform(cam_ego_from_global)

        # 4. Transform lidar points from camera ego frame to camera sensor frame
        # (inverse of calibrated_cam to get from camera ego to camera sensor frame)
        cam_cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        cam_cs_from_cam_ego = transform_matrix(cam_cs_record['translation'], Quaternion(cam_cs_record['rotation']),
                                              inverse=False)
        cam_sensor_from_cam_ego = np.linalg.inv(cam_cs_from_cam_ego)
        pc.transform(cam_sensor_from_cam_ego)

        # 5. Remove points behind the camera
        points = pc.points # Get current points from LidarPointCloud object
        print(f"DEBUG: points shape before Z-filter: {points.shape}") # DEBUG
        points = points[:, points[2, :] > 0] # Keep points in front of camera (Z > 0)
        print(f"DEBUG: points shape after Z-filter: {points.shape}") # DEBUG

        if points.shape[1] == 0:
            tqdm.write(f"Skipping sample {sample_token}: No points in front of camera.")
            continue # Skip this sample if no points remain
        
        # 6. Project to image and filter points by distance
        # Intrinsics
        camera_intrinsics = np.array(cam_cs_record['camera_intrinsic'])
        
        # Project 3D points to 2D image plane
        points_uvz = camera_intrinsics @ points[:3, :] # (3,3) @ (3,N) = (3,N)
        points_uvz[0, :] /= points_uvz[2, :] # u = u/z
        points_uvz[1, :] /= points_uvz[2, :] # v = v/z
        
        # Extract u, v, depth (z)
        u_coords = points_uvz[0, :]
        v_coords = points_uvz[1, :]
        depth_values = points_uvz[2, :] # This is the Z-depth in camera frame (meters)

        # Filter points outside image boundaries and by depth range
        valid_idx = np.where(
            (u_coords >= 0) & (u_coords < img_w) &
            (v_coords >= 0) & (v_coords < img_h) &
            (depth_values >= min_distance) & (depth_values <= max_distance)
        )[0]
        
        u_coords_filtered = u_coords[valid_idx].astype(int)
        v_coords_filtered = v_coords[valid_idx].astype(int)
        depth_values_filtered = depth_values[valid_idx]
        
        # Fill depth map
        depth_map[v_coords_filtered, u_coords_filtered] = depth_values_filtered
        
        # Save depth map (16-bit PNG, depth in millimeters)
        # NuScenes format often uses image name as identifier
        output_filename = Path(cam_data['filename']).name.replace('.jpg', '.png').replace('.jpeg', '.png')
        save_path = output_dir / camera_name / output_filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert meters to millimeters for 16-bit PNG storage
        # Ensure it's unsigned 16-bit integer
        depth_map_mm = (depth_map * 1000).astype(np.uint16)
        Image.fromarray(depth_map_mm).save(save_path)


def main():
    parser = argparse.ArgumentParser(description="Generate depth maps for NuScenes camera images.")
    parser.add_argument("--nuscenes_root", type=str, default="/home/cvlab/hsy/Datasets/nuscenes",
                        help="Root directory of the NuScenes dataset.")
    parser.add_argument("--version", type=str, default="v1.0-test",
                        help="NuScenes dataset version (e.g., 'v1.0-test').")
    parser.add_argument("--camera_name", type=str, default="CAM_FRONT",
                        help="Name of the camera to generate depth maps for.")
    parser.add_argument("--output_dir", type=str, default="depth_gt",
                        help="Output directory relative to nuscenes_root to save depth maps.")
    parser.add_argument("--min_distance", type=float, default=1.0,
                        help="Minimum valid depth distance in meters.")
    parser.add_argument("--max_distance", type=float, default=70.0,
                        help="Maximum valid depth distance in meters.")
    
    args = parser.parse_args()

    # Resolve output directory
    output_path = Path(args.nuscenes_root) / args.output_dir

    generate_depth_maps(
        nuscenes_root=Path(args.nuscenes_root),
        version=args.version,
        camera_name=args.camera_name,
        output_dir=output_path,
        min_distance=args.min_distance,
        max_distance=args.max_distance
    )

if __name__ == "__main__":
    main()
