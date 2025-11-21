import os
import numpy as np
import logging
from PIL import Image
from pathlib import Path
import torch

from nuscenes.nuscenes import NuScenes
# from nuscenes.utils.geometry_utils import get_cam_intrinsics # Not directly used in this dataloader
# from nuscenes.utils.data_classes import LidarPointCloud # Not directly used in this dataloader
# from nuscenes.utils.geometry_utils import transform_matrix # Not directly used in this dataloader
# from pyquaternion import Quaternion # Not directly used in this dataloader

from torch.utils.data import Dataset
from tqdm import tqdm


class NuscenesComparisonDataset(Dataset):
    """
    NuScenes dataset loader tailored for test_comparison.py and test_video_comparison.py.
    Provides original resolution images and metric depth.
    Each NuScenes scene is treated as one sequence.
    """
    def __init__(self,
                 data_root: str,
                 split: str,
                 dataset_name: str = 'nuscenes',
                 camera_name: str = 'CAM_FRONT',
                 output_depth_dir_name: str = 'depth_gt', # Name of the generated depth directory
                 min_depth: float = 1.0, # Min valid depth from generation script
                 max_depth: float = 70.0, # Max valid depth from generation script
                 limit_scenes=None # Limit number of scenes to process
                 ):
        self.root_dir = Path(data_root) # Expects /home/cvlab/hsy/Datasets/nuscenes
        self.split = split # Expects 'test'
        self.dataset_name = dataset_name
        self.camera_name = camera_name
        self.output_depth_dir = self.root_dir / output_depth_dir_name / self.camera_name # e.g., /nuscenes/depth_gt/CAM_FRONT
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.limit_scenes = limit_scenes # Store limit

        logging.info(f"Initializing {self.dataset_name} Comparison Dataset for split {self.split} at {self.root_dir}")
        logging.info(f"Loading images from: {self.root_dir / 'samples' / self.camera_name}")
        logging.info(f"Loading depths from: {self.output_depth_dir}")

        # Initialize NuScenes object for metadata and intrinsics
        # The nuscenes object needs to know the dataroot of its v1.0-test folder
        self.nusc = NuScenes(version=f"v1.0-{split}", dataroot=self.root_dir, verbose=False)

        # Build list of sequences, where each sequence is a list of frame data
        self.sequences = self._load_data_list()
        
        logging.info(f"Loaded {len(self.sequences)} NuScenes sequences for {self.split} comparison.")


    def _load_data_list(self):
        """
        Collects image, depth, and intrinsics paths, grouped by scene.
        Each scene becomes one sequence.
        """
        all_sequences = []

        # Iterate through each scene in the NuScenes object (already filtered by version)
        for scene_record in tqdm(self.nusc.scene, desc=f"Collecting NuScenes scenes for {self.split} split"):
            current_scene_frames = [] # List to store frame data for this scene

            # Iterate through samples in the current scene
            sample_token = scene_record['first_sample_token']
            while sample_token != '':
                sample = self.nusc.get('sample', sample_token)
                
                # Get CAM_FRONT sample_data
                cam_data_token = sample['data'][self.camera_name]
                cam_data = self.nusc.get('sample_data', cam_data_token)
                
                # Construct image and depth paths
                img_path = self.root_dir / cam_data['filename'] # e.g., /nuscenes/samples/CAM_FRONT/n008...jpg
                
                # Depth map filename is image filename with .jpg replaced by .png
                depth_filename = Path(cam_data['filename']).name.replace('.jpg', '.png')
                depth_path = self.output_depth_dir / depth_filename # e.g., /nuscenes/depth_gt/CAM_FRONT/n008...png

                # Retrieve intrinsics for this specific image
                calibrated_cam = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
                intrinsics_matrix = np.array(calibrated_cam['camera_intrinsic']) # Already checked it's 'camera_intrinsic'
                fx = intrinsics_matrix[0, 0]
                fy = intrinsics_matrix[1, 1]
                cx = intrinsics_matrix[0, 2]
                cy = intrinsics_matrix[1, 2]
                intrinsics = torch.tensor([fx, fy, cx, cy], dtype=torch.float32)

                # Ensure image and generated depth map exist
                if img_path.exists() and depth_path.exists():
                    current_scene_frames.append({
                        'token': cam_data_token, # For scene name in getitem
                        'timestamp': cam_data['timestamp'], # For sorting
                        'img_path': str(img_path),
                        'depth_path': str(depth_path),
                        'intrinsics': intrinsics
                    })
                else:
                    logging.warning(f"Missing file for {img_path.name} or {depth_path.name}. Skipping frame in scene {scene_record['name']}.")

                # Move to next sample in scene
                sample_token = sample['next']
            
            if len(current_scene_frames) > 0:
                # Sort frames by timestamp to ensure chronological order within the scene
                current_scene_frames = sorted(current_scene_frames, key=lambda x: x['timestamp'])

                # Append this list of frames as a single sequence
                all_sequences.append(current_scene_frames)
        
        # Apply limit_scenes if specified
        if self.limit_scenes is not None:
            logging.info(f"Limiting NuScenes scenes to first {self.limit_scenes}.")
            all_sequences = all_sequences[:self.limit_scenes]
            logging.info(f"Total {len(all_sequences)} NuScenes sequences after limiting.")

        return all_sequences


    def __len__(self):
        return len(self.sequences)


    def __getitem__(self, idx):
        # Get the full sequence (a list of frame_data dictionaries)
        sequence_frames_data = self.sequences[idx]

        images = []
        depths = []
        intrinsics_list = []
        # No object_wise for comparison dataset for now

        for frame_data in sequence_frames_data:
            img_path = frame_data['img_path']
            depth_path = frame_data['depth_path']
            intrinsics = frame_data['intrinsics']

            # Load RGB image
            img = Image.open(img_path).convert('RGB')
            img = np.array(img).astype(np.float32) / 255.0 # Normalize to [0, 1]
            img = torch.from_numpy(img).permute(2, 0, 1) # HWC to CHW

            # Load depth map (16-bit PNG, millimeters to meters)
            depth_img = Image.open(depth_path)
            depth_map = np.array(depth_img).astype(np.float32) / 1000.0 # Convert to meters
            
            # Mark invalid (0 or < min_depth or > max_depth)
            invalid_mask = (depth_map <= 0) | (depth_map < self.min_depth) | (depth_map > self.max_depth) | np.isnan(depth_map) | np.isinf(depth_map)
            depth_map[invalid_mask] = 0 # Mark invalid as 0 for consistency in evaluation

            depth_map = torch.from_numpy(depth_map).unsqueeze(0) # [1, H, W]

            images.append(img)
            depths.append(depth_map)
            intrinsics_list.append(intrinsics)

        # Stack all frames into sequence tensors
        images_tensor = torch.stack(images, dim=0) # [T, 3, H, W]
        depths_tensor = torch.stack(depths, dim=0) # [T, 1, H, W]
        intrinsics_tensor = torch.stack(intrinsics_list, dim=0) # [T, 4]

        # Return as a dictionary matching ComparisonDataset output format
        # Note: ComparisonDataset collate_fn will add batch dimension [1, T, ...]
        return {
            'images': images_tensor, # [T, 3, H, W]
            'depths': depths_tensor, # [T, 1, H, W]
            'intrinsics': intrinsics_tensor, # [T, 4]
            'focal_lengths': intrinsics_tensor[:, 0], # [T] - fx values
            'focal_lengths_actual': intrinsics_tensor[:, 0], # [T] - fx values
            'dataset_name': self.dataset_name,
            'scene_name': sequence_frames_data[0]['token'] # Use first frame's token as scene name for logging
        }