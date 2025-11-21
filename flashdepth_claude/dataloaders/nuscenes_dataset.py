import os
import numpy as np
import logging
from PIL import Image
from pathlib import Path
import torch
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes # Added import

from .base_dataset_pairs import BaseDatasetPairs


class NuScenesDepth(BaseDatasetPairs):
    """
    NuScenes dataset loader for metric depth estimation.

    Directory structure expected (after depth map generation):
    nuscenes/ (top_level_root_dir)
        samples/
            CAM_FRONT/
                n008-xxx.jpg
        depth_gt/
            CAM_FRONT/
                n008-xxx.png (16-bit, in mm)
        v1.0-test/ (metadata folder)
            category.json
            ...
    """
    def __init__(self, root_dir, split, load_cache=None, nusc_version='v1.0-test', limit_scenes=None, **kwargs):
        # root_dir points to parent Datasets directory, append 'nuscenes' subdirectory
        self.root_dir_nuscenes = Path(root_dir) / 'nuscenes'  # NuScenes directory (e.g., /data/datasets/nuscenes)
        self.split = split # e.g., 'test'
        self.nusc_version = nusc_version # e.g., 'v1.0-test'
        self.limit_scenes = limit_scenes # Limit number of scenes to process

        # Define paths BEFORE super().__init__() because get_filter_scenes() will use them
        self.rgb_root_dir = self.root_dir_nuscenes / 'samples' / 'CAM_FRONT'
        self.depth_root_dir = self.root_dir_nuscenes / 'depth_gt' / 'CAM_FRONT'

        logging.info(f"Initializing NuScenesDepth for split {self.split} at {self.root_dir_nuscenes}")

        super().__init__(dataset_name='nuscenes', root_dir=self.root_dir_nuscenes, split=split, load_cache=load_cache, **kwargs)

        # NuScenes CAM_FRONT original resolution: 1600x900
        self.orig_resolution = (1600, 900)

        # Remove default resolution as canonicalization will handle it
        if 'resolution' in self.reshape_list:
            del self.reshape_list['resolution']


    def depth_read(self, path, return_torch=False, **kwargs):
        """
        Read depth from NuScenes 16-bit PNG depth file.
        Depth is stored in millimeters, convert to meters, then to inverse depth.
        Invalid pixels are marked as -1.
        """
        # Load 16-bit PNG
        depth_img = Image.open(path)
        depth_map = np.array(depth_img).astype(np.float32)

        # Convert from millimeters to meters
        depth_map = depth_map / 1000.0

        # Mark invalid pixels as -1 (0 depth, NaN, Inf)
        invalid_mask = (depth_map <= 0) | np.isnan(depth_map) | np.isinf(depth_map)
        depth_map[invalid_mask] = -1

        # Convert to inverse depth (for Gear5)
        inverse_depth = np.zeros_like(depth_map)
        valid_pixels = depth_map > 0
        inverse_depth[valid_pixels] = 1.0 / depth_map[valid_pixels]
        inverse_depth[invalid_mask] = -1  # Re-mark invalid pixels for inverse depth as well

        return inverse_depth

    def get_cache_path(self, cache_dir):
        return os.path.join(cache_dir, f'nuscenes_{self.split}_pairs.pkl')

    def _build_pairs(self):
        """
        Override BaseDatasetPairs._build_pairs() to handle NuScenes' custom structure.
        NuScenes data is organized differently, so we build pairs directly from the API.
        """
        # Initialize NuScenes object
        nusc = NuScenes(version=self.nusc_version, dataroot=self.root_dir_nuscenes, verbose=False)
        
        logging.info(f"[DEBUG nuscenes_dataset] Building pairs for NuScenes, split={self.split}")
        
        # Iterate through each scene
        scene_count = 0
        for scene_record in tqdm(nusc.scene, desc=f"Collecting NuScenes scenes for {self.split} split"):
            current_scene_frames = []
            
            # Iterate through samples in the current scene
            sample_token = scene_record['first_sample_token']
            while sample_token != '':
                sample = nusc.get('sample', sample_token)
                
                # Get CAM_FRONT sample_data
                cam_data_token = sample['data']['CAM_FRONT']
                cam_data = nusc.get('sample_data', cam_data_token)
                
                # Construct image and depth paths
                img_path = self.root_dir_nuscenes / cam_data['filename']
                
                # Depth map filename is image filename with .jpg replaced by .png
                depth_filename = Path(cam_data['filename']).name.replace('.jpg', '.png')
                depth_path = self.depth_root_dir / depth_filename
                
                # Ensure image and generated depth map exist
                if img_path.exists() and depth_path.exists():
                    current_scene_frames.append({
                        'img_path': str(img_path),
                        'depth_path': str(depth_path),
                        'img_name': img_path.name,
                        'depth_name': depth_filename,
                        'timestamp': cam_data['timestamp']
                    })
                else:
                    logging.warning(f"Missing file for {img_path.name} or {depth_path.name}. Skipping.")
                
                # Move to next sample in scene
                sample_token = sample['next']
            
            if len(current_scene_frames) > 0:
                # Sort samples by timestamp to ensure chronological order within the scene
                current_scene_frames = sorted(current_scene_frames, key=lambda x: x['timestamp'])
                
                # Add scene name
                scene_name = scene_record['name']
                self.scenes.append(scene_name)
                
                # Build pair_dicts for this scene using create_pair_dict()
                scene_pairs = []
                for idx, frame_info in enumerate(current_scene_frames):
                    # Use BaseDatasetPairs.create_pair_dict() to ensure correct format
                    rgb_dir = os.path.dirname(frame_info['img_path'])
                    depth_dir = os.path.dirname(frame_info['depth_path'])
                    
                    pair_dict = self.create_pair_dict(
                        rgb_path=rgb_dir,
                        depth_path=depth_dir,
                        img_name=frame_info['img_name'],
                        depth_name=frame_info['depth_name'],
                        scene_index=idx,
                        scene_length=len(current_scene_frames),
                        scene_name=scene_name
                    )
                    scene_pairs.append(pair_dict)
                
                # For test/val, append the entire scene as one sequence
                if self.split != 'train':
                    self.pairs.append(scene_pairs)
                else:
                    # For training, add each frame as a separate pair
                    self.pairs.extend(scene_pairs)
                
                scene_count += 1
                
                # Apply limit_scenes if specified
                if self.limit_scenes is not None and scene_count >= self.limit_scenes:
                    logging.info(f"Limiting NuScenes scenes to first {self.limit_scenes}.")
                    break
        
        if len(self.scenes) == 0:
            raise RuntimeError(f"No valid scenes found in {self.root_dir_nuscenes}")
        
        self.scenes = sorted(self.scenes)
        logging.info(f'Number of scenes / length of {self.dataset_name}: {len(self.scenes)} / {len(self.pairs)}')


    def get_filter_scenes(self, split):
        """
        Not used for NuScenes. _build_pairs() handles scene collection directly.
        """
        return []


    # Methods required by BaseDatasetPairs (not used since _build_pairs is overridden)
    def get_rgb_depth_paths(self, scenes_path, scene_name):
        """Not used - _build_pairs() handles this directly."""
        raise NotImplementedError("NuScenes uses custom _build_pairs()")

    def get_sorted_image_files(self, rgb_path):
        """Not used - _build_pairs() handles this directly."""
        raise NotImplementedError("NuScenes uses custom _build_pairs()")

    def get_depth_name(self, img_name):
        """Not used - _build_pairs() handles this directly."""
        raise NotImplementedError("NuScenes uses custom _build_pairs()")

    def get_all_scenes(self, scenes_path):
        """Not used - _build_pairs() handles this directly."""
        return []

    def get_scenes_path(self):
        """Not used - _build_pairs() handles this directly."""
        return self.root_dir_nuscenes

    def get_focal_length(self, pair, image_shape):
        """
        Get focal length for NuScenes CAM_FRONT.
        
        NuScenes CAM_FRONT: fx=910.0 for 1600x900 resolution.
        This method returns the focal length scaled to the current image_shape.
        """
        original_fx = 910.0  # NuScenes CAM_FRONT intrinsics
        original_width = 1600  # NuScenes CAM_FRONT resolution
        
        # Scale to current image width
        current_width = image_shape[1]
        fx_scaled = original_fx * (current_width / original_width)
        return fx_scaled
