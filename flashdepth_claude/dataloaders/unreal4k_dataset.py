import os
import cv2
import torch
import numpy as np
import logging
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from PIL import Image
import h5py
import torch.distributed as dist
import pickle
from .base_dataset_pairs import BaseDatasetPairs



class Unreal4kDepth(BaseDatasetPairs):
    def __init__(self, root_dir, split, load_cache=None):
        self.root_dir = os.path.join(root_dir, 'unreal4k')
        super().__init__(dataset_name='unreal4k', root_dir=self.root_dir, split=split, load_cache=load_cache)
        # Set default parameters (resized version at 0.55× scale)
        self.reshape_list['resolution'] = (2112, 1188)
        self.reshape_list['stride'] = 2

    def get_cache_path(self, cache_dir):
        return os.path.join(cache_dir, 'unreal4k_train_pairs.pkl')

    def get_all_scenes(self, scenes_path):
        all_scenes = [s for s in os.listdir(scenes_path) 
                     if os.path.isdir(os.path.join(scenes_path, s))]
        return sorted(all_scenes)

    def get_filter_scenes(self, split):
        all_scenes = self.get_all_scenes(self.root_dir)
        # if split == 'test':
        #     return all_scenes[3:]
        return []

    def get_rgb_depth_paths(self, scenes_path, scene_name):
        item_path = os.path.join(scenes_path, scene_name)
        return (os.path.join(item_path, 'Image0'),
                os.path.join(item_path, 'Disp0'))

    def get_sorted_image_files(self, rgb_path):
        all_imgs = [f for f in os.listdir(rgb_path) if f.endswith('.png')]
        all_imgs = sorted(all_imgs, key=lambda x: int(os.path.basename(x).split('.png')[0]))
        return all_imgs
        # if self.split == 'train':
        #     return all_imgs
        # else:
        #     return all_imgs[::50]  # Take every 50th frame

    def get_depth_name(self, img_name):
        return img_name.replace('.png', '.npy')

    def depth_read(self, path, is_inverse=False, return_torch=False, **kwargs):
        """
        Read depth from UnrealStereo4K.

        UnrealStereo4K stores DISPARITY maps in .npy files, not metric depth.
        We convert disparity to metric depth using:
            depth (m) = (baseline × focal_length) / disparity
        
        Baselines:
        - Indoor scenes (seq 4, 6): 0.2m (20cm)
        - Outdoor scenes (seq 0, 1, 2, 3, 5, 7, 8): 0.5m (50cm)
        
        Focal length (downsampled resolution 2112×1188): fx = 1056

        Args:
            path: Path to .npy disparity file
            is_inverse: If True, convert metric depth to inverse depth (1/m)
            return_torch: If True, return torch.Tensor

        Returns:
            Inverse depth (1/m) if is_inverse=True, else metric depth (m)
        """
        # Load disparity
        disparity = np.load(path)
        
        # Extract sequence ID from path
        # Path format: .../UnrealStereo4K_0000X/Disp0/XXXXX.npy
        seq_id = int(path.split('UnrealStereo4K_')[1].split('/')[0][-1])
        
        # Determine baseline based on sequence ID
        INDOOR_SEQS = [4, 6]
        BASELINE_INDOOR = 0.2   # 20cm
        BASELINE_OUTDOOR = 0.5  # 50cm
        
        baseline = BASELINE_INDOOR if seq_id in INDOOR_SEQS else BASELINE_OUTDOOR
        
        # Focal length for downsampled resolution (2112×1188)
        # Note: Original resolution (3840×2160) had fx=1920
        # Downsampled by factor 0.55, so fx = 1920 × 0.55 = 1056
        fx = 1056.0
        
        # Convert disparity to metric depth
        # depth = (baseline × fx) / disparity
        # Handle division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            depth_meters = (baseline * fx) / disparity
        
        # Handle invalid values
        MAX_VALID_DEPTH = 1000.0  # meters (outdoor scenes can be far)
        invalid_mask = np.logical_or.reduce((
            np.isinf(depth_meters),
            np.isnan(depth_meters),
            depth_meters <= 0,
            depth_meters > MAX_VALID_DEPTH,
            disparity <= 0
        ))

        if invalid_mask.any() and kwargs.get('print_minmax', False):
            logging.info(f"Found invalid values in {path}: "
                        f"inf: {np.isinf(depth_meters).sum()}, "
                        f"nan: {np.isnan(depth_meters).sum()}, "
                        f"<=0: {(depth_meters <= 0).sum()}, "
                        f">1000m: {(depth_meters > MAX_VALID_DEPTH).sum()}")

        if is_inverse:
            # Convert metric depth to inverse depth for training
            inverse_depth = np.zeros_like(depth_meters)
            valid_mask = ~invalid_mask
            inverse_depth[valid_mask] = 1.0 / depth_meters[valid_mask]
            inverse_depth[invalid_mask] = -1
            result = inverse_depth
        else:
            # Return metric depth
            depth_meters[invalid_mask] = -1
            result = depth_meters

        if return_torch:
            result = torch.from_numpy(result).float()

        return result

    def get_focal_length(self, pair, image_shape):
        """
        Get focal length for UnrealStereo4K dataset (resized version).

        Original UnrealStereo4K: fx=1920 for 3840×2160 resolution.
        Resized version (0.55×): fx=1056 for 2112×1188 resolution.
        This method returns the focal length scaled to the current image_shape.
        """
        original_fx = 1056.0  # Resized from 1920 at 0.55× scale
        original_width = 2112  # Resized from 3840 at 0.55× scale

        # Scale to current image width
        current_width = image_shape[1]
        fx_scaled = original_fx * (current_width / original_width)
        return fx_scaled
