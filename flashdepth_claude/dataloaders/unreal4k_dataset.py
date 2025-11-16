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
        self.root_dir = os.path.join(root_dir, 'unrealstereo4k')
        super().__init__(dataset_name='unreal4k', root_dir=self.root_dir, split=split, load_cache=load_cache)
        # Set default parameters
        self.reshape_list['resolution'] = (3840,2160)
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

    def depth_read(self, path, return_torch=False, **kwargs):
        # unrealstereo4k provides disparity maps, would need to use baseline and focal length to get depth for training,
        # but for evaluation we align the scale so it doesn't matter
        inverse_depth = np.load(path)

        invalid_mask = np.logical_or.reduce((
            np.isinf(inverse_depth),
            np.isnan(inverse_depth),
            inverse_depth < 0
        ))

        if invalid_mask.any():
            logging.info(f"Found invalid values in {path}: "
                        f"inf: {np.isinf(inverse_depth).sum()}, "
                        f"nan: {np.isnan(inverse_depth).sum()}, "
                        f"<0: {(inverse_depth < 0).sum()}")

        inverse_depth[invalid_mask] = -1

        if return_torch:
            inverse_depth = torch.from_numpy(inverse_depth).float()

        return inverse_depth

    def get_focal_length(self, pair, image_shape):
        """
        Get focal length for UnrealStereo4K dataset.

        UnrealStereo4K has intrinsics in Extrinsics0/*.txt files (first line).
        Format (line 1): fx skew cx 0 fy cy 0 0 1
        All scenes use the same intrinsics: fx=1920 for 3840x2160 resolution.

        Args:
            pair (dict): Data pair containing scene name
            image_shape (tuple): (H, W) image shape AFTER resizing

        Returns:
            float: Focal length in pixels for current image shape
        """
        # Try to read intrinsics from Extrinsics file
        scene_name = pair['scene']
        img_path = pair['rgb_path']
        img_name = os.path.basename(img_path)
        frame_idx = int(os.path.splitext(img_name)[0])

        extrinsics_dir = os.path.join(self.root_dir, scene_name, 'Extrinsics0')
        extrinsics_file = os.path.join(extrinsics_dir, f'{frame_idx:05d}.txt')

        original_fx = 1920.0  # Default fallback
        original_width = 3840

        if os.path.exists(extrinsics_file):
            try:
                with open(extrinsics_file, 'r') as f:
                    # Parse first line: fx skew cx 0 fy cy 0 0 1
                    k_values = list(map(float, f.readline().split()))
                    if len(k_values) >= 9:
                        original_fx = k_values[0]  # fx from K matrix
            except Exception as e:
                logging.warning(f"Could not read intrinsics from {extrinsics_file}: {e}")

        # Scale to current image width
        current_width = image_shape[1]
        fx_scaled = original_fx * (current_width / original_width)
        return fx_scaled
