# Check for endianness, based on Daniel Scharstein's optical flow code.
# Using little-endian architecture, these two should be equal.
TAG_FLOAT = 202021.25
TAG_CHAR = 'PIEH'

import os
import cv2
import torch
import numpy as np
import logging
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from PIL import Image
import torch.distributed as dist
from .base_dataset_pairs import BaseDatasetPairs

class SintelDepth(BaseDatasetPairs):
    def __init__(self, root_dir, split, load_cache=None):
        self.root_dir = os.path.join(root_dir, 'sintel/images/training/clean')
        super().__init__(dataset_name='sintel', root_dir=self.root_dir, split=split, load_cache=load_cache)
        # Set default parameters
        self.reshape_list['resolution'] = (1024,436)
        

    def get_cache_path(self, cache_dir):
        return os.path.join(cache_dir, 'sintel_pairs.pkl')

    def get_filter_scenes(self, split):
        all_scenes = self.get_all_scenes(self.root_dir)
        if split == 'val':
            return all_scenes[8:]  
        return []

    def get_rgb_depth_paths(self, scenes_path, scene_name):
        return (os.path.join(scenes_path, scene_name),
                os.path.join(scenes_path.replace('images/training/clean', 'depth/training/depth'), scene_name))

    def get_sorted_image_files(self, rgb_path):
        return sorted([f for f in os.listdir(rgb_path) if f.endswith('.png')],
                     key=lambda x: int(x.split('_')[1].split('.')[0]))

    def get_depth_name(self, img_name):
        return img_name.replace('.png', '.dpt')

    def depth_read(self, filename, **kwargs):
        """ Read depth data from file, return as numpy array. """
        f = open(filename, 'rb')
        check = np.fromfile(f, dtype=np.float32, count=1)[0]
        assert check == TAG_FLOAT, ' depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT, check)
        width = np.fromfile(f, dtype=np.int32, count=1)[0]
        height = np.fromfile(f, dtype=np.int32, count=1)[0]
        size = width * height
        assert width > 0 and height > 0 and size > 1 and size < 100000000, ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(width, height)
        depth = np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width))
        
        invalid_mask = np.logical_or.reduce((
            np.isinf(depth),
            np.isnan(depth),
            depth == 0,
            depth < 1e-5
        ))

        if invalid_mask.any():
            logging.info(f"Found invalid values in {filename}: "
                        f"inf: {np.isinf(depth).sum()}, "
                        f"nan: {np.isnan(depth).sum()}, "
                        f"=0: {(depth == 0).sum()}, "
                        f"<0: {(depth < 0).sum()}")

        sky_mask = depth > 1e4
        
        depth[invalid_mask] = -1
        inverse_depth = 1 / depth
        inverse_depth[sky_mask] = 0
        inverse_depth[invalid_mask] = -1
        
        return inverse_depth

