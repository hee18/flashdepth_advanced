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
    def __init__(self, root_dir, split, load_cache=None, use_segmentation=False, return_dict=False, **kwargs):
        """
        Initialize Sintel dataset.

        Args:
            root_dir: Root directory
            split: Dataset split ('train', 'val', 'test')
            load_cache: Cache directory path
            use_segmentation: Whether to load segmentation data (for sintel_seg) - currently ignored
            return_dict: Whether to return dict (True) or tuple (False) - currently ignored
            **kwargs: Additional arguments (ignored for now)

        Note:
            use_segmentation and return_dict are accepted for API compatibility but not used.
            For object-wise evaluation with segmentation, use SintelSegmentationDataset directly.
        """
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

    def get_focal_length(self, pair, image_shape):
        """
        Get focal length for Sintel dataset.

        Sintel provides per-frame intrinsics in cam_data/training/camdata_left/*.cam files.
        Binary format:
          - TAG_FLOAT (float32): validation value (202021.25)
          - Intrinsic matrix M: 9 float64 values (3×3)
          - Extrinsic matrix N: 12 float64 values (3×4) [not used]

        Args:
            pair (dict): Data pair with 'image' and 'depth' paths
            image_shape (tuple): (H, W) image shape

        Returns:
            float: Focal length in pixels
        """
        # Extract frame info from image path (e.g., .../clean/scene_name/frame_0001.png)
        img_path = pair['image']

        # Read camera file
        cam_path = img_path.replace('images/training/clean', 'cam_data/training/camdata_left').replace('.png', '.cam')

        try:
            with open(cam_path, 'rb') as f:
                # Read TAG_FLOAT validation value (float32)
                tag_val = np.fromfile(f, dtype=np.float32, count=1)[0]
                if abs(tag_val - TAG_FLOAT) > 0.01:  # Allow small floating point error
                    logging.warning(f"Unexpected tag in {cam_path}: {tag_val} (expected {TAG_FLOAT})")

                # Read intrinsic matrix M (9 float64 values, reshape to 3×3)
                M = np.fromfile(f, dtype=np.float64, count=9).reshape(3, 3)
                fx = float(M[0, 0])

                # Note: Extrinsic matrix N (12 float64) follows but we don't need it
                return fx
        except Exception as e:
            logging.warning(f"Error reading camera from {cam_path}: {e}, using fallback")
            # Fallback: typical value for 1024×436 with ~50° FOV
            return image_shape[1] * 0.9

