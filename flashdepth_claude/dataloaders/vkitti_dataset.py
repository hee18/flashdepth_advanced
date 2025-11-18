"""
VKITTI2 Dataset for depth estimation training

Directory structure:
    data_root/
        vkitti/
            Scene01/
                clone/
                    frames/
                        rgb/Camera_0/rgb_00000.jpg
                        depth/Camera_0/depth_00000.png (uint16, centimeters)
"""

import os
import cv2
import torch
import numpy as np
import logging
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
from .base_dataset_pairs import BaseDatasetPairs

logger = logging.getLogger(__name__)


class VKITTIDepth(BaseDatasetPairs):
    def __init__(self, root_dir, split, load_cache=None, **kwargs):
        """
        Initialize VKITTI2 dataset.

        Args:
            root_dir: Root directory
            split: Dataset split ('train', 'val', 'test')
            load_cache: Cache directory path
            **kwargs: Additional arguments (ignored for now)
        """
        self.root_dir = os.path.join(root_dir, 'vkitti')
        super().__init__(dataset_name='vkitti', root_dir=self.root_dir, split=split, load_cache=load_cache)

        # Set default parameters (VKITTI original: 1242×375 = 3.312 ratio)
        # Keep near original resolution with 14x divisibility for both base and 2k
        self.reshape_list['resolution'] = (1246, 378)  # (W, H) - 3.296 ratio, both 14x divisible

    def get_cache_path(self, cache_dir):
        return os.path.join(cache_dir, 'vkitti_pairs.pkl')

    def get_filter_scenes(self, split):
        """Filter scenes based on split"""
        
        # The user wants to test on all 5 scenes. The previous logic was inverted,
        # causing test scenes to be filtered out.
        # For split='test', we now return an empty list to prevent any filtering.
        if split == 'test':
            return []

        all_scenes = self.get_all_scenes(self.root_dir)

        # For completeness, also fixing the inverted 'val' split logic.
        # This will now correctly return only 'Scene01' and 'Scene02' for validation.
        if split == 'val':
            val_scenes = ['Scene01', 'Scene02']
            return [s for s in all_scenes if not any(x in s for x in val_scenes)]
            
        return []

    def get_all_scenes(self, scenes_path):
        """Get all scene directories"""
        vkitti_root = Path(scenes_path)

        # Get all Scene directories
        scene_dirs = sorted([d.name for d in vkitti_root.iterdir()
                           if d.is_dir() and d.name.startswith('Scene')])

        # Get all scene/condition combinations
        # Only use 'clone' condition by default for consistency
        scene_condition_paths = []
        for scene_name in scene_dirs:
            scene_dir = vkitti_root / scene_name
            # Only use 'clone' condition
            clone_dir = scene_dir / 'clone' / 'frames'
            if clone_dir.exists():
                scene_condition_paths.append(f"{scene_name}/clone/frames")

        return scene_condition_paths

    def get_rgb_depth_paths(self, scenes_path, scene_name):
        """Get RGB and depth paths for a scene"""
        vkitti_root = Path(scenes_path)
        scene_path = vkitti_root / scene_name

        rgb_path = scene_path / 'rgb' / 'Camera_0'
        depth_path = scene_path / 'depth' / 'Camera_0'

        return (str(rgb_path), str(depth_path))

    def get_sorted_image_files(self, rgb_path):
        """Get sorted image files"""
        return sorted([f for f in os.listdir(rgb_path) if f.startswith('rgb_') and f.endswith('.jpg')],
                     key=lambda x: int(x.split('_')[1].split('.')[0]))

    def get_depth_name(self, img_name):
        """Convert image filename to depth filename"""
        # rgb_00000.jpg -> depth_00000.png
        frame_num = img_name.split('_')[1].split('.')[0]
        return f'depth_{frame_num}.png'

    def depth_read(self, filename, is_inverse=False, **kwargs):
        """
        Read depth data from VKITTI2 PNG file.

        VKITTI2 depth format:
        - uint16 PNG
        - Values in centimeters (cm)
        - Need to convert to meters

        Args:
            filename: Path to depth PNG file
            is_inverse: If True, return inverse depth (1/m)
            **kwargs: Additional arguments

        Returns:
            numpy array of depth or inverse depth
        """
        # Read uint16 PNG
        depth_cm = cv2.imread(filename, cv2.IMREAD_ANYDEPTH)  # uint16

        if depth_cm is None:
            raise ValueError(f"Failed to read depth file: {filename}")

        # Convert from centimeters to meters
        depth_m = depth_cm.astype(np.float32) / 100.0

        # Handle invalid values
        invalid_mask = (depth_cm == 0) | (depth_m > 1e4)  # Sky or invalid

        if is_inverse:
            # Convert to inverse depth
            inverse_depth = np.zeros_like(depth_m, dtype=np.float32)
            valid_mask = ~invalid_mask & (depth_m > 1e-5)
            inverse_depth[valid_mask] = 1.0 / depth_m[valid_mask]
            inverse_depth[invalid_mask] = -1  # Mark invalid
            return inverse_depth
        else:
            # Metric depth
            depth_m[invalid_mask] = -1  # Mark invalid
            return depth_m

    def get_focal_length(self, pair, image_shape):
        """
        Get focal length for VKITTI2.

        VKITTI2 uses fixed intrinsics for all sequences:
        - Original resolution: 1242×375
        - fx = 725.0087 pixels (at original resolution)

        Args:
            pair: Data pair dict
            image_shape: (height, width) of current image

        Returns:
            Focal length in pixels at original resolution
        """
        # VKITTI2 fixed intrinsics (from documentation)
        # At original resolution 1242×375
        fx_original = 725.0087

        return fx_original
