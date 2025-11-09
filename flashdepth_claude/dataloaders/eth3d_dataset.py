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

# ETH3D doesn't use OpenEXR


class Eth3dDepth(BaseDatasetPairs):
    def __init__(self, root_dir, split, load_cache=None):
        self.root_dir = os.path.join(root_dir, 'eth3d')
        super().__init__(dataset_name='eth3d', root_dir=self.root_dir, split=split, load_cache=load_cache)
        # Set default parameters
        self.reshape_list['resolution'] = (6048,4032)


    def get_cache_path(self, cache_dir):
        return os.path.join(cache_dir, 'eth3d_pairs.pkl')


    def get_all_scenes(self, scenes_path):
        all_scenes = [s for s in os.listdir(scenes_path) 
                     if os.path.isdir(os.path.join(scenes_path, s))]
        return sorted(all_scenes)

    def get_filter_scenes(self, split):
        all_scenes = self.get_all_scenes(self.get_scenes_path())
        # filter_scenes = scenes to EXCLUDE
        if split == 'val':
            # Exclude first 8 scenes (use last N scenes for validation)
            # Also exclude multi_view_training_dslr_undistorted directory
            exclude = sorted(all_scenes)[:8]
            if 'multi_view_training_dslr_undistorted' not in exclude:
                exclude.append('multi_view_training_dslr_undistorted')
            return exclude
        # For train: exclude nothing (use all)
        return []

    def get_rgb_depth_paths(self, scenes_path, scene_name):
        item_path = os.path.join(scenes_path, scene_name)
        return (os.path.join(item_path, 'images/dslr_images'),
                os.path.join(item_path, 'ground_truth_depth/dslr_images'))

    def get_sorted_image_files(self, rgb_path):
        all_imgs = [f for f in os.listdir(rgb_path) if f.endswith('.JPG')]
        return sorted(all_imgs, key=lambda x: int(os.path.basename(x).split('DSC_')[1].split('.JPG')[0]))

    def get_depth_name(self, img_name):
        return img_name  # ETH3D uses same filename for depth and RGB

    def depth_read(self, path, return_torch=False, **kwargs):
        # ETH3D depth is always stored at 6048x4032 resolution (original/native resolution)
        # This is BEFORE any resizing in the dataloader preprocessing
        w, h = 6048, 4032

        depth = np.fromfile(path, dtype=np.float32)
        assert depth.size == h * w, f"Mismatch between file size ({depth.size}) and expected depth dimensions ({h}x{w}={h*w})"
        depth = depth.reshape((h, w))

        invalid_mask = depth == np.inf
        depth[invalid_mask] = -1
        
        inverse_depth = 1 / depth
        inverse_depth[invalid_mask] = -1
        
        if kwargs.get('print_minmax', False):
            logging.info(f"minmax depth for {path}: {inverse_depth.min():.3f}, {inverse_depth.max():.3f}")

        if return_torch:
            inverse_depth = torch.from_numpy(inverse_depth).float()

        return inverse_depth

    def get_focal_length(self, pair, image_shape):
        """
        Get focal length for ETH3D dataset.

        ETH3D provides per-image intrinsics in cameras.txt (COLMAP format).
        Format: CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy (for original 6048×4032 resolution)

        Args:
            pair (dict): Data pair with 'image' and 'depth' paths
            image_shape (tuple): (H, W) image shape AFTER resizing

        Returns:
            float: Focal length in pixels for current image shape
        """
        # Extract scene directory and image name
        img_path = pair['image']
        scene_dir = os.path.dirname(os.path.dirname(os.path.dirname(img_path)))  # Go up from images/dslr_images to scene/
        img_name = os.path.basename(img_path)

        # Read cameras.txt and images.txt
        cameras_path = os.path.join(scene_dir, 'dslr_calibration_undistorted', 'cameras.txt')
        images_path = os.path.join(scene_dir, 'dslr_calibration_undistorted', 'images.txt')

        try:
            # Parse cameras.txt to get camera parameters
            cameras = {}
            with open(cameras_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#') or not line:
                        continue
                    parts = line.split()
                    camera_id = int(parts[0])
                    model = parts[1]
                    if model == 'PINHOLE':
                        fx_original = float(parts[4])
                        cameras[camera_id] = fx_original

            # Parse images.txt to find camera_id for current image
            with open(images_path, 'r') as f:
                lines = f.readlines()
                for i in range(len(lines)):
                    line = lines[i].strip()
                    if line.startswith('#') or not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 10:
                        # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
                        # NAME format: dslr_images_undistorted/DSC_0323.JPG
                        name = parts[9]
                        # Check if basename matches (handle both with and without path)
                        if name == img_name or os.path.basename(name) == img_name:
                            camera_id = int(parts[8])
                            fx_original = cameras[camera_id]
                            
                            # Scale focal length to current image width (from original 6048)
                            original_width = 6048
                            current_width = image_shape[1]
                            fx_scaled = fx_original * (current_width / original_width)
                            return fx_scaled

            # If not found, use first camera as fallback
            if cameras:
                fx_original = list(cameras.values())[0]
                logging.warning(f"Camera for {img_name} not found in images.txt, using first camera fx={fx_original:.1f}")
                # Scale to current width
                fx_scaled = fx_original * (image_shape[1] / 6048)
                return fx_scaled

            raise ValueError(f"No cameras found in {cameras_path}")
        except Exception as e:
            logging.warning(f"Error reading camera from {cameras_path}: {e}, using fallback")
            # Fallback: typical value scaled to current width
            fx_fallback = image_shape[1] * 0.8
            return fx_fallback