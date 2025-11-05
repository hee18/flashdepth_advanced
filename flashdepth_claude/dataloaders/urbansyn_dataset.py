import os
import cv2
import torch
import numpy as np
import logging
import json
from .base_dataset_pairs import BaseDatasetPairs
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

class UrbanSynDepth(BaseDatasetPairs):
    def __init__(self, root_dir, split, load_cache=None):
        self.root_dir = os.path.join(root_dir, 'urbansyn')
        super().__init__(dataset_name='urbansyn', root_dir=self.root_dir, split=split, load_cache=load_cache)
        # Set default parameters
        self.reshape_list['resolution'] = (2048,1024)

        # Load camera metadata once (all frames share same intrinsics)
        self.camera_fx = self._load_camera_metadata()
        

    def get_cache_path(self, cache_dir):
        return os.path.join(cache_dir, 'urbansyn_pairs.pkl')

    def _load_camera_metadata(self):
        """
        Load camera metadata from root directory.
        UrbanSyn has a single camera_metadata.json file at the root.
        """
        metadata_path = os.path.join(self.root_dir, 'camera_metadata.json')
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Extract camera parameters
            camera_params = metadata['parameters'][0]['Camera']
            focal_length_mm = camera_params['focalLength_mm']
            sensor_width_mm = camera_params['sensorWidth_mm']

            # Compute fx in pixels for 2048×1024 images
            width = 2048
            fx = focal_length_mm * width / sensor_width_mm

            logging.info(f"UrbanSyn camera fx: {fx:.2f} pixels (focal={focal_length_mm}mm, sensor={sensor_width_mm}mm)")
            return fx
        except Exception as e:
            logging.warning(f"Error reading camera metadata from {metadata_path}: {e}, using fallback fx=1731.0")
            return 1731.0  # Typical value for 2048×1024 with ~60° FOV

    def get_all_scenes(self, scenes_path):
        # UrbanSyn is a single sequence, treat the root as one "scene"
        return ['urbansyn']

    def get_filter_scenes(self, split):
        return []

    def get_rgb_depth_paths(self, scenes_path, scene_name):
        # UrbanSyn has rgb/ and depth/ directly in root, not in scene subdirectories
        return (os.path.join(scenes_path, 'rgb'),
                os.path.join(scenes_path, 'depth'))

    def get_sorted_image_files(self, rgb_path):
        all_imgs = [f for f in os.listdir(rgb_path) if f.endswith('.png')]
        # logging.info(f"only using first 1000 images from urbansyn")
        return sorted(all_imgs, key=lambda x: int(os.path.basename(x).split('rgb_')[1].split('.png')[0]))[0:1000]

    def get_depth_name(self, img_name):
        return img_name.replace('.png', '.exr').replace('rgb_', 'depth_')

    def depth_read(self, path, return_torch=False, **kwargs):
        # according to documentation, *1e5 gives meters
        depth = cv2.imread(path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
        depth *= 1e5

        ss_path = path.replace('depth/depth_', 'ss/ss_').replace('.exr', '.png')
        segmentation_mask = cv2.imread(ss_path, cv2.IMREAD_ANYDEPTH)  # only need one channel for the id values

        assert depth.shape == segmentation_mask.shape, 'depth and seg mask should have same shape'

        sky_mask = segmentation_mask == 10  # class ID 10 => sky
        depth[sky_mask] = -1  # avoid division issues
        
        inverse_depth = 1 / depth
        inverse_depth[sky_mask] = 0
        
        if kwargs.get('print_minmax', False):
            logging.info(f"minmax depth for {path}: {inverse_depth.min():.3f}, {inverse_depth.max():.3f}")

        if return_torch:
            inverse_depth = torch.from_numpy(inverse_depth).float()

        return inverse_depth

    def get_focal_length(self, pair, image_shape):
        """
        Get focal length for UrbanSyn dataset.

        UrbanSyn has a single camera_metadata.json at the root.
        All frames share the same intrinsics (loaded in __init__).

        Args:
            pair (dict): Data pair (not used)
            image_shape (tuple): (H, W) image shape (not used)

        Returns:
            float: Focal length in pixels
        """
        return self.camera_fx