import os
import numpy as np
import logging
from .base_dataset_pairs import BaseDatasetPairs


class NuScenesDepth(BaseDatasetPairs):
    """
    NuScenes dataset loader for metric depth estimation.

    Directory structure expected:
    nuscenes/
        train/
            scene-xxxx/
                CAM_FRONT/
                    rgb/
                        xxx.jpg
                    depth/
                        xxx.npy
        val/
            ...
    """
    def __init__(self, root_dir, split, load_cache=None):
        # NuScenes has train/val splits
        self.root_dir = os.path.join(root_dir, f'nuscenes/{split}')
        super().__init__(dataset_name='nuscenes', root_dir=self.root_dir, split=split, load_cache=load_cache)

        # NuScenes camera resolution: 1600x900
        self.reshape_list['resolution'] = (1600, 900)

    def depth_read(self, path, return_torch=False, **kwargs):
        """
        Read depth from NuScenes depth file.
        Assumes depth is stored as metric depth in meters.
        """
        # Load metric depth (should be in meters)
        depth_map = np.load(path).astype(np.float32)

        # Convert to inverse depth for consistency with other datasets
        # Invalid pixels should be marked with 0 or negative values
        inverse_depth = np.where(depth_map > 0, 1.0 / depth_map, 0.0)

        return inverse_depth

    def get_cache_path(self, cache_dir):
        return os.path.join(cache_dir, f'nuscenes_{self.split}_pairs.pkl')

    def get_filter_scenes(self, split):
        """Get all scenes for the given split"""
        all_scenes = self.get_all_scenes(self.get_scenes_path())
        return sorted(all_scenes)

    def get_rgb_depth_paths(self, scenes_path, scene_name):
        """Get paths to RGB and depth directories"""
        item_path = os.path.join(scenes_path, scene_name)
        return (os.path.join(item_path, 'CAM_FRONT/rgb'),
                os.path.join(item_path, 'CAM_FRONT/depth'))

    def get_sorted_image_files(self, rgb_path):
        """Get sorted list of image files"""
        all_imgs = [f for f in os.listdir(rgb_path) if f.endswith('.jpg') or f.endswith('.png')]
        return sorted(all_imgs, key=lambda x: int(os.path.basename(x).split('.')[0]))

    def get_depth_name(self, img_name):
        """Convert image filename to depth filename"""
        base_name = os.path.splitext(img_name)[0]
        return f"{base_name}.npy"
