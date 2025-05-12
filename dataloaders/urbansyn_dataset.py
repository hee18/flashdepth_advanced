import os
import cv2
import torch
import numpy as np
import logging
from .base_dataset_pairs import BaseDatasetPairs
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

class UrbanSynDepth(BaseDatasetPairs):
    def __init__(self, root_dir, split, load_cache=None):
        self.root_dir = os.path.join(root_dir, 'urbansyn')
        super().__init__(dataset_name='urbansyn', root_dir=self.root_dir, split=split, load_cache=load_cache)
        # Set default parameters
        self.reshape_list['resolution'] = (2048,1024)
        

    def get_cache_path(self, cache_dir):
        return os.path.join(cache_dir, 'urbansyn_pairs.pkl')

    def get_all_scenes(self, scenes_path):
        all_scenes = [s for s in os.listdir(scenes_path) 
                     if os.path.isdir(os.path.join(scenes_path, s))]
        return sorted(all_scenes, key=lambda x: int(os.path.basename(x)))

    def get_filter_scenes(self, split):
        return []

    def get_rgb_depth_paths(self, scenes_path, scene_name):
        item_path = os.path.join(scenes_path, scene_name)
        return (os.path.join(item_path, 'rgb'),
                os.path.join(item_path, 'depth'))

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