import os
import cv2
import torch
import numpy as np
import logging
from .base_dataset_pairs import BaseDatasetPairs

class MvsSynthDepth(BaseDatasetPairs):
    def __init__(self, root_dir, split, load_cache=None):
        self.root_dir = os.path.join(root_dir, 'mvs-synth/GTAV_1080')
        super().__init__(dataset_name='mvssynth', root_dir=self.root_dir, split=split, load_cache=load_cache)
        self.reshape_list['resolution'] = (1920, 1080)
        self.reshape_list['stride'] = 1

    def get_cache_path(self, cache_dir):
        return os.path.join(cache_dir, 'mvssynth_pairs.pkl')

    def get_all_scenes(self, scenes_path):
        all_scenes = [s for s in os.listdir(scenes_path) 
                     if os.path.isdir(os.path.join(scenes_path, s))]
        return sorted(all_scenes, key=lambda x: int(os.path.basename(x)))

    def get_filter_scenes(self, split):
        all_scenes = self.get_all_scenes(self.get_scenes_path())
        if split == 'val':
            return [s for s in all_scenes if s not in ['0118', '0119']]  # only use these two scenes
        elif split == 'train':
            return ['0118', '0119']  # leave for validation
        return []

    def get_rgb_depth_paths(self, scenes_path, scene_name):
        item_path = os.path.join(scenes_path, scene_name)
        return (os.path.join(item_path, 'images'),
                os.path.join(item_path, 'depths'))

    def get_sorted_image_files(self, rgb_path):
        all_imgs = [f for f in os.listdir(rgb_path) if f.endswith('.png')]
        return sorted(all_imgs, key=lambda x: int(os.path.basename(x).split('.png')[0]))

    def get_depth_name(self, img_name):
        return img_name.replace('.png', '.exr')

    def depth_read(self, path, return_torch=False, **kwargs):
        # raw depth values are roughly 70 to 10k
        # not sure about units, but dividing by 10 gives reasonable meter values when visualizing
        depth = cv2.imread(path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
        depth = depth / 10
        
        # no invalid values; sky is inf (https://phuang17.github.io/DeepMVS/mvs-synth.html)
        sky_mask = np.isinf(depth)
        depth[sky_mask] = -1  # avoid division issues
       
        inverse_depth = 1 / depth
        inverse_depth[sky_mask] = 0
        
        if kwargs.get('print_minmax', False):
            logging.info(f"minmax depth for {path}: {inverse_depth.min():.3f}, {inverse_depth.max():.3f}")

        if return_torch:
            inverse_depth = torch.from_numpy(inverse_depth).float()

        return inverse_depth