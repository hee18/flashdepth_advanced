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


class TartanairDepth(BaseDatasetPairs):
    def __init__(self, root_dir, split, load_cache=None):
        self.root_dir = os.path.join(root_dir, 'tartanair')
        super().__init__(dataset_name='tartanair', root_dir=self.root_dir, split=split, load_cache=load_cache)
        # Set default parameters
        self.reshape_list['resolution'] = (640,480)
        self.reshape_list['stride'] = 2

    def get_cache_path(self, cache_dir):
        return os.path.join(cache_dir, 'tartanair_pairs.pkl')

    def get_all_scenes(self, scenes_path):
        scene_names = [s for s in os.listdir(scenes_path) if os.path.isdir(os.path.join(scenes_path, s))]
        all_scenes = []
        for scene_name in scene_names:
            if 'amusement' in scene_name:
                continue
            scene_full_path = os.path.join(scenes_path, scene_name)
            easy_path = os.path.join(scene_full_path, "Easy")
            hard_path = os.path.join(scene_full_path, "Hard")
            
            # Get directories inside Easy path
            if os.path.isdir(easy_path):
                easy_dirs = [os.path.join(scene_name, "Easy", d) for d in os.listdir(easy_path) 
                        if os.path.isdir(os.path.join(easy_path, d))]
                all_scenes.extend(easy_dirs)
            
            # Get directories inside Hard path
            if os.path.isdir(hard_path):
                hard_dirs = [os.path.join(scene_name, "Hard", d) for d in os.listdir(hard_path)
                        if os.path.isdir(os.path.join(hard_path, d))]
                all_scenes.extend(hard_dirs)
        
        return sorted(all_scenes)

    def get_filter_scenes(self, split):
        return []

    def get_rgb_depth_paths(self, scenes_path, scene_name):
        item_path = os.path.join(scenes_path, scene_name)
        return (os.path.join(item_path, 'image_left'),
                os.path.join(item_path, 'depth_left'))

    def get_sorted_image_files(self, rgb_path):
        all_imgs = [f for f in os.listdir(rgb_path) if f.endswith('.png')]
        return sorted(all_imgs, key=lambda x: int(os.path.basename(x).split('_left.png')[0]))

    def get_depth_name(self, img_name):
        return img_name.replace('_left.png', '_left_depth.npy')

    def depth_read(self, path, return_torch=False, **kwargs):
        depth = np.load(path)
        
        invalid_mask = np.logical_or.reduce((
            np.isinf(depth),
            np.isnan(depth),
            depth == 0,
            depth < 0
        ))
            
        depth[invalid_mask] = -1 
       
        inverse_depth = 1 / depth
        inverse_depth[inverse_depth < 0.001] = 0  # sky is roughly 10k in tartanair
        inverse_depth[invalid_mask] = -1
        
        if kwargs.get('print_minmax', False):
            logging.info(f"minmax depth for {path}: {inverse_depth.min():.3f}, {inverse_depth.max():.3f}")

        if return_torch:
            inverse_depth = torch.from_numpy(inverse_depth).float()

        return inverse_depth
