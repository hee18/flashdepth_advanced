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
import OpenEXR
import pickle
from .base_dataset_pairs import BaseDatasetPairs
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


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
        if split == 'val':
            return sorted(all_scenes)[8:]
            # return ['relief_2', 'meadow', 'terrains']
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
        imgpath = path.replace('ground_truth_depth', 'images')
        h, w = cv2.imread(imgpath, cv2.IMREAD_ANYDEPTH).shape

        depth = np.fromfile(path, dtype=np.float32)
        assert depth.size == h * w, "Mismatch between file size and expected depth dimensions"
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