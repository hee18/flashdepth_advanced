import os
import cv2
import torch
import numpy as np
import logging
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from PIL import Image
import torch.distributed as dist
import pickle
from .base_dataset_pairs import BaseDatasetPairs

class PointOdysseyDepth(BaseDatasetPairs):
    def __init__(self, root_dir, split, load_cache=None):
        self.root_dir = os.path.join(root_dir, 'pointodyssey/train')
        super().__init__(dataset_name='pointodyssey', root_dir=self.root_dir, split=split, load_cache=load_cache)
        # Set default parameters
        self.reshape_list['resolution'] = (960, 540)
        self.reshape_list['stride'] = 2


    def get_cache_path(self, cache_dir):
        return os.path.join(cache_dir, 'pointodyssey_pairs.pkl')


    def get_all_scenes(self, scenes_path):
        cut3r_selected_scenes = [
            'cnb_dlab_0215_3rd', 'cnb_dlab_0215_ego1',
            'cnb_dlab_0225_3rd', 'cnb_dlab_0225_ego1',
            'dancing', 'dancingroom0_3rd', 'footlab_3rd',
            'footlab_ego1', 'footlab_ego2', 'girl', 'girl_egocentric',
            'human_egocentric', 'human_in_scene', 'human_in_scene1',
            'kg', 'kg_ego1', 'kg_ego2',
            'kitchen_gfloor', 'kitchen_gfloor_ego1', 'kitchen_gfloor_ego2',
            'scene_carb_h_tables', 'scene_carb_h_tables_ego1', 'scene_carb_h_tables_ego2',
            'scene_j716_3rd', 'scene_j716_ego1', 'scene_j716_ego2',
            'scene_recording_20210910_S05_S06_0_3rd', 'scene_recording_20210910_S05_S06_0_ego2',
            'scene1_0129', 'scene1_0129_ego',
            'seminar_h52_3rd', 'seminar_h52_ego1', 'seminar_h52_ego2'
        ]
        return [s for s in sorted(os.listdir(scenes_path)) if s in cut3r_selected_scenes]

    def get_filter_scenes(self, split):
        return []

    def get_rgb_depth_paths(self, scenes_path, scene_name):
        item_path = os.path.join(scenes_path, scene_name)
        return (os.path.join(item_path, 'rgbs'),
                os.path.join(item_path, 'depths'))

    def get_sorted_image_files(self, rgb_path):
        all_imgs = [f for f in os.listdir(rgb_path) if f.endswith('.jpg')]
        return sorted(all_imgs, key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))

    def get_depth_name(self, img_name):
        return img_name.replace('.jpg', '.png').replace('rgb', 'depth')

    def depth_read(self, path, return_torch=False, **kwargs):
        # https://github.com/Junyi42/monst3r/blob/main/dust3r/datasets/pointodyssey.py#L183
        # https://github.com/y-zheng18/point_odyssey/blob/main/utils/reprojection.py#L70
        depth16 = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        depth = depth16.astype(np.float32) / 65535.0 * 1000.0  # 1000 is the max depth in the dataset

        # sanity check
        if np.isnan(depth).any() or np.isinf(depth).any():
            logging.info(f"nan or inf in inverse depth for {path}")
        if depth.min() < 0:
            logging.info(f"negative depth for {path}")
        if depth.max() > 1000:
            logging.info(f"max depth for {path}")

        # pixels where depth==0 are invalid (indoor scenes, no sky)
        min_depth = 0.01  # 1cm, reasonable value based on observations
        invalid_mask = depth == 0
        depth[invalid_mask] = -1  # temporarily remove zeros to avoid division by zero

        inverse_depth = np.where(depth < min_depth, 1/min_depth, 1.0 / depth)
        inverse_depth[invalid_mask] = -1

        if kwargs.get('print_minmax', False):
            logging.info(f"minmax depth for {path}: {inverse_depth.min():.3f}, {inverse_depth.max():.3f}")

        if return_torch:
            inverse_depth = torch.from_numpy(inverse_depth).float()

        return inverse_depth

if __name__ == '__main__':
    dataset = PointOdysseyDepth(root_dir='/root/gene/data/pointodyssey', split='train')
    print(len(dataset))
    exit()