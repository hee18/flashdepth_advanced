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



class DynamicReplicaDepth(BaseDatasetPairs):
    def __init__(self, root_dir, split, load_cache=None):
        self.root_dir = os.path.join(root_dir, 'dynamic_replica/train')
        super().__init__(dataset_name='dynamicreplica', root_dir=self.root_dir, split=split, load_cache=load_cache)
        # Set default parameters
        self.reshape_list['resolution'] = (1280,720)
        self.reshape_list['stride'] = 2
       

    def get_cache_path(self, cache_dir):
        return os.path.join(cache_dir, 'dynamicreplica_pairs.pkl')

    def get_all_scenes(self, scenes_path):
        all_scenes = [s for s in os.listdir(scenes_path) 
                     if os.path.isdir(os.path.join(scenes_path, s)) and '_left' in s]
        return sorted(all_scenes)

    def get_filter_scenes(self, split):
        all_scenes = self.get_all_scenes(self.get_scenes_path())
        if split == 'val':
            return [s for s in all_scenes if s not in ['a1e031-7_obj_source_left', '1a1407-3_obj_source_left']]
        elif split == 'train':
            return ['009850-3_obj_source_left']  # github issue says this scene is invalid
        return []

    def get_rgb_depth_paths(self, scenes_path, scene_name):
        item_path = os.path.join(scenes_path, scene_name)
        return (os.path.join(item_path, 'images'),
                os.path.join(item_path, 'depths'))

    def get_sorted_image_files(self, rgb_path):
        all_imgs = [f for f in os.listdir(rgb_path) if f.endswith('.png')]
        return sorted(all_imgs, key=lambda x: int(os.path.basename(x).split('_left-')[1].split('.png')[0]))

    def get_depth_name(self, img_name):
        return img_name.replace('_left-', '_left_').replace('.png', '.geometric.png')

    def depth_read(self, path, return_torch=False, **kwargs):
        # https://github.com/facebookresearch/dynamic_stereo/blob/dfe2907faf41b810e6bb0c146777d81cb48cb4f5/datasets/dynamic_stereo_datasets.py#L59
        with Image.open(path) as depth_pil:
            depth = (
                np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
                .astype(np.float32)
                .reshape((depth_pil.size[1], depth_pil.size[0]))
            )

        
        invalid_mask = np.logical_or.reduce((
            np.isinf(depth),
            np.isnan(depth),
            depth == 0,
            depth<0
        ))

        # if invalid_mask.any():
        #     logging.info(f"Found invalid values in {path}: "
        #                 f"inf: {np.isinf(depth).sum()}, "
        #                 f"nan: {np.isnan(depth).sum()}, "
        #                 f"=0: {(depth == 0).sum()}, "
        #                 f"<0: {(depth < 0).sum()}")
            
        depth[invalid_mask] = -1 
       
        inverse_depth = 1 / depth
        inverse_depth[invalid_mask] = -1
        
        if kwargs.get('print_minmax', False):
            logging.info(f"minmax depth for {path}: {inverse_depth.min():.3f}, {inverse_depth.max():.3f}")

      
        if return_torch:
            inverse_depth = torch.from_numpy(inverse_depth).float()

        return inverse_depth
