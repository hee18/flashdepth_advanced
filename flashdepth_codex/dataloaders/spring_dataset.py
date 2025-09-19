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


class SpringDepth(BaseDatasetPairs):
    def __init__(self, root_dir, split, load_cache=None):
        self.root_dir = os.path.join(root_dir, 'spring/train')
        super().__init__(dataset_name='spring', root_dir=self.root_dir, split=split, load_cache=load_cache)
        # 1920x1080 (ratio 1.77)
        self.reshape_list['resolution'] = (1920,1080)
        self.reshape_list['stride'] = 2


    def depth_read(self, path, is_inverse=True, resize_factor=1.0, return_torch=False, **kwargs):
        # https://github.com/Junyi42/monst3r/blob/main/dust3r/datasets/pointodyssey.py#L183
        # https://spring-benchmark.org/faq --> compute depth from disparity
        # Z = fx * B / d; baseline = 0.065m; Z=depth, d=disparity, fx=focal length; d of sky is set to 0
        # since I want inverse depth, directly calculate 1/Z = d / (fx * B) so I don't need to worry about dividing by 0

        assert is_inverse, "Spring dataset only supports inverse depth"
        
        SPRING_BASELINE = 0.065
            
        with h5py.File(path, "r") as f:
            if "disparity" not in f.keys():
                raise IOError(f"File {path} does not have a 'disparity' key. Is this a valid dsp5 file?")
            disparity = f["disparity"][()]

        assert disparity.shape == (2160,3840), "Spring disparity shape is not (2160,3840)"
        
        
        # TODO: better documentation / flags to determine how to handle disparity being 2x image size
        disparity = disparity[::2, ::2] # first cut in half to match image resolution (do it here, since cropping later assumes same dimensions)

        # get intrinsics
        index = int(os.path.basename(path).split('left_')[1].split('.')[0]) -1 # 1-indexed, -1 to get index
        intrinsics_path = os.path.dirname(path.replace('disp1_left', 'cam_data'))+'/intrinsics.txt'
        fx = np.loadtxt(intrinsics_path)[index][0]

        #TODO: scale focal length based on change in resolution; original resolution is 1920x1080
        # does not change with cropping 
        fx = fx * resize_factor

        inverse_depth = disparity / (fx * SPRING_BASELINE)

        # check if there are any nans or infs
        # if np.isnan(inverse_depth).any() or np.isinf(inverse_depth).any():
        #     logging.info(f"nan or inf in inverse depth for {path}")

        
        inverse_depth = np.where(np.isnan(inverse_depth), -1, inverse_depth)
        inverse_depth = np.where(np.isinf(inverse_depth), -1, inverse_depth)

        if kwargs.get('print_minmax', False):
            logging.info(f"minmax depth for {path}: {inverse_depth.min():.3f}, {inverse_depth.max():.3f}")

      
        if return_torch:
            inverse_depth = torch.from_numpy(inverse_depth).float()

        return inverse_depth

    def get_cache_path(self, cache_dir):
        return os.path.join(cache_dir, f'spring_pairs_{self.split}.pkl')

    # def get_scenes_path(self):
    #     return os.path.join(self.root_dir, 'train')

    def get_filter_scenes(self, split):
        if split == 'val':
            all_scenes = self.get_all_scenes(self.get_scenes_path())
            return [scene for i, scene in enumerate(all_scenes) if i % 3 != 2]
        elif split == 'train':
            return []
        return []

    def get_rgb_depth_paths(self, scenes_path, scene_name):
        item_path = os.path.join(scenes_path, scene_name)
        return (os.path.join(item_path, 'frame_left'),
                os.path.join(item_path, 'disp1_left'))

    def get_sorted_image_files(self, rgb_path):
        all_imgs = [f for f in os.listdir(rgb_path) if f.endswith('.png')]
        return sorted(all_imgs, key=lambda x: int(os.path.basename(x).split('left_')[1].split('.')[0]))

    def get_depth_name(self, img_name):
        return img_name.replace('.png', '.dsp5').replace('frame_left_', 'disp1_left_')