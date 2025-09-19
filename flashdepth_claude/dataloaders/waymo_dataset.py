import os
import numpy as np
import logging
from .base_dataset_pairs import BaseDatasetPairs


# randomly sampled 30 scenes from the waymo validation split
testing_scenes = ['segment-12831741023324393102_2673_230_2693_230', 'segment-10837554759555844344_6525_000_6545_000', 'segment-11450298750351730790_1431_750_1451_750', 'segment-11434627589960744626_4829_660_4849_660', 'segment-14127943473592757944_2068_000_2088_000', 'segment-12940710315541930162_2660_000_2680_000', 'segment-12306251798468767010_560_000_580_000', 'segment-11616035176233595745_3548_820_3568_820', 'segment-14165166478774180053_1786_000_1806_000', 'segment-1071392229495085036_1844_790_1864_790', 'segment-12374656037744638388_1412_711_1432_711', 'segment-13469905891836363794_4429_660_4449_660', 'segment-1405149198253600237_160_000_180_000', 'segment-10289507859301986274_4200_000_4220_000', 'segment-12102100359426069856_3931_470_3951_470', 'segment-14081240615915270380_4399_000_4419_000', 'segment-11048712972908676520_545_000_565_000', 'segment-13573359675885893802_1985_970_2005_970', 'segment-10689101165701914459_2072_300_2092_300', 'segment-12496433400137459534_120_000_140_000', 'segment-11356601648124485814_409_000_429_000', 'segment-11406166561185637285_1753_750_1773_750', 'segment-10868756386479184868_3000_000_3020_000', 'segment-1024360143612057520_3580_000_3600_000', 'segment-13299463771883949918_4240_000_4260_000', 'segment-12820461091157089924_5202_916_5222_916', 'segment-10203656353524179475_7625_000_7645_000', 'segment-13178092897340078601_5118_604_5138_604', 'segment-12134738431513647889_3118_000_3138_000', 'segment-13356997604177841771_3360_000_3380_000']


class WaymoDepth(BaseDatasetPairs):
    def __init__(self, root_dir, split, load_cache=None):
        self.root_dir = os.path.join(root_dir, 'waymo/val')  
        super().__init__(dataset_name='waymo', root_dir=self.root_dir, split=split, load_cache=load_cache)
        # 1920x1280
        self.reshape_list['resolution'] = (1920, 1280)

    def depth_read(self, path, return_torch=False, **kwargs):
        h, w = 1280, 1920
        
        # Load the sparse depth points (N,3)
        depth_points = np.load(path)
        
        # Initialize depth map with -1
        depth_map = np.full((h, w), -1, dtype=np.float32)
        
        # Extract coordinates and depths
        x_coords = depth_points[:, 0].astype(np.int32)
        y_coords = depth_points[:, 1].astype(np.int32)
        depths = depth_points[:, 2]
        
        # Filter valid coordinates (within image bounds)
        valid_mask = (x_coords >= 0) & (x_coords < w) & (y_coords >= 0) & (y_coords < h)
        x_coords = x_coords[valid_mask]
        y_coords = y_coords[valid_mask]
        depths = depths[valid_mask]
        
        # Place depths in the depth map
        depth_map[y_coords, x_coords] = depths
        
        # Create inverse depth map (keeping -1 for invalid pixels)
        inverse_depth = np.where(depth_map > 0, 1.0 / depth_map, -1)
        
        return inverse_depth

    def get_cache_path(self, cache_dir):
        return os.path.join(cache_dir, 'waymo_pairs.pkl')

    def get_filter_scenes(self, split):
        all_scenes = self.get_all_scenes(self.get_scenes_path())
        
        if split == 'val':
            return sorted(all_scenes)[8:] # only use first 8 scenes
        elif split == 'test':
            return [s for s in all_scenes if s not in testing_scenes] # only use the 30 testing scenes
        return []  

    def get_rgb_depth_paths(self, scenes_path, scene_name):
        item_path = os.path.join(scenes_path, scene_name)
        return (os.path.join(item_path, 'FRONT/rgb/original'),
                os.path.join(item_path, 'FRONT/depth'))

    def get_sorted_image_files(self, rgb_path):
        all_imgs = [f for f in os.listdir(rgb_path) if f.endswith('.jpg')]
        return sorted(all_imgs, key=lambda x: int(os.path.basename(x).split('.jpg')[0]))

    def get_depth_name(self, img_name):
        return img_name.replace('.jpg', '.npy')