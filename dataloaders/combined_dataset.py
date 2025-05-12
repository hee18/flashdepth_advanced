import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import torch.distributed as dist
import numpy as np
import logging
import os
from os.path import join
import math

try:
    from .depthanything_preprocess import _load_and_process_image, _load_and_process_depth
    from .base_dataset_pairs import BaseDatasetPairs
except:
    from depthanything_preprocess import _load_and_process_image, _load_and_process_depth
    from base_dataset_pairs import BaseDatasetPairs

class CombinedDataset(Dataset):
    def __init__(self, root_dir, enable_dataset_flags, resolution=None, split='train',
                 video_length=8, seed=42, tmp_res=None, color_aug=False):
        '''
        enable_dataset_flags: list of datasets to use; e.g. ['spring', 'mvs-synth', 'urbansyn', 'eth3d', 'waymo']

        # must have a couple of 2k, preferably dynamic datasets for testing
        current options: eth3d, waymo, spring 
        eth resolution: 6048x4032 (ratio 1.5)
        waymo: 1920x1280 (ratio 1.5)
        spring: 1920x1080 (ratio 1.77)

        # training datasets
        # use a unique resolution for each dataset to preserve aspect ratio, and potentially do cropping where possible (e.g. pointodyssey)
        current: mvs-synth (1920x1080)->1960x1120, urbansyn (2048x1024)->2072x1064, 
            pointodyssey (960x540)-> 504x280 (bc not downsampling) / 1008x560 (enc-dec), dynamic replica (1280x720)->1288x728
        
        
        # might not be able to do vkitti and sintel because their height is too low (300/400); 
        # would only work if I can pass them through without the unet; but current experiments show that it doesn't work
        to add: vkitti, tartanair, sintel (sintel might have slightly weird depth values)


        There aren't many other 2k videos for training (spring is the only one I'm familiar with),
        so I'll mix in hd as well
        # 2k: mvs-synth, urbansyn, unrealstereo4k; (maybe waymo, do an ablation; maybe hoi4d)
        # lower res + dynamic: dynamic replica, pointodyssey, sintel, vkitti, tartanair, bedlam
        for the lower res datasets, I can either just do the same 4x downsample through unet;
        or have a condition in the model to not pass them through the unet


        raw resolutions: pointodyssey is 960x540, spring is 1920x1080, sintel is 1024x436;
        dynamic replica: 1280x720; vkitti: 1242, 375; 
        tartanair: 640x480; hypersim: 1024x768; IRS: 960x540
        3dkenburs: 512x512; bedlam: 1280x720, synscapes: 1440x720
        mapillary: 640x360; nyu depth: 640x480; bonn 640x480

        res > full hd
        spring, eth3d, unrealsstereo4k (3840x2160), waymo, ARKitScenes, mvs-synth, phonedepth, urbansyn, hoi4d


        '''
        np.random.seed(seed)
        torch.manual_seed(seed)


        
        cache_dir = '/root/gene/video-depth/dataloaders/pairs_cache' if split != 'test' else None

        self.pairslist = {}
        self.depth_read_list = {}
        self.reshape_list = {}
        self.tmp_res = tmp_res

        ## testing sets (don't need same resolution because batch size always 1)
        # unreal4k 7 scenes
        # sintel 23 -> 16 scenes
        # waymo 50 -> 8 scenes; urbansyn 8 val scenes; 50 images each

        for dataset_name in enable_dataset_flags:
            dataset = BaseDatasetPairs.create(dataset_name, root_dir, split, load_cache=cache_dir)
            self.pairslist[dataset_name] = dataset.pairs
            self.depth_read_list[dataset_name] = dataset.depth_read
            self.reshape_list[dataset_name] = dataset.reshape_list

        if resolution == 'base':
            if split == 'train':
                for dataset in self.reshape_list:
                    self.reshape_list[dataset]['resolution'] = (518, 518)
                    self.reshape_list[dataset]['crop_type'] = 'center'
                    if dataset in ['spring', 'mvs-synth']:
                        self.reshape_list[dataset]['resize_factor'] = 0.5
                    if dataset in ['pointodyssey']:
                        self.reshape_list[dataset]['resize_factor'] = 1.0
                    if dataset in ['dynamicreplica']:
                        self.reshape_list[dataset]['resize_factor'] = 0.75
         
            else:
                for dataset in self.reshape_list:
                    self.reshape_list[dataset]['crop_type'] = None
                    if dataset in ['eth3d', 'waymo']:
                        self.reshape_list[dataset]['resolution'] = (784,518)
                    if dataset in ['sintel']:
                        self.reshape_list[dataset]['resolution'] = (1022,434)
                    if dataset in ['urbansyn']:
                        self.reshape_list[dataset]['resolution'] = (1036,518)
                    if dataset in ['unreal4k']:
                        self.reshape_list[dataset]['resolution'] = (924,518) 


        elif resolution == '2k':
            if split == 'train':
                for dataset in self.reshape_list:
                    self.reshape_list[dataset]['resolution'] = (1918, 1078)
                    self.reshape_list[dataset]['crop_type'] = 'random'
                    self.reshape_list[dataset]['stride'] = 2
                # if dataset in ['tartanair', 'pointodyssey', 'dynamicreplica']:
                #     self.reshape_list[dataset]['resolution'] = (518, 518)
                # if dataset in ['dynamicreplica']:
                #     self.reshape_list[dataset]['resize_factor'] = 0.8

                if dataset in ['tartanair']:
                    self.reshape_list[dataset]['resolution'] = (630, 476)
                if dataset in ['pointodyssey']:
                    self.reshape_list[dataset]['resolution'] = (952, 532)
                if dataset in ['dynamicreplica']:
                    self.reshape_list[dataset]['resolution'] = (1274, 714)

            else:
                for dataset in self.reshape_list:
                    self.reshape_list[dataset]['crop_type'] = None
                    if dataset in ['eth3d', 'waymo']:
                        self.reshape_list[dataset]['resolution'] = (1918,1274) 
                    if dataset in ['sintel']:
                        self.reshape_list[dataset]['resolution'] = (1022,434)
                    if dataset in ['urbansyn']:
                        self.reshape_list[dataset]['resolution'] = (2044,1022)
                    if dataset in ['unreal4k']:
                        self.reshape_list[dataset]['resolution'] = (2044,1148) 

        else:
            raise ValueError(f"Resolution should be 'base' or '2k' for training")
        
        self.pairs = []


        for dataset_name in enable_dataset_flags:
            indices = list(range(len(self.pairslist[dataset_name])))
            self.pairs.extend([(dataset_name, i) for i in indices])
            logging.info(f"length of {dataset_name} for {split}: {len(self.pairslist[dataset_name])}")

        if split != 'train':    
            logging.info(f"enabled datasets for {split}: {enable_dataset_flags}")
            logging.info(f"length of combined dataset: {len(self.pairs)}")

      
        self.video_length = video_length
        self.split = split
        self.color_aug = color_aug

        

        

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):

        if self.split == 'val':
            dataset_idx, scene_idx = self.pairs[idx]
            scene = self.pairslist[dataset_idx][scene_idx]
            
            images = []
            depths = []
            for pair in scene:
                image, _current_crop = _load_and_process_image(pair['image'], **self.reshape_list[dataset_idx])
                depth = self.depth_read_list[dataset_idx](pair['depth'], is_inverse=True) # needed for scaling focal length, currently only for Spring
                # don't resize depth; resize output to match gt instead in inference loop
                # depth = _load_and_process_depth(depth, image.shape, _current_crop, **self.reshape_list[dataset_idx])

                # if self.reshape_list[dataset_idx]['crop_type'] == 'center':
                #     h, w = depth.shape
                #     start_y, start_x, target_h, target_w = _current_crop
                #     # Scale crop coordinates to original depth resolution
                #     orig_start_y = int((start_y / image.shape[1]) * h)
                #     orig_start_x = int((start_x / image.shape[2]) * w)
                #     orig_target_h = int((target_h / image.shape[1]) * h)
                #     orig_target_w = int((target_w / image.shape[2]) * w)
                #     depth = depth[orig_start_y:orig_start_y + orig_target_h, 
                #                 orig_start_x:orig_start_x + orig_target_w]
        
                images.append(image)
                depths.append(torch.from_numpy(depth).float()) # not resizing depth, using original resolution

            return_name = dataset_idx
            return torch.stack(images).float(), torch.stack(depths).float(), return_name


        elif self.split == 'test':
            dataset_idx, scene_idx = self.pairs[idx]
            scene = self.pairslist[dataset_idx][scene_idx]
            
            images = []
            for pair in scene:
                image, _current_crop = _load_and_process_image(pair['image'], **self.reshape_list[dataset_idx])
             
                images.append(image)

            return_name = os.path.join(dataset_idx, pair['scene_name']) 
            return torch.stack(images).float(), return_name


        # dataset_idx: i-th dataset; e.g. pointodyssey is 0, spring is 1...etc
        # pair_idx: the i-th (img, depth) pair in the dataset, for instance, pair_idx \in [0, 5000] in Spring
        dataset_idx, pair_idx = self.pairs[idx]  
        dataset_list = self.pairslist[dataset_idx]
        pair = dataset_list[pair_idx]

        # pair = {
        #     'image': img_path,
        #     'depth': depth_path,
        #     'scene_index': index of current frame in scene,
        #     'scene_length': total frames in scene,
        #     'scene_name': name of scene
        # }

        scene_index = pair['scene_index']
        scene_length = pair['scene_length']
        stride = self.reshape_list[dataset_idx]['stride']


        # Check if we can go both forward and backward
        can_go_forward = scene_index + (self.video_length - 1) * stride <= scene_length - 1
        can_go_backward = scene_index >= (self.video_length - 1) * stride
        
        if can_go_forward and can_go_backward:
            # Randomly choose direction
            if torch.rand(1).item() > 0.5:
                sequence_indices = list(range(scene_index, scene_index + self.video_length * stride, stride))
            else:
                start_pos = scene_index - (self.video_length - 1) * stride
                sequence_indices = list(range(start_pos, scene_index + 1, stride))
        elif can_go_forward:
            # Only enough frames ahead
            sequence_indices = list(range(scene_index, scene_index + self.video_length * stride, stride))
        elif can_go_backward:
            # Must go backward
            start_pos = scene_index - (self.video_length - 1) * stride
            sequence_indices = list(range(start_pos, scene_index + 1, stride))
        else:
            # Can't go either way - use remaining frames forward then wrap around backward
            remaining_forward = scene_length - scene_index
            remaining_forward_frames = math.ceil(remaining_forward / stride)
            remaining_needed = max(self.video_length - remaining_forward_frames, 0)

            # Get forward frames
            sequence_indices = list(range(scene_index, scene_length, stride))

            # Add backward frames if needed
            if remaining_needed > 0:
                start = scene_index - remaining_needed * stride
                backward_indices = list(range(start, scene_index, stride))
                sequence_indices.extend(backward_indices)

            # Final safeguard to enforce video_length
            if len(sequence_indices) > self.video_length:
                sequence_indices = sequence_indices[:self.video_length]
            elif len(sequence_indices) < self.video_length:
                # repeat the last frame 
                sequence_indices.append(sequence_indices[-1])
        
        # Get the base offset for this scene in the flat list
        scene_start_idx = pair_idx - scene_index  # This gives us the index where this scene starts
        
        
        
        # Load all frames in sequence
        images = []
        depths = []
        # Transform scene-relative indices to dataset-relative indices
        sequence_indices = [scene_start_idx + s for s in sequence_indices]
        for seq_i, seq_idx in enumerate(sequence_indices):        
            try:
                # pair = self.pairslist[dataset_idx][seq_idx]
                pair = dataset_list[seq_idx]
            except Exception as e:
                print("dataset, pair idx: ", dataset_idx, pair_idx)
                print(f"seq indices: {sequence_indices}")
                print("pairslist len: ", len(self.pairslist[dataset_idx]))
                dist.barrier()
            image, _current_crop = _load_and_process_image(pair['image'], **self.reshape_list[dataset_idx], color_aug=self.color_aug)
            print_depth_minmax = False #seq_i == 0
            depth = self.depth_read_list[dataset_idx](pair['depth'], is_inverse=True,  print_minmax=print_depth_minmax) # needed for scaling focal length, currently only for Spring
            depth = _load_and_process_depth(depth, image.shape, _current_crop, **self.reshape_list[dataset_idx])
            images.append(image)
            depths.append(depth)
            
        try:
            images = torch.stack(images, dim=0)  # [T, C, H, W]
            depths = torch.stack(depths, dim=0) if self.split != 'test' else None  # [T, H, W]
        except Exception as e:
            import ipdb; ipdb.set_trace()
            dist.barrier()
        
        return images.float(), depths, dataset_idx #, pair['scene_name']
        
if __name__ == '__main__':
    import sys
    from tqdm import tqdm
    sys.path.append('/root/gene/video-depth/')
    sys.path.append('/root/gene/video-depth/dataloaders/')
    from utils.helpers import *
    from utils import logging_config
    from PIL import Image
    import os
    from os.path import join

    # from spring_dataset import SpringDepth
    from depthanything_preprocess import _load_and_process_image, _load_and_process_depth
    # from mvssynth_dataset import MvsSynthDepth
    # from urbansyn_dataset import UrbanSynDepth
    # from eth3d_dataset import Eth3dDepth
    # from pointodyssey_dataset import PointOdysseyDepth
    # from dynamicreplica_dataset import DynamicReplicaDepth
    # from tartanair_dataset import TartanairDepth
    # from waymo_dataset import WaymoDepth
    # from unreal4k_dataset import Unreal4kDepth


    logging_config.configure_logging()

    # use whichever one is faster
    root_dir = '/fsx_scanline/from_eyeline/gene/rgbd_data/'
    root_dir2= '/root/gene/data'
    dataset = CombinedDataset(root_dir=root_dir2, enable_dataset_flags=['unreal4k'],
                              split='val', crop_type='center', resolution=(518,518))

    #dataset = Unreal4kDepth(root_dir='/root/gene/data', split='train', load_cache=None)

    savedir = 'vis_unreal4k-metric'
    os.makedirs(savedir, exist_ok=True)


    for idx, batch in tqdm(enumerate(dataset)):
        video, gt_depth, dataset_name = batch 
        video = video[::100,...]
        gt_depth = gt_depth[::100,...]
        print("video shape: ", video.shape)
        video_save = torch_batch_to_np_arr(video)

        gt_save = vis_depth_metric(gt_depth, input_is_inverse=True)

        # gt_save = depth_to_np_arr(gt_depth)
        
        grid = save_gifs_as_grid(video_save,gt_frames=None, pred_frames=gt_save, 
        output_path=join(savedir, f'{idx}.gif'), fixed_height=280)