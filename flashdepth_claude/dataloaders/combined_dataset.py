import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import numpy as np
import logging
import os
from os.path import join
import math

from .depthanything_preprocess import _load_and_process_image, _load_and_process_depth
from .base_dataset_pairs import BaseDatasetPairs


class CombinedDataset(Dataset):
    def __init__(self, root_dir, enable_dataset_flags, resolution=None, split='train',
                 video_length=8, seed=42, tmp_res=None, color_aug=False):
        '''
        enable_dataset_flags: list of datasets to use; e.g. ['spring', 'mvs-synth', 'urbansyn', 'eth3d', 'waymo', 'waymo_seg']

        # must have a couple of 2k, preferably dynamic datasets for testing
        current options: eth3d, waymo, waymo_seg, spring
        eth resolution: 6048x4032 (ratio 1.5)
        waymo/waymo_seg: 1920x1280 (ratio 1.5)
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


        
        cache_dir = './dataloaders/pairs_cache' if split != 'test' else None

        self.pairslist = {}
        self.depth_read_list = {}
        self.reshape_list = {}
        self.tmp_res = tmp_res


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
                    if dataset in ['eth3d', 'waymo', 'waymo_seg']:
                        self.reshape_list[dataset]['resolution'] = (784,518)
                    elif dataset in ['sintel']:
                        self.reshape_list[dataset]['resolution'] = (1022,434)
                    elif dataset in ['urbansyn']:
                        self.reshape_list[dataset]['resolution'] = (1036,518)
                    elif dataset in ['unreal4k']:
                        self.reshape_list[dataset]['resolution'] = (924,518)
                    elif dataset in ['tartanair']:
                        self.reshape_list[dataset]['resolution'] = (518, 518)
                        self.reshape_list[dataset]['crop_type'] = 'center' 


        elif resolution == '2k':
            if split == 'train':
                for dataset in self.reshape_list:
                    self.reshape_list[dataset]['resolution'] = (1918, 1078)
                    self.reshape_list[dataset]['crop_type'] = 'random'
                    self.reshape_list[dataset]['stride'] = 2
            else:
                for dataset in self.reshape_list:
                    self.reshape_list[dataset]['crop_type'] = None
                    if dataset in ['eth3d', 'waymo', 'waymo_seg']:
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

        

        

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):

        if self.split == 'val':
            dataset_idx, scene_idx = self.pairs[idx]
            scene = self.pairslist[dataset_idx][scene_idx]

            # Apply video_length limit for validation to ensure consistent batch sizes
            if len(scene) > self.video_length:
                scene = scene[:self.video_length]

            images = []
            depths = []
            for pair in scene:
                # Debug: check if pair is string or dict
                if isinstance(pair, str):
                    continue  # Skip invalid pairs
                elif not isinstance(pair, dict):
                    continue  # Skip non-dict pairs

                try:
                    image, _current_crop = _load_and_process_image(pair['image'], **self.reshape_list[dataset_idx])
                    depth = self.depth_read_list[dataset_idx](pair['depth'], is_inverse=True) # Load inverse depth (1/m) for training
                    # Keep GT at ORIGINAL resolution (like original FlashDepth)
                    # Prediction will be interpolated to GT resolution during validation

                    images.append(image)
                    depths.append(torch.from_numpy(depth).float()) # Keep original resolution
                except Exception as e:
                    print(f"Error loading validation pair: {e}")
                    continue

            # Skip if no valid pairs found
            if len(images) == 0:
                print(f"Warning: No valid pairs found for validation idx {idx}, skipping")
                return None

            return_name = dataset_idx
            return torch.stack(images).float(), torch.stack(depths).float(), return_name


        elif self.split == 'test':
            dataset_idx, scene_idx = self.pairs[idx]
            scene = self.pairslist[dataset_idx][scene_idx]

            # Apply video_length limit for test split to prevent memory issues
            if len(scene) > self.video_length:
                # Take the first video_length frames
                scene = scene[:self.video_length]

            images = []
            depths = []
            for pair in scene:
                image, _current_crop = _load_and_process_image(pair['image'], **self.reshape_list[dataset_idx])
                depth = self.depth_read_list[dataset_idx](pair['depth'], is_inverse=True) # Load inverse depth (1/m) for testing

                images.append(image)
                depths.append(torch.from_numpy(depth).float()) # not resizing depth, using original resolution like train

            return_name = os.path.join(dataset_idx, pair['scene_name'])
            return torch.stack(images).float(), torch.stack(depths).float(), return_name


        # dataset_idx: i-th dataset; e.g. pointodyssey is 0, spring is 1...etc
        # pair_idx: the i-th (img, depth) pair in the dataset, for instance, pair_idx \in [0, 5000] in Spring
        dataset_idx, pair_idx = self.pairs[idx]  
        dataset_list = self.pairslist[dataset_idx]
        pair = dataset_list[pair_idx]


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
                raise e
            image, _current_crop = _load_and_process_image(pair['image'], **self.reshape_list[dataset_idx])
            print_depth_minmax = False #seq_i == 0
            depth = self.depth_read_list[dataset_idx](pair['depth'], is_inverse=True, print_minmax=print_depth_minmax) # Load inverse depth (1/m) for training
            depth = _load_and_process_depth(depth, image.shape, _current_crop, **self.reshape_list[dataset_idx])
            images.append(image)
            depths.append(depth)
            
        try:
            images = torch.stack(images, dim=0)  # [T, C, H, W]
            depths = torch.stack(depths, dim=0) if self.split != 'test' else None  # [T, H, W]
        except Exception as e:
            print(f"Error stacking tensors in dataset {dataset_idx}: {e}")
            print(f"Images length: {len(images)}")
            if self.split != 'test':
                print(f"Depths length: {len(depths)}")
            print(f"Image shapes: {[img.shape if hasattr(img, 'shape') else type(img) for img in images]}")
            if self.split != 'test' and len(depths) > 0:
                print(f"Depth shapes: {[d.shape if hasattr(d, 'shape') else type(d) for d in depths]}")
            raise e
        
        return images.float(), depths, dataset_idx #, pair['scene_name'] #, pair['scene_name'] #, pair['scene_name']
       