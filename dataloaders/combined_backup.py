import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import torch.distributed as dist
import numpy as np
import logging
import os
from os.path import join
import math

try:
    from .pointodyssey_dataset import PointOdysseyDepth
    from .spring_dataset import SpringDepth
    from .depthanything_preprocess import _load_and_process_image, _load_and_process_depth
    from .sintel_dataset import SintelDepth
    from .mvssynth_dataset import MvsSynthDepth
    from .urbansyn_dataset import UrbanSynDepth
    from .eth3d_dataset import Eth3dDepth
    from .waymo_dataset import WaymoDepth
    from .dynamicreplica_dataset import DynamicReplicaDepth
    from .tartanair_dataset import TartanairDepth
    from .unreal4k_dataset import Unreal4kDepth
    from .base_dataset_pairs import BaseDatasetPairs
except:
    pass

class CombinedDataset(Dataset):
    def __init__(self, root_dir, enable_dataset_flags, resolution=None, split='train', stride=1,
                 video_length=8, crop_type='random', resize_factor=1.0, seed=42, tmp_res=None):
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

        # basically, the 2k datasets help with high res, and the dynamic datasets help with consistency
        # I'm considering removing urbansyn because it's not categorized by scene, so might be hard for mamba? (might be fine because metric)


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


        
        cache_dir = '/root/gene/video-depth/dataloaders/pairs_cache'
        dataset_args = {
            'split': split,
            'load_cache': cache_dir
        }

        


        self.pairslist = {}
        self.depth_read_list = {}
        self.reshape_list = {}
        self.tmp_res = tmp_res

        ## testing sets (don't need same resolution because batch size always 1)
        # unreal4k 7 scenes; urbansyn 1 scene
        # sintel 23 -> 16 scenes
        # waymo 50 -> 8 scenes

        for dataset_name in enable_dataset_flags:
            dataset = BaseDatasetPairs(dataset_name, root_dir, split, load_cache=cache_dir)
            self.pairslist[dataset_name] = dataset.pairs
            self.depth_read_list[dataset_name] = dataset.depth_read

        if 'eth3d' in enable_dataset_flags:
            eth3d = Eth3dDepth(join(root_dir, 'eth3d'), **dataset_args)
            self.pairslist['eth3d'] = eth3d.pairs
            self.depth_read_list['eth3d'] = eth3d.depth_read
            self.reshape_list['eth3d'] = {'resolution': (1904,1232), 'crop_type': 'center', 'stride': 2, 'resize_factor': 1}

        if 'waymo' in enable_dataset_flags:
            waymo = WaymoDepth(join(root_dir, 'waymo'), **dataset_args)
            self.pairslist['waymo'] = waymo.pairs
            self.depth_read_list['waymo'] = waymo.depth_read
            self.reshape_list['waymo'] = {'resolution': (1904,1232), 'crop_type': 'center', 'stride': 2, 'resize_factor': 1}

        if 'unreal4k' in enable_dataset_flags:
            unreal4k = Unreal4kDepth(join(root_dir, 'unrealstereo4k'), **dataset_args)
            self.pairslist['unreal4k'] = unreal4k.pairs
            self.depth_read_list['unreal4k'] = unreal4k.depth_read # 1920x1080
            self.reshape_list['unreal4k'] = {'resolution': (1904,1064), 'crop_type': 'center', 'stride': 2, 'resize_factor': 1}

        if 'sintel' in enable_dataset_flags:
            sintel = SintelDepth(join(root_dir, 'sintel'), **dataset_args)
            self.pairslist['sintel'] = sintel.pairs
            self.depth_read_list['sintel'] = sintel.depth_read
            self.reshape_list['sintel'] = {'resolution': (1022,434), 'crop_type': None, 'stride': 2, 'resize_factor': 1}

        if 'urbansyn' in enable_dataset_flags:
            urbansyn = UrbanSynDepth(join(root_dir, 'urbansyn'), **dataset_args)
            self.pairslist['urbansyn'] = urbansyn.pairs
            self.depth_read_list['urbansyn'] = urbansyn.depth_read
            self.reshape_list['urbansyn'] = {'resolution': (2016,1008), 'crop_type': None, 'stride': 1, 'resize_factor': 1}

        ## training sets
        if 'spring' in enable_dataset_flags:
            spring = SpringDepth(join(root_dir, 'spring'),  **dataset_args)
            self.pairslist['spring'] = spring.pairs
            self.depth_read_list['spring'] = spring.depth_read
            self.reshape_list['spring'] = {'resolution': (1960,1120), 'crop_type': None, 'stride': 4, 'resize_factor': 1}
            
        if 'mvs-synth' in enable_dataset_flags:
            mvssynth = MvsSynthDepth(join(root_dir, 'mvs-synth'), **dataset_args)
            self.pairslist['mvs-synth'] = mvssynth.pairs
            self.depth_read_list['mvs-synth'] = mvssynth.depth_read
            self.reshape_list['mvs-synth'] = {'resolution': (1960,1120), 'crop_type': None, 'stride': 2, 'resize_factor': 1}

            
        if 'pointodyssey' in enable_dataset_flags:
            pointodyssey = PointOdysseyDepth(join(root_dir, 'pointodyssey'), **dataset_args)
            self.pairslist['pointodyssey'] = pointodyssey.pairs
            self.depth_read_list['pointodyssey'] = pointodyssey.depth_read
            self.reshape_list['pointodyssey'] = {'resolution': (1008,560), 'crop_type': None, 'stride': 4, 'resize_factor': 1}

        if 'dynamicreplica' in enable_dataset_flags:
            dynamicreplica = DynamicReplicaDepth(join(root_dir, 'dynamic_replica'), **dataset_args)
            self.pairslist['dynamicreplica'] = dynamicreplica.pairs
            self.depth_read_list['dynamicreplica'] = dynamicreplica.depth_read
            self.reshape_list['dynamicreplica'] = {'resolution': (1288,728), 'crop_type': None, 'stride': 4, 'resize_factor': 1}

        if 'tartanair' in enable_dataset_flags:
            tartanair = TartanairDepth(join(root_dir, 'tartanair'), **dataset_args)
            self.pairslist['tartanair'] = tartanair.pairs
            self.depth_read_list['tartanair'] = tartanair.depth_read
            self.reshape_list['tartanair'] = {'resolution': (1008, 728), 'crop_type': None, 'stride': 4, 'resize_factor': 1}
        

        if tmp_res == 'base_res' or tmp_res == 'base_res_not_uniform':
            logging.info("\nusing base_res!!\n")
            for dataset in self.reshape_list:
                self.reshape_list[dataset]['resolution'] = (518, 518)
                self.reshape_list[dataset]['crop_type'] = 'random'
            if 'pointodyssey' in enable_dataset_flags:
                self.reshape_list['pointodyssey']['stride'] = 4
                self.reshape_list['pointodyssey']['resize_factor'] = 0.8
            if 'dynamicreplica' in enable_dataset_flags:
                self.reshape_list['dynamicreplica']['stride'] = 4
                self.reshape_list['dynamicreplica']['resize_factor'] = 0.6
            if 'tartanair' in enable_dataset_flags:
                self.reshape_list['tartanair']['stride'] = 4
                # self.reshape_list['tartanair']['resize_factor'] = 0.8
            # if 'urbansyn' in enable_dataset_flags:
            #     self.reshape_list['urbansyn']['stride'] = 1
            #     self.reshape_list['urbansyn']['resize_factor'] = 0.4
            if 'mvs-synth' in enable_dataset_flags:
                self.reshape_list['mvs-synth']['stride'] = 2
                self.reshape_list['mvs-synth']['resize_factor'] = 0.4
            if 'spring' in enable_dataset_flags and split == 'train':
                self.reshape_list['spring']['stride'] = 4
                self.reshape_list['spring']['resize_factor'] = 0.4
            
            if 'spring' in enable_dataset_flags and split == 'val':
                self.reshape_list['spring']['resolution'] = (924, 518)
                self.reshape_list['spring']['crop_type'] = None
            if 'eth3d' in enable_dataset_flags:
                self.reshape_list['eth3d']['resolution'] = (770, 518)
                self.reshape_list['eth3d']['crop_type'] = None
            if 'waymo' in enable_dataset_flags: 
                self.reshape_list['waymo']['resolution'] = (770, 518)
                self.reshape_list['waymo']['crop_type'] = None
            if 'sintel' in enable_dataset_flags:
                self.reshape_list['sintel']['resolution'] = (1022,434)
                self.reshape_list['sintel']['crop_type'] = None
            if 'urbansyn' in enable_dataset_flags:
                self.reshape_list['urbansyn']['resolution'] = (1036,518)
                self.reshape_list['urbansyn']['crop_type'] = None

          

        elif tmp_res == 'tmp_2000':
            logging.info("\nusing tmp_2000!!\n")
            for dataset in self.reshape_list:
                self.reshape_list[dataset]['resolution'] = (2016,1344)
                self.reshape_list[dataset]['crop_type'] = 'center'
        elif tmp_res == 'tmp_1000':
            logging.info("\nusing tmp_1000!!\n")
            for dataset in self.reshape_list:
                self.reshape_list[dataset]['resolution'] = (1344,896)
                self.reshape_list[dataset]['crop_type'] = 'center'
        elif tmp_res == 'tmp_500':
            if 'pointodyssey' in enable_dataset_flags:  
                logging.info("\nusing tmp_500!!\n")
                self.reshape_list['pointodyssey']['resolution'] = (504,280)
                self.reshape_list['pointodyssey']['crop_type'] = 'random'
                self.reshape_list['pointodyssey']['resize_factor'] = 0.6

            if 'dynamicreplica' in enable_dataset_flags:
                self.reshape_list['dynamicreplica']['resolution'] = (504,280) # 1280x720
                self.reshape_list['dynamicreplica']['crop_type'] = 'random' 
                self.reshape_list['dynamicreplica']['resize_factor'] = 0.5

            if 'tartanair' in enable_dataset_flags:
                self.reshape_list['tartanair']['resolution'] = (504,378)
                self.reshape_list['tartanair']['crop_type'] = 'random'
                self.reshape_list['tartanair']['resize_factor'] = 0.9

        elif tmp_res == 'base_res_harder' or tmp_res == 'base_res_harder_not_uniform':
            logging.info("\nusing base_res_harder!!\n")
            for dataset in self.reshape_list:
                self.reshape_list[dataset]['resolution'] = (518, 518)
                self.reshape_list[dataset]['crop_type'] = 'random'
            if 'pointodyssey' in enable_dataset_flags:
                self.reshape_list['pointodyssey']['stride'] = 10
                self.reshape_list['pointodyssey']['resize_factor'] = 1.0
            if 'dynamicreplica' in enable_dataset_flags:
                self.reshape_list['dynamicreplica']['stride'] = 6
                self.reshape_list['dynamicreplica']['resize_factor'] = 0.8
            if 'tartanair' in enable_dataset_flags:
                self.reshape_list['tartanair']['stride'] = 10
            if 'mvs-synth' in enable_dataset_flags:
                self.reshape_list['mvs-synth']['stride'] = 4
                self.reshape_list['mvs-synth']['resize_factor'] = 0.6
            if 'spring' in enable_dataset_flags and split == 'train':
                self.reshape_list['spring']['stride'] = 6
                self.reshape_list['spring']['resize_factor'] = 0.6
                
            if 'spring' in enable_dataset_flags and split == 'val':
                self.reshape_list['spring']['resolution'] = (924, 518)
                self.reshape_list['spring']['crop_type'] = None
            if 'eth3d' in enable_dataset_flags:
                self.reshape_list['eth3d']['resolution'] = (770, 518)
                self.reshape_list['eth3d']['crop_type'] = None
            if 'waymo' in enable_dataset_flags: 
                self.reshape_list['waymo']['resolution'] = (770, 518)
                self.reshape_list['waymo']['crop_type'] = None
            if 'sintel' in enable_dataset_flags:
                self.reshape_list['sintel']['resolution'] = (1022,434)
                self.reshape_list['sintel']['crop_type'] = None

            if 'urbansyn' in enable_dataset_flags:
                self.reshape_list['urbansyn']['resolution'] = (1036,518)
                self.reshape_list['urbansyn']['crop_type'] = None

        if 'not_uniform' in tmp_res and split == 'train':
            self.original_dynamicreplica_pairs = self.pairslist['dynamicreplica']
            self.pairslist['dynamicreplica'] = self.pairslist['dynamicreplica'][::2]
            self.original_spring_pairs = self.pairslist['spring']
            self.pairslist['spring']*=10
        
        
        self.pairs = []


        for dataset_name in enable_dataset_flags:
            indices = list(range(len(self.pairslist[dataset_name])))
            self.pairs.extend([(dataset_name, i) for i in indices])
            logging.info(f"length of {dataset_name} for training: {len(self.pairslist[dataset_name])}")

        if split != 'train':    
            logging.info(f"enabled datasets for {split}: {enable_dataset_flags}")
            logging.info(f"length of combined dataset: {len(self.pairs)}")

        # if split == 'train':
        #     self.pairs = self.pairs * 100


        self.video_length = video_length
        # self.crop_type = crop_type
        # self.resolution = resolution
        # self.resize_factor = resize_factor

        self.split = split

        

        

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
                # depth = _load_and_process_depth(depth, image.shape, _current_crop, **self.reshape_list[dataset_idx])

                if self.reshape_list[dataset_idx]['crop_type'] == 'center':
                    h, w = depth.shape
                    start_y, start_x, target_h, target_w = _current_crop
                    # Scale crop coordinates to original depth resolution
                    orig_start_y = int((start_y / image.shape[1]) * h)
                    orig_start_x = int((start_x / image.shape[2]) * w)
                    orig_target_h = int((target_h / image.shape[1]) * h)
                    orig_target_w = int((target_w / image.shape[2]) * w)
                    depth = depth[orig_start_y:orig_start_y + orig_target_h, 
                                orig_start_x:orig_start_x + orig_target_w]
        
                images.append(image)
                depths.append(torch.from_numpy(depth).float()) # not resizing depth, using original resolution
            return torch.stack(images).float(), torch.stack(depths).float(), dataset_idx

        # dataset_idx: i-th dataset; e.g. pointodyssey is 0, spring is 1...etc
        # pair_idx: the i-th (img, depth) pair in the dataset, for instance, pair_idx \in [0, 5000] in Spring
        dataset_idx, pair_idx = self.pairs[idx]

        if 'not_uniform' in self.tmp_res:
            if dataset_idx == 'dynamicreplica':
                pair_idx = pair_idx * 2  # Reverse the [::2] effect
                pair = self.original_dynamicreplica_pairs[pair_idx]
                dataset_list = self.original_dynamicreplica_pairs
            elif dataset_idx == 'spring':
                pair_idx = pair_idx % len(self.original_spring_pairs)
                pair = self.original_spring_pairs[pair_idx]
                dataset_list = self.original_spring_pairs
            else:
                dataset_list = self.pairslist[dataset_idx]
                pair = dataset_list[pair_idx]
        else:   
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
            image, _current_crop = _load_and_process_image(pair['image'], **self.reshape_list[dataset_idx])
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
    from utils.helpers import *
    from utils import logging_config
    from PIL import Image
    import os
    from os.path import join

    from spring_dataset import SpringDepth
    from depthanything_preprocess import _load_and_process_image, _load_and_process_depth
    from mvssynth_dataset import MvsSynthDepth
    from urbansyn_dataset import UrbanSynDepth
    from eth3d_dataset import Eth3dDepth
    from pointodyssey_dataset import PointOdysseyDepth
    from dynamicreplica_dataset import DynamicReplicaDepth
    from tartanair_dataset import TartanairDepth
    from waymo_dataset import WaymoDepth
    from unreal4k_dataset import Unreal4kDepth

    logging_config.configure_logging()

    # use whichever one is faster
    root_dir = '/fsx_scanline/from_eyeline/gene/rgbd_data/'
    root_dir2= '/root/gene/data'
    dataset = CombinedDataset(root_dir=root_dir2, enable_dataset_flags=['unreal4k'],
                              split='val', crop_type=None, resolution=(910,518))



    savedir = 'vis_unreal4k-depth'
    os.makedirs(savedir, exist_ok=True)


    for idx, batch in tqdm(enumerate(dataset)):
        video, gt_depth, dataset_name = batch 
        video = video[::100,...]
        gt_depth = gt_depth[::100,...]
        print("video shape: ", video.shape)
        video_save = torch_batch_to_np_arr(video)

        #gt_save = vis_depth_metric(gt_depth, input_is_inverse=True)

        gt_save = depth_to_np_arr(gt_depth)
        
        grid = save_gifs_as_grid(video_save,gt_frames=None, pred_frames=gt_save, 
        output_path=join(savedir, f'{idx}.gif'), fixed_height=280)