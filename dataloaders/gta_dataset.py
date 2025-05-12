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



class GTADepth(Dataset):
    def __init__(self, root_dir, resolution=None, split='train', 
                 video_length=1, crop_type='random', resize_factor=1.0, load_cache=False):
        
        if os.path.exists(os.path.join(load_cache, 'gta_pairs.pkl')):
            logging.info(f"loading cache for gta")
            self.pairs = pickle.load(open(os.path.join(load_cache, 'gta_pairs.pkl'), 'rb'))
        
        else:
            self.root_dir = root_dir
            self.resolution = resolution
            self.split = split
            self.video_length = video_length
            
            self.crop_type = crop_type

            # Get all scenes
            scenes_path = os.path.join(root_dir, 'MonoDepth_HRSD_GTA/GTA')
            all_scenes = [s for s in os.listdir(scenes_path) if os.path.isdir(os.path.join(scenes_path, s)) and 'datarun' in s]
            
            all_scenes = sorted(all_scenes)
            self.scenes = []
            self.pairs = []

            if split == 'val':
                filter_scenes = [] 
            elif split == 'train':
                filter_scenes = []
            else:
                filter_scenes = []

            for item in all_scenes:
                if split != 'train':
                    scene_pairs = []
                if item in filter_scenes:
                    continue
                item_path = os.path.join(scenes_path, item, 'val')
                rgb_path = os.path.join(item_path, 'images')
                depth_path = os.path.join(item_path, 'depths')
                if os.path.isdir(item_path) and (os.path.isdir(depth_path) or split!='train') and os.path.isdir(rgb_path):
                    self.scenes.append(item)
                else:
                    continue


                count = 0
                all_imgs = [f for f in os.listdir(rgb_path) if f.endswith('.png')]
                all_imgs = sorted(all_imgs, key=lambda x: int(os.path.basename(x).split('_left-')[1].split('.png')[0]))
                for img_name in all_imgs:
                    # example rgb: /root/gene/data/dynamic_replica/train/a1e031-7_obj_source_left/images/a1e031-7_obj_source_left-0299.png
                    # example depth: /root/gene/data/dynamic_replica/train/a1e031-7_obj_source_left/depths/a1e031-7_obj_source_left_0299.geometric.png
                    depth_name = img_name.replace('_left-', '_left_').replace('.png', '.geometric.png')
                    if os.path.exists(os.path.join(depth_path, depth_name)) and split == 'train':
                        self.pairs.append({
                            'image': os.path.join(rgb_path, img_name),
                            'depth': os.path.join(depth_path, depth_name),
                            'scene_index': count,
                            'scene_length': len(all_imgs),
                            'scene_name': item
                        })
                        count += 1
                    else:
                        scene_pairs.append({
                            'image': os.path.join(rgb_path, img_name),
                            'depth': os.path.join(depth_path, depth_name),
                            'scene_length': len(all_imgs),
                            'scene_name': item
                        })
                if split != 'train':
                    self.pairs.append(scene_pairs)
            
            if len(self.scenes) == 0:
                raise RuntimeError(f"No valid scenes found in {scenes_path}")

            self.scenes = sorted(self.scenes)
            logging.info(f'number of scenes / length of gta: {len(self.scenes)} / {len(self.pairs)}')
                    
            
            self.resize_factor = resize_factor 
            # if self.crop_type == 'random' and split == 'train':
            #     logging.info(f'using random crop with resize factor {resize_factor}')
            
            logging.info(f"saving cache for gta")
            pickle.dump(self.pairs, open(os.path.join(load_cache, 'gta_pairs.pkl'), 'wb'))
            
            
            

    def __len__(self):
        return len(self.pairs)
        
    def __getitem__(self, idx):

        pass



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


def read_rgb_raw(file_path, width, height):
    """
    Reads an RGB .raw file and returns a numpy array of shape (height, width, 3).
    Assumes each pixel is represented by 3 bytes (R, G, B) in row-major order.
    
    Parameters:
        file_path (str): Path to the RGB .raw file.
        width (int): Width of the image.
        height (int): Height of the image.
        
    Returns:
        np.ndarray: Array of shape (height, width, 3) containing the image data.
    """
    expected_size = width * height * 3
    with open(file_path, 'rb') as f:
        data = f.read()

    import ipdb; ipdb.set_trace()
    if len(data) != expected_size:
        raise ValueError(f"Expected {expected_size} bytes, got {len(data)} bytes.")
    rgb = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
    return rgb

def convertColorRAW2PNG(filepath,dsize=(800,1280)):
    with open(filepath, 'rb') as file:
        colorBuf = file.read()
    color = np.frombuffer(colorBuf, dtype=np.uint8) 
    import ipdb; ipdb.set_trace()
    image = np.reshape(color, (dsize[0], dsize[1], 4), 'C')
    import ipdb; ipdb.set_trace()



def read_depth_raw(file_path, width, height):
    """
    Reads a depth .raw file and returns a numpy array of shape (height, width).
    Assumes each depth value is a 16-bit unsigned integer.
    
    Parameters:
        file_path (str): Path to the depth .raw file.
        width (int): Width of the depth image.
        height (int): Height of the depth image.
        
    Returns:
        np.ndarray: Array of shape (height, width) containing the depth values.
    """
    expected_size = width * height * 2  # 2 bytes per depth value (uint16)
    with open(file_path, 'rb') as f:
        data = f.read()
    if len(data) != expected_size:
        raise ValueError(f"Expected {expected_size} bytes, got {len(data)} bytes.")
    depth = np.frombuffer(data, dtype=np.uint16).reshape((height, width))
    return depth

# Example usage:
if __name__ == '__main__':
    rgb_file = "/root/gene/data/gta/MonoDepth_HRSD_GTA/GTA/datarun1/val/05_04_2021__16_43_48/left-color.raw"
    depth_file = "/root/gene/data/gta/MonoDepth_HRSD_GTA/GTA/datarun1/val/05_04_2021__16_43_48/left-depth.raw"
    img_width = 1920   
    img_height = 1080

    try:
        rgb_image = convertColorRAW2PNG(rgb_file)
        rgb_image = read_rgb_raw(rgb_file, img_width, img_height)
        depth_image = read_depth_raw(depth_file, img_width, img_height)
        print("RGB image shape:", rgb_image.shape)
        print("Depth image shape:", depth_image.shape)
    except Exception as e:
        print("An error occurred:", e)