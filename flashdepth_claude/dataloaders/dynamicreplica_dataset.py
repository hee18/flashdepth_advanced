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
import json
import gzip
from .base_dataset_pairs import BaseDatasetPairs



class DynamicReplicaDepth(BaseDatasetPairs):
    def __init__(self, root_dir, split, load_cache=None):
        self.root_dir = os.path.join(root_dir, 'dynamicreplica/train')
        super().__init__(dataset_name='dynamicreplica', root_dir=self.root_dir, split=split, load_cache=load_cache)
        # Set default parameters
        self.reshape_list['resolution'] = (1280,720)
        self.reshape_list['stride'] = 2

        # Load focal lengths from annotation file
        self.focal_length_cache = self._load_focal_lengths(root_dir)
       

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

    def _load_focal_lengths(self, root_dir):
        """
        Load focal lengths from frame_annotations_train.jgz file.

        The annotation file contains per-frame camera intrinsics in NDC (Normalized Device Coordinates) format.
        We convert NDC focal length to pixel coordinates:
        fx_pixel = fx_ndc × width / 2

        Returns:
            dict: Mapping from image path to focal length in pixels
        """
        annotation_path = os.path.join(root_dir, 'dynamicreplica', 'frame_annotations_train.jgz')
        focal_length_cache = {}

        if not os.path.exists(annotation_path):
            logging.warning(f"DynamicReplica annotation file not found: {annotation_path}")
            return focal_length_cache

        try:
            logging.info(f"Loading DynamicReplica focal lengths from {annotation_path}")
            with gzip.open(annotation_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    frame_data = json.loads(line)

                    # Get image path and size
                    image_path = frame_data['image']['path']
                    width = frame_data['image']['size'][1]  # [height, width]

                    # Get NDC focal length
                    viewpoint = frame_data.get('viewpoint', {})
                    focal_length_ndc = viewpoint.get('focal_length', [None, None])
                    fx_ndc = focal_length_ndc[0]

                    if fx_ndc is not None:
                        # Convert NDC to pixel coordinates
                        fx_pixel = fx_ndc * width / 2.0
                        focal_length_cache[image_path] = fx_pixel

            logging.info(f"Loaded {len(focal_length_cache)} focal lengths from DynamicReplica annotations")

        except Exception as e:
            logging.error(f"Error loading DynamicReplica focal lengths: {e}")

        return focal_length_cache

    def get_focal_length(self, pair, image_shape):
        """
        Get focal length for DynamicReplica dataset.

        Reads from annotation file (frame_annotations_train.jgz) which contains
        per-frame intrinsics in NDC format. Converts to pixel coordinates:
        fx_pixel = fx_ndc × width / 2

        Fallback: fx = width / 2.0 if annotation not found

        Args:
            pair (dict): Data pair with 'image' key containing image path
            image_shape (tuple): (H, W) image shape

        Returns:
            float: Focal length in pixels
        """
        # Try to get from cache first
        img_path = pair.get('image', '')

        # Extract relative path from full path
        # annotation uses format: "sequence_name/images/filename.png"
        if 'dynamicreplica' in img_path:
            # Extract path relative to dynamicreplica/train/
            parts = img_path.split('dynamicreplica/train/')
            if len(parts) > 1:
                relative_path = parts[1]
                if relative_path in self.focal_length_cache:
                    return self.focal_length_cache[relative_path]

        # Fallback: use default pinhole model
        height, width = image_shape
        fx = width / 2.0
        return fx
