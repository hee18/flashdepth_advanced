import os
from os.path import join
import cv2
import torch
import numpy as np
import tempfile, shutil
import glob
import logging
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize
from PIL import Image
import torch.distributed as dist
from .depthanything_preprocess import _load_and_process_image, _load_and_process_depth

class RandomDataset(Dataset):
    def __init__(self, root_dir, resolution=None, crop_type=None, large_dir=None):
        self.root_dir = root_dir
        self.resolution = resolution
        self.crop_type = crop_type
        self.large_dir = large_dir

        if large_dir is not None:
            mp4_paths = glob.glob(join(root_dir, '*.mp4'))

            # keep_paths = [
            #     '2795748-uhd_3840_2160_25fps.mp4',
            #     '5192026-hd_1920_1080_30fps.mp4',
            #     '5262568-uhd_3840_2160_25fps.mp4',
            #     '7334676-hd_1920_1080_24fps.mp4',
            #     '7502169-uhd_3840_2160_25fps.mp4',
            #     '7583319-hd_1920_1080_25fps.mp4',
            #     '8746767-uhd_4096_2160_30fps.mp4',
            #     '12123583_3840_2160_25fps.mp4'    
            # ]
            # mp4_paths = [p for p in mp4_paths if os.path.basename(p) in keep_paths]
            
            print(f"Found {mp4_paths} in {root_dir}")
            
            if mp4_paths:
                self.seq_paths = sorted(mp4_paths)
                self.is_mp4 = True

            else:   
                # e.g. root_dir = '/root/gene/video-depth/evaluation/testdata/waymo/scenes/'
                seq_paths = glob.glob(join(root_dir, '*'))
                seq_paths = sorted([ join(seq_path, 'images') for seq_path in seq_paths if os.path.isdir(join(seq_path, 'images'))])
                self.seq_paths = seq_paths
                self.is_mp4 = False

        else:
            img_paths = []
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                img_paths.extend(glob.glob(join(root_dir, ext)))
            img_paths = sorted(img_paths, key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))
            img_paths = img_paths[:1000]
            imgs = []
            for img_path in img_paths:
                img, _current_crop = _load_and_process_image(img_path, resolution=self.resolution, crop_type=self.crop_type)
                imgs.append(img)
            
            self.imgs = torch.stack(imgs).unsqueeze(0).float() # batch dim=1; 1,N,3,H,W
            self.seq_paths = [self.imgs]

        # import ipdb; ipdb.set_trace()
        # torch.save(self.imgs, 'dataloaders/unreal4k_8_imgs.pt')
        # self.imgs = torch.load('dataloaders/unreal4k_8_imgs.pt')
        

    def __len__(self):
        return len(self.seq_paths)
        
    def __getitem__(self, idx):

        if self.large_dir is None:
            return self.imgs[idx]
        else:
            if self.is_mp4:
                img_paths, tmpdirname = self.parse_seq_path(self.seq_paths[idx])
                img_paths = sorted(img_paths, key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))[5:]
                imgs = []

                first_img = cv2.imread(img_paths[0])
                h, w = first_img.shape[:2]
                if max(h, w) > 2044: # set long side to 1024
                    scale = 2044 / max(h, w)
                    res = (int(w * scale), int(h * scale))
                else:
                    res = (w, h)

                for img_path in img_paths:
                    img, _ = _load_and_process_image(img_path, resolution=res, crop_type=None)
                    imgs.append(img)
                
                if tmpdirname is not None:
                    shutil.rmtree(tmpdirname)

                return dict(batch=torch.stack(imgs).float(), 
                        scene_name=os.path.basename(self.seq_paths[idx].split('.')[0]))
            
            else:
                seq_path = self.seq_paths[idx]
                img_paths = []
                for ext in ['*.png', '*.jpg', '*.jpeg', '*.JPG']:
                    img_paths.extend(glob.glob(join(seq_path, ext)))
                img_paths = sorted(img_paths, key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))
                img_paths = img_paths[:1000]
                imgs = []
                for img_path in img_paths:
                    img, _current_crop = _load_and_process_image(img_path, resolution=self.resolution, crop_type=self.crop_type)
                    imgs.append(img)
                
                return dict(batch=torch.stack(imgs).float(), 
                            scene_name=os.path.basename(seq_path.split('/images')[0]))
                

    def parse_seq_path(self, p):
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file {p}")
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if video_fps == 0:
            cap.release()
            raise ValueError(f"Error: Video FPS is 0 for {p}")
        frame_interval = 1
        frame_indices = list(range(0, total_frames, frame_interval))
        print(
            f" - Video FPS: {video_fps}, Frame Interval: {frame_interval}, Total Frames to Read: {len(frame_indices)}"
        )
        img_paths = []
        tmpdirname = tempfile.mkdtemp()
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(tmpdirname, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)
            img_paths.append(frame_path)
        cap.release()
        return img_paths, tmpdirname