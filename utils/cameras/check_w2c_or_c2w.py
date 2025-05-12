import os, sys, glob, pickle
from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation

from unproject_mesh import unproject_depth, render_view

# submodule_path = ("/sensei-fs-3/users/gchou/video-depth/data")
# assert os.path.exists(submodule_path)
# sys.path.insert(0, submodule_path)
# # sintel_io file is also provided in sintel original site, if paths change later
# from sintel_io import depth_read, cam_read

# '''
# this file is to check if dataset poses are c2w or w2c
# idea: If w2c, then we can take frame 1, unproject using depth and intrinsics, then use c2w (inverse of w2c) to get to world coordinates.
# Then, we can warp the point cloud using the w2c target pose and we should end up with the correct warped image.
# If the warped image is correct, then we know the poses are w2c.

# Details
# 1. img should be in range [0,1], and output warped image should be in range [0,1]
# 2. depthmap should be depth (not inverse depth / disparity) and have shape (h,w), no channel dimension
# 3. intrinsics should be 3x3 matrix and extrinsics should be 4x4 matrix

# Code / example here:
# 1. sintel dataset, turns out w2c is correct 
# 2. intrinsics and h,w all the same, so no need to resize, use different extrinsics...etc
# '''


# img = Image.open('/mnt/localssd/sintel/images/training/clean/temple_2/frame_0001.png').convert('RGB')
# img = np.array(img) / 255.0
# h, w = img.shape[:2]
# depthmap = depth_read('/mnt/localssd/sintel/depth/training/depth/temple_2/frame_0001.dpt')
# intrinsics, w2c_original_pose = cam_read('/mnt/localssd/sintel/depth/training/camdata_left/temple_2/frame_0001.cam')
# w2c_original_pose = np.vstack((w2c_original_pose, np.array([0, 0, 0, 1])))
# c2w_original_pose = np.linalg.inv(w2c_original_pose)
# _, w2c_target_pose = cam_read('/mnt/localssd/sintel/depth/training/camdata_left/temple_2/frame_0050.cam')
# w2c_target_pose = np.vstack((w2c_target_pose, np.array([0, 0, 0, 1])))
# c2w_target_pose = np.linalg.inv(w2c_target_pose)

# mesh = unproject_depth(None, img, depthmap, intrinsics, c2w_original_pose, scale_factor=1.0, add_faces=True, prune_edge_faces=True)
# warped_image, _ = render_view(h, w, intrinsics, c2w_target_pose, mesh)
# Image.fromarray( (warped_image*255).astype(np.uint8) ).save('test.png')



####---------------####### 
# second example
# checking monst3r output 
from os.path import join

index1=20
index2=37

filepath = '/root/gene/video-depth/baselines/monst3r/sintel_out/vis_temple'
img = Image.open(join(filepath, f'frame_{index1:04d}.png')).convert('RGB')
img = np.array(img) / 255.0
h, w = img.shape[:2]
print(f"img shape: {img.shape}")


depthmap = np.load(join(filepath, f'frame_{index1:04d}.npy'))

intrinsics_path = join(filepath, "pred_intrinsics.txt")
intrinsics = np.loadtxt(intrinsics_path)
intrinsics = np.array(intrinsics, np.float32).reshape(-1, 3, 3)[0]

poses_path = join(filepath, "pred_traj.txt")
poses = np.loadtxt(poses_path)
poses = np.array(poses, np.float32)
poses = np.concatenate(
    [
        Rotation.from_quat(np.concatenate([poses[:, 5:], poses[:, 4:5]], -1)).as_matrix(),
        poses[:, 1:4, None],
    ],
    -1,
).astype(np.float32)

# Convert to homogeneous transformation matrices (ensure shape is (N, 4, 4))
num_frames = poses.shape[0]
ones = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (num_frames, 1, 1))
poses = np.concatenate([poses, ones], axis=1)
w2c_original_pose = poses[index1]
w2c_target_pose = poses[index2]
c2w_original_pose = np.linalg.inv(w2c_original_pose)
c2w_target_pose = np.linalg.inv(w2c_target_pose)

mesh = unproject_depth(None, img, depthmap, intrinsics, c2w_original_pose, scale_factor=1.0, add_faces=True, prune_edge_faces=True)
warped_image, _ = render_view(h, w, intrinsics, c2w_target_pose, mesh)
Image.fromarray( (warped_image*255).astype(np.uint8) ).save(join(filepath, f'assuming_w2c_{index1}_warped_to_{index2}.png'))
