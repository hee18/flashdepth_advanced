import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import sys
import json
import argparse
sys.path.append('/root/gene/video-depth/')
from utils.eval_metrics.metrics import compute_depth_metrics


def resize_batch(batch, hw):
    """
    Resize a batch of images/depths using bilinear interpolation
    Args:
        batch: NumPy array of shape (B, H, W) or (B, H, W, C)
        size: tuple of (height, width)
    Returns:
        Resized batch with shape (B, new_H, new_W) or (B, new_H, new_W, C)
    """
    h, w = hw
    resized = []
    for i in range(batch.shape[0]):
        img = batch[i]
        # cv2.resize expects (w, h) order
        resized.append(cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR))
    return np.stack(resized)


def load_and_process_depths(depth_files, gt_shape):
    """
    Load depth files and resize them to match ground truth shape
    """
    depths = []
    for depth_file in depth_files:
        depths.append(np.load(depth_file))
    depths = np.stack(depths)
    return resize_batch(depths, gt_shape[-2:])


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate depth estimation methods')
    parser.add_argument('--src_base', type=str, required=True,
                        help='Base directory containing scene folders with ground truth depths')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save the output metrics JSON file')
    parser.add_argument('--methods', type=str, nargs='+', default='flashdepth',
                        help='List of method names to evaluate')
    parser.add_argument('--paths', type=str, nargs='+', required=True,
                        help='List of paths corresponding to each method')
    
    args = parser.parse_args()
    
    # Ensure methods and paths have the same length
    if len(args.methods) != len(args.paths):
        raise ValueError("Number of methods must match number of paths")
    
    return args


if __name__ == "__main__":
    args = parse_args()
    
    # Base directory for ground truth
    src_base = args.src_base
    
    # Create a dictionary mapping method names to their paths
    method_paths = dict(zip(args.methods, args.paths))
    
    # Get all scene directories
    scene_dirs = [d for d in os.listdir(src_base) 
                 if os.path.isdir(os.path.join(src_base, d))]
    scene_dirs = sorted(scene_dirs)
    
    # Initialize dictionary to store metrics for each method
    all_metrics = {method: {} for method in args.methods}
    
    for scene in tqdm(scene_dirs, desc="Processing scenes", total=len(scene_dirs)):
        src_depth_dir = os.path.join(src_base, scene, "depths")
        
        # Load ground truth depths
        depth_files = sorted(glob(os.path.join(src_depth_dir, "*.npy")),
                           key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))
        
        print(f"Processing {scene} depths...")
        gt_depths = []
        for depth_file in depth_files:
            gt_depths.append(np.load(depth_file))
        gt_depths = np.stack(gt_depths)
        
        # Process each method
        for method in args.methods:
            method_path = method_paths[method]
            method_files = sorted(glob(os.path.join(method_path, scene, "depth_npy_files/*.npy")),
                               key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))
            
            # Load and process depths for this method
            method_depths = load_and_process_depths(method_files, gt_depths.shape)
            
            # Compute metrics
            method_metrics = compute_depth_metrics(method_depths, gt_depths)
            all_metrics[method][scene] = method_metrics
        
        print(f"Completed processing {scene}")

    # Calculate averages across all scenes for each method
    for method in args.methods:
        avg_metrics = {}
        for metric in next(iter(all_metrics[method].values())).keys():
            values = [scene_metrics[metric] for scene_metrics in all_metrics[method].values()]
            avg_metrics[metric] = float(np.mean(values))
        all_metrics[method]['average'] = avg_metrics

    # Create a separate averages dictionary with all methods' averages
    averages = {method: all_metrics[method]['average'] for method in args.methods}
    all_metrics['averages'] = averages

    # Save metrics to JSON file
    output_path = args.output_path
    with open(output_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)

    print(f"All depth processing complete! Metrics saved to {output_path}")


'''
python metrics.py \
--src_base /root/gene/video-depth/evaluation/testdata/sintel/scenes  \
--output_path /root/gene/video-depth/evaluation/testdata/sintel/check-vitl.json \
--methods prev_vitl \
--paths /root/gene/video-depth/configs/mamba/l1-len5-harder-not-uniform/sintel/
'''