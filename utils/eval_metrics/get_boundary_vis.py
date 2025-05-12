import os, sys
import glob
import numpy as np
import cv2
from PIL import Image
import re
from tqdm import tqdm
import heapq
from boundary_metrics import fgbg_depth, SI_boundary_F1

def create_edge_visualization(depth_paths, output_path='edge_visualization.png', spacing=5, threshold=1.15):
    """
    Create a visualization of edge maps from multiple depth maps.
    
    Args:
        depth_paths: List of 5 paths to depth maps (4 predictions + 1 ground truth)
        output_path: Path to save the visualization
        spacing: Spacing between edge maps in pixels
        threshold: Threshold for edge detection
    """
    if len(depth_paths) != 5:
        raise ValueError("Expected 5 depth map paths (4 predictions + 1 ground truth)")
    
    # Load depth maps and compute edge maps
    edge_maps = []
    
    for path in depth_paths:
        depth = np.load(path)
        
        # Get edge masks
        left, top, right, bottom = fgbg_depth(depth, threshold)
        
        # Combine horizontal and vertical edges into single mask
        edges = np.zeros_like(depth, dtype=bool)
        edges[:, :-1] |= left | right
        edges[:-1, :] |= top | bottom
        
        # Convert to uint8 for visualization (0-255)
        edges = (edges * 255).astype(np.uint8)
        
        # Make it 3 channels for consistent concatenation
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        edge_maps.append(edges_rgb)
    
    # Ensure all maps have the same dimensions (resize to match ground truth)
    gt_shape = edge_maps[-1].shape[:2]  # Ground truth is the last one
    for i in range(len(edge_maps) - 1):  # Skip ground truth
        if edge_maps[i].shape[:2] != gt_shape:
            edge_maps[i] = cv2.resize(edge_maps[i], (gt_shape[1], gt_shape[0]), 
                                     interpolation=cv2.INTER_NEAREST)
    
    # Create spacing image
    spacing_img = np.ones((gt_shape[0], spacing, 3), dtype=np.uint8) * 255
    
    # Concatenate images horizontally with spacing
    result = edge_maps[0]
    for i in range(1, len(edge_maps)):
        result = np.hstack((result, spacing_img, edge_maps[i]))
    
    # Save the visualization
    cv2.imwrite(output_path, result)
    
    return result

# Adapted from create_side_by_side.py
datasets = ['eth3d', 'sintel', 'waymo', 'unreal4k', 'urbansyn']

paths = {
    'cut3r': '../../evaluation/baselines/CUT3R/output/{dataset}/{scene}/depth/*.npy', # remember to invert this one
    'depthcrafter': '../../evaluation/baselines/DepthCrafter/outputs/{dataset}/{scene}/depth_npys/*.npy',
    'vda': '../../evaluation/baselines/Video-Depth-Anything/outputs/{dataset}/{scene}/depth_npys/*.npy',
    'ours': '../../configs/third-distill/skip123-block4/{dataset}/{scene}/depth_npy_files/*/*.npy',
    'gt': '../../evaluation/testdata/{dataset}/scenes/{scene}/depths/*.npy',
}

def natural_sort_key(s):
    """Sort strings with numbers in natural order."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def load_and_resize_depth(path, target_size, invert=False):
    """Load and resize a depth map to target size."""
    depth = np.load(path)
    if invert:
        depth = 1.0 / (depth + 1e-8)  # Invert with small epsilon to avoid division by zero
    
    # Resize depth map
    depth_resized = cv2.resize(depth, target_size, interpolation=cv2.INTER_LINEAR)
    return depth_resized

def create_edge_map_comparison(dataset, scene):
    """Create edge map comparison for a dataset/scene, selecting top frames based on different criteria."""
    output_dir = f'vis_edge_maps/{dataset}/{scene}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories for different criteria
    diff_dir = os.path.join(output_dir, 'best_diff')
    ours_dir = os.path.join(output_dir, 'best_ours')
    os.makedirs(diff_dir, exist_ok=True)
    os.makedirs(ours_dir, exist_ok=True)
    
    # Get file paths for each method
    method_files = {}
    for method, path_template in paths.items():
        path = path_template.replace('{dataset}', dataset).replace('{scene}', scene)
        files = sorted(glob.glob(path), key=natural_sort_key)
        # Limit to max 100 frames
        if len(files) > 100:
            print(f"Limiting {method} to 100 frames (from {len(files)})")
            files = files[:100]
        if files:
            method_files[method] = files
        else:
            print(f"Warning: No files found for {method} at {path}")
    
    # Check if we have all methods
    required_methods = ['cut3r', 'depthcrafter', 'vda', 'ours', 'gt']
    if not all(method in method_files for method in required_methods):
        missing = [m for m in required_methods if m not in method_files]
        print(f"Skipping {dataset}/{scene} - missing methods: {missing}")
        return
    
    # Determine number of frames (use the minimum count across all methods)
    num_frames = min(len(files) for files in method_files.values())
    print(f"Processing {num_frames} frames for {dataset}/{scene}")
    
    # Calculate F1 scores for each frame and method
    frame_scores = []
    
    for i in tqdm(range(num_frames), desc=f"Calculating F1 scores for {dataset}/{scene}"):
        # Skip if any method doesn't have this frame
        if any(i >= len(method_files[method]) for method in required_methods):
            continue
            
        # Load ground truth depth
        gt_path = method_files['gt'][i]
        gt_depth = np.load(gt_path)
        
        # Calculate F1 scores for each method
        method_scores = {}
        for method in ['cut3r', 'depthcrafter', 'vda', 'ours']:
            method_path = method_files[method][i]
            method_depth = np.load(method_path)
            
            # Invert CUT3R depth
            if method == 'cut3r':
                method_depth = 1.0 / (method_depth + 1e-8)
                
            # Resize if needed
            if method_depth.shape != gt_depth.shape:
                method_depth = cv2.resize(method_depth, 
                                         (gt_depth.shape[1], gt_depth.shape[0]), 
                                         interpolation=cv2.INTER_LINEAR)
                
            # Calculate F1 score
            f1 = SI_boundary_F1(method_depth, gt_depth)
            method_scores[method] = f1
        
        # Calculate difference between ours and vda
        diff = method_scores['ours'] - method_scores['vda']
        
        # Store frame index, difference, ours score, and all scores
        frame_scores.append((i, diff, method_scores['ours'], method_scores))
    
    # Create two lists of top frames based on different criteria
    
    # 1. Sort by difference (ours - vda) in descending order
    diff_frames = sorted(frame_scores, key=lambda x: x[1], reverse=True)
    diff_top_frames = diff_frames[:min(10, len(diff_frames))]
    
    # 2. Sort by our F1 score in descending order
    ours_frames = sorted(frame_scores, key=lambda x: x[2], reverse=True)
    ours_top_frames = ours_frames[:min(10, len(ours_frames))]
    
    # Create visualizations for top frames by difference
    for rank, (frame_idx, diff, ours_score, scores) in enumerate(diff_top_frames):
        # Collect paths for this frame
        frame_paths = [
            method_files['cut3r'][frame_idx],
            method_files['depthcrafter'][frame_idx],
            method_files['vda'][frame_idx],
            method_files['ours'][frame_idx],
            method_files['gt'][frame_idx]
        ]
        
        # Create output path with rank, frame index, and score difference
        output_path = os.path.join(diff_dir, f'rank{rank+1:02d}_frame{frame_idx:04d}_diff{diff:.4f}.png')
        
        # Create visualization
        create_edge_visualization(frame_paths, output_path)
        
        # Save scores to a text file
        score_path = os.path.join(diff_dir, f'rank{rank+1:02d}_frame{frame_idx:04d}_scores.txt')
        with open(score_path, 'w') as f:
            f.write(f"Frame: {frame_idx}\n")
            f.write(f"Difference (ours - vda): {diff:.4f}\n\n")
            f.write("F1 Scores:\n")
            for method, score in scores.items():
                f.write(f"{method}: {score:.4f}\n")
    
    # Create a summary file for difference criterion
    summary_path = os.path.join(diff_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Top frames where 'ours' outperforms 'vda' for {dataset}/{scene}\n\n")
        for rank, (frame_idx, diff, ours_score, scores) in enumerate(diff_top_frames):
            f.write(f"Rank {rank+1}: Frame {frame_idx}, Diff: {diff:.4f}\n")
            for method, score in scores.items():
                f.write(f"  {method}: {score:.4f}\n")
            f.write("\n")
    
    # Create visualizations for top frames by our F1 score
    for rank, (frame_idx, diff, ours_score, scores) in enumerate(ours_top_frames):
        # Collect paths for this frame
        frame_paths = [
            method_files['cut3r'][frame_idx],
            method_files['depthcrafter'][frame_idx],
            method_files['vda'][frame_idx],
            method_files['ours'][frame_idx],
            method_files['gt'][frame_idx]
        ]
        
        # Create output path with rank, frame index, and our score
        output_path = os.path.join(ours_dir, f'rank{rank+1:02d}_frame{frame_idx:04d}_score{ours_score:.4f}.png')
        
        # Create visualization
        create_edge_visualization(frame_paths, output_path)
        
        # Save scores to a text file
        score_path = os.path.join(ours_dir, f'rank{rank+1:02d}_frame{frame_idx:04d}_scores.txt')
        with open(score_path, 'w') as f:
            f.write(f"Frame: {frame_idx}\n")
            f.write(f"Our F1 Score: {ours_score:.4f}\n\n")
            f.write("All F1 Scores:\n")
            for method, score in scores.items():
                f.write(f"{method}: {score:.4f}\n")
    
    # Create a summary file for our F1 score criterion
    summary_path = os.path.join(ours_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Top frames with highest 'ours' F1 scores for {dataset}/{scene}\n\n")
        for rank, (frame_idx, diff, ours_score, scores) in enumerate(ours_top_frames):
            f.write(f"Rank {rank+1}: Frame {frame_idx}, Our F1: {ours_score:.4f}\n")
            for method, score in scores.items():
                f.write(f"  {method}: {score:.4f}\n")
            f.write("\n")

def process_dataset(dataset):
    """Process all scenes for a given dataset."""
    # Find all scenes for this dataset
    dataset_path = f'../../evaluation/testdata/{dataset}/scenes'
    if not os.path.exists(dataset_path):
        print(f"Dataset path not found: {dataset_path}")
        return
        
    scenes = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    
    # Use tqdm to track progress across scenes
    for scene in tqdm(scenes, desc=f"Processing {dataset} scenes"):
        print(f"Processing {dataset}/{scene}")
        create_edge_map_comparison(dataset, scene)
    
    print(f"{dataset} completed!")

def main():
    """Main function to process datasets."""
    # Create the output directory
    os.makedirs('vis_edge_maps', exist_ok=True)
    
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
        if dataset in datasets:
            process_dataset(dataset)
        else:
            print(f"Unknown dataset: {dataset}")
            print(f"Available datasets: {datasets}")
    else:
        print("Please specify a dataset:")
        print(f"Available datasets: {datasets}")

if __name__ == "__main__":
    main()
