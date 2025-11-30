#!/usr/bin/env python3
"""
Depth Completion Script for ETH3D and Waymo_seg Datasets

Implements IP-Basic algorithm with Guided Filter refinement for
sparse-to-dense depth completion.

IP-Basic: Morphological operations + multi-scale hole filling
Guided Filter: Edge-aware smoothing using RGB as guide

References:
- IP-Basic: https://github.com/kujason/ip_basic (KITTI depth completion)
- Guided Filter: He et al., "Guided Image Filtering", ECCV 2010
"""

import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


def guided_filter(guide, src, radius=8, eps=0.01):
    """
    Guided filter for edge-aware smoothing.

    Args:
        guide: Guide image (H, W, 3) or (H, W), values in [0, 1]
        src: Source image to filter (H, W), values in any range
        radius: Filter radius
        eps: Regularization parameter (larger = more smoothing)

    Returns:
        Filtered image (H, W)
    """
    if guide.ndim == 3:
        # Convert to grayscale for simplicity
        guide = cv2.cvtColor((guide * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    guide = guide.astype(np.float32)
    src = src.astype(np.float32)

    # Mean filter
    mean_guide = cv2.boxFilter(guide, -1, (radius, radius))
    mean_src = cv2.boxFilter(src, -1, (radius, radius))
    mean_guide_src = cv2.boxFilter(guide * src, -1, (radius, radius))
    mean_guide_guide = cv2.boxFilter(guide * guide, -1, (radius, radius))

    # Covariance and variance
    cov_guide_src = mean_guide_src - mean_guide * mean_src
    var_guide = mean_guide_guide - mean_guide * mean_guide

    # Linear coefficients
    a = cov_guide_src / (var_guide + eps)
    b = mean_src - a * mean_guide

    # Mean of coefficients
    mean_a = cv2.boxFilter(a, -1, (radius, radius))
    mean_b = cv2.boxFilter(b, -1, (radius, radius))

    # Output
    output = mean_a * guide + mean_b

    return output


def ip_basic_completion(depth, max_depth=100.0):
    """
    IP-Basic depth completion algorithm.

    Uses morphological operations and multi-scale processing to fill
    holes in sparse depth maps. Proven effective on KITTI LiDAR data.

    Args:
        depth: Sparse depth map (H, W), 0 or negative = invalid
        max_depth: Maximum valid depth value for normalization

    Returns:
        Completed depth map (H, W)
    """
    # Create valid mask
    valid_mask = (depth > 0) & np.isfinite(depth)

    # Normalize depth to [0, 1] for processing
    depth_normalized = depth.copy()
    depth_normalized[~valid_mask] = 0
    depth_normalized = np.clip(depth_normalized / max_depth, 0, 1)

    # Convert to uint16 for morphological operations (more precision than uint8)
    depth_uint16 = (depth_normalized * 65535).astype(np.uint16)

    # ===== Stage 1: Diamond dilation to expand sparse points =====
    # Small diamond kernel for initial expansion
    kernel_diamond = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    # Dilate multiple times to expand sparse depth
    dilated = depth_uint16.copy()
    for _ in range(5):
        dilated = cv2.dilate(dilated, kernel_diamond)

    # Only keep dilated values where original was empty
    depth_filled = np.where(valid_mask, depth_uint16, dilated)

    # ===== Stage 2: Multi-scale hole filling =====
    # Use different kernel sizes to fill holes of various sizes
    kernel_sizes = [5, 7, 11, 15, 23, 31]

    for ksize in kernel_sizes:
        # Create circular kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

        # Close operation (dilation + erosion) to fill small holes
        closed = cv2.morphologyEx(depth_filled, cv2.MORPH_CLOSE, kernel)

        # Only update where still empty
        still_empty = depth_filled == 0
        depth_filled = np.where(still_empty, closed, depth_filled)

    # ===== Stage 3: Large hole filling with inpainting =====
    # Find remaining holes
    remaining_holes = depth_filled == 0
    if remaining_holes.any():
        # Use Navier-Stokes based inpainting for large holes
        depth_uint8 = (depth_filled / 256).astype(np.uint8)
        hole_mask = remaining_holes.astype(np.uint8) * 255

        # Inpaint
        inpainted = cv2.inpaint(depth_uint8, hole_mask, inpaintRadius=10, flags=cv2.INPAINT_NS)

        # Merge back (convert back to uint16 scale)
        depth_filled = np.where(remaining_holes, inpainted.astype(np.uint16) * 256, depth_filled)

    # ===== Stage 4: Bilateral filter for edge-aware smoothing =====
    depth_float = depth_filled.astype(np.float32) / 65535.0
    depth_smoothed = cv2.bilateralFilter(
        depth_float,
        d=9,              # Diameter of pixel neighborhood
        sigmaColor=0.1,   # Filter sigma in color space
        sigmaSpace=9      # Filter sigma in coordinate space
    )

    # Convert back to original depth scale
    depth_completed = depth_smoothed * max_depth

    # Preserve original valid values
    depth_completed = np.where(valid_mask, depth, depth_completed)

    return depth_completed


def complete_depth_with_guided_filter(depth, rgb_image, max_depth=100.0,
                                       guided_radius=16, guided_eps=0.001):
    """
    Complete sparse depth using IP-Basic + Guided Filter refinement.

    Args:
        depth: Sparse depth map (H, W)
        rgb_image: RGB guide image (H, W, 3), values in [0, 1]
        max_depth: Maximum depth value
        guided_radius: Guided filter radius
        guided_eps: Guided filter regularization

    Returns:
        Completed depth map (H, W)
    """
    # Step 1: IP-Basic completion
    depth_ip_basic = ip_basic_completion(depth, max_depth)

    # Step 2: Guided filter refinement using RGB as guide
    depth_refined = guided_filter(rgb_image, depth_ip_basic,
                                   radius=guided_radius, eps=guided_eps)

    # Preserve original valid values
    valid_mask = (depth > 0) & np.isfinite(depth)
    depth_refined = np.where(valid_mask, depth, depth_refined)

    # Ensure positive values
    depth_refined = np.maximum(depth_refined, 0)

    return depth_refined


# ===== ETH3D Processing =====

def load_eth3d_depth(depth_path, image_size=None):
    """
    Load ETH3D depth file (raw float32 with inf for invalid).

    ETH3D stores depth at a fixed resolution of 8064x3024 regardless of
    original image size. We need to resize to match the target image.

    Args:
        depth_path: Path to depth file
        image_size: (W, H) tuple for the target size

    Returns:
        depth: (H, W) array, inf values converted to 0
    """
    with open(depth_path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.float32)

    # ETH3D stores depth at fixed resolution 8064x3024
    ETH3D_DEPTH_WIDTH = 8064
    ETH3D_DEPTH_HEIGHT = 3024
    expected_size = ETH3D_DEPTH_WIDTH * ETH3D_DEPTH_HEIGHT

    if len(data) == expected_size:
        depth = data.reshape((ETH3D_DEPTH_HEIGHT, ETH3D_DEPTH_WIDTH))
    else:
        # Fallback: try to find dimensions
        n = len(data)
        found = False
        for h in range(2000, 5000):
            if n % h == 0:
                w = n // h
                if 1.5 < w / h < 4.0:  # Reasonable landscape aspect ratio
                    depth = data.reshape((h, w))
                    found = True
                    break
        if not found:
            raise ValueError(f"Cannot determine dimensions for depth data of size {len(data)}")

    # Convert inf to 0 (invalid)
    depth = np.where(np.isinf(depth), 0, depth)

    # Resize to target image size if provided
    if image_size is not None:
        W, H = image_size
        if depth.shape != (H, W):
            # Use INTER_NEAREST to preserve depth values at sparse pixels
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)

    return depth


def save_eth3d_depth(depth, output_path):
    """Save ETH3D depth as float32 binary (same format as original)."""
    depth_save = depth.astype(np.float32)
    # Mark invalid pixels with inf
    depth_save = np.where(depth_save <= 0, np.inf, depth_save)
    depth_save.tofile(output_path)


def process_eth3d_scene(scene_path, output_suffix="_completed"):
    """
    Process a single ETH3D scene.

    Args:
        scene_path: Path to scene directory (e.g., /path/eth3d/courtyard)
        output_suffix: Suffix for output directory
    """
    scene_path = Path(scene_path)
    depth_dir = scene_path / "ground_truth_depth" / "dslr_images"
    image_dir = scene_path / "images" / "dslr_images"
    output_dir = scene_path / f"ground_truth_depth{output_suffix}" / "dslr_images"

    if not depth_dir.exists():
        print(f"Depth directory not found: {depth_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    depth_files = sorted(depth_dir.glob("*.JPG"))
    print(f"Processing {len(depth_files)} files in {scene_path.name}")

    for depth_file in tqdm(depth_files, desc=scene_path.name):
        # Load RGB image
        image_file = image_dir / depth_file.name
        if not image_file.exists():
            print(f"Image not found: {image_file}")
            continue

        rgb = np.array(Image.open(image_file)).astype(np.float32) / 255.0
        H, W = rgb.shape[:2]

        # Load sparse depth
        depth = load_eth3d_depth(depth_file, image_size=(W, H))

        # Resize if needed
        if depth.shape != (H, W):
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)

        # Get max depth for normalization
        valid_depth = depth[depth > 0]
        max_depth = np.percentile(valid_depth, 99) if len(valid_depth) > 0 else 100.0
        max_depth = max(max_depth * 1.5, 10.0)  # Add margin

        # Complete depth
        depth_completed = complete_depth_with_guided_filter(
            depth, rgb, max_depth=max_depth,
            guided_radius=16, guided_eps=0.001
        )

        # Save completed depth
        output_file = output_dir / depth_file.name
        save_eth3d_depth(depth_completed, output_file)


# ===== Waymo Processing =====

def load_waymo_depth(depth_path, image_size, return_lidar_y_min=False):
    """
    Load Waymo sparse depth (N, 3) format and convert to dense.

    Args:
        depth_path: Path to .npy file with (N, 3) sparse depth
        image_size: (W, H) tuple
        return_lidar_y_min: If True, also return the minimum y coordinate of LiDAR

    Returns:
        depth: (H, W) dense array, 0 = invalid
        lidar_y_min: (optional) minimum y coordinate where LiDAR data exists
    """
    sparse_depth = np.load(depth_path)  # (N, 3): x, y, depth
    W, H = image_size

    # Get LiDAR y_min before any filtering
    lidar_y_min = int(sparse_depth[:, 1].min()) if len(sparse_depth) > 0 else 0

    # Create dense depth map
    depth = np.zeros((H, W), dtype=np.float32)

    # Fill in sparse values
    x_coords = sparse_depth[:, 0].astype(np.int32)
    y_coords = sparse_depth[:, 1].astype(np.int32)
    depth_values = sparse_depth[:, 2]

    # Clip coordinates to valid range
    valid_mask = (x_coords >= 0) & (x_coords < W) & (y_coords >= 0) & (y_coords < H)
    x_coords = x_coords[valid_mask]
    y_coords = y_coords[valid_mask]
    depth_values = depth_values[valid_mask]

    depth[y_coords, x_coords] = depth_values

    if return_lidar_y_min:
        return depth, lidar_y_min
    return depth


def save_waymo_depth(depth, output_path, original_sparse_path=None):
    """
    Save Waymo depth as dense .npy (H, W) float32.

    Note: We save as dense format for completed depth, not sparse.
    """
    np.save(output_path, depth.astype(np.float32))


def process_waymo_sequence(seq_path, camera="FRONT", output_suffix="_completed"):
    """
    Process a single Waymo sequence.

    Args:
        seq_path: Path to sequence directory
        camera: Camera name (FRONT, FRONT_LEFT, etc.)
        output_suffix: Suffix for output directory
    """
    seq_path = Path(seq_path)
    camera_path = seq_path / camera

    depth_dir = camera_path / "depth"
    image_dir = camera_path / "rgb" / "original"
    output_dir = camera_path / f"depth{output_suffix}"

    if not depth_dir.exists():
        print(f"Depth directory not found: {depth_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    depth_files = sorted(depth_dir.glob("*.npy"))

    for depth_file in tqdm(depth_files, desc=seq_path.name):
        # Load RGB image
        frame_id = depth_file.stem
        image_file = image_dir / f"{frame_id}.jpg"
        if not image_file.exists():
            print(f"Image not found: {image_file}")
            continue

        rgb = np.array(Image.open(image_file)).astype(np.float32) / 255.0
        H, W = rgb.shape[:2]

        # Load sparse depth and convert to dense (also get LiDAR y_min)
        depth, lidar_y_min = load_waymo_depth(depth_file, image_size=(W, H), return_lidar_y_min=True)

        # Get max depth for normalization (Waymo can have large depth values)
        valid_depth = depth[depth > 0]
        max_depth = np.percentile(valid_depth, 99) if len(valid_depth) > 0 else 100.0
        max_depth = max(max_depth * 1.5, 100.0)  # Waymo has larger depth range

        # Complete depth
        depth_completed = complete_depth_with_guided_filter(
            depth, rgb, max_depth=max_depth,
            guided_radius=16, guided_eps=0.001
        )

        # Fill -1 above lidar_y_min (where LiDAR doesn't reach - typically sky)
        # This indicates "no valid depth" in regions without LiDAR coverage
        if lidar_y_min > 0:
            depth_completed[:lidar_y_min, :] = -1.0

        # Save completed depth (as dense npy)
        output_file = output_dir / f"{frame_id}.npy"
        save_waymo_depth(depth_completed, output_file)


def process_eth3d_dataset(dataset_root, num_workers=4):
    """Process all ETH3D scenes."""
    dataset_root = Path(dataset_root)

    # Find all scenes (directories with ground_truth_depth subdirectory)
    scenes = []
    for scene_dir in dataset_root.iterdir():
        if scene_dir.is_dir() and (scene_dir / "ground_truth_depth").exists():
            scenes.append(scene_dir)

    print(f"Found {len(scenes)} ETH3D scenes")

    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            list(tqdm(executor.map(process_eth3d_scene, scenes),
                      total=len(scenes), desc="ETH3D scenes"))
    else:
        for scene in tqdm(scenes, desc="ETH3D scenes"):
            process_eth3d_scene(scene)


def process_waymo_dataset(dataset_root, split="val", num_workers=4):
    """Process all Waymo sequences."""
    dataset_root = Path(dataset_root) / split

    # Find all sequences
    sequences = []
    for seq_dir in dataset_root.iterdir():
        if seq_dir.is_dir() and seq_dir.name.startswith("segment-"):
            sequences.append(seq_dir)

    print(f"Found {len(sequences)} Waymo sequences in {split}")

    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            list(tqdm(executor.map(process_waymo_sequence, sequences),
                      total=len(sequences), desc="Waymo sequences"))
    else:
        for seq in tqdm(sequences, desc="Waymo sequences"):
            process_waymo_sequence(seq)


def visualize_completion(depth_sparse, depth_completed, rgb, output_path, lidar_y_min=None):
    """Create visualization comparing sparse and completed depth.

    Args:
        depth_sparse: Sparse depth map (H, W)
        depth_completed: Completed depth map (H, W), -1 = no LiDAR coverage
        rgb: RGB image (H, W, 3), values in [0, 1]
        output_path: Path to save visualization
        lidar_y_min: Optional y coordinate where LiDAR starts
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # RGB image
    axes[0].imshow(rgb)
    axes[0].set_title("RGB Image")
    axes[0].axis("off")

    # Sparse depth
    valid_mask = depth_sparse > 0
    vmin = np.percentile(depth_sparse[valid_mask], 2) if valid_mask.any() else 0
    vmax = np.percentile(depth_sparse[valid_mask], 98) if valid_mask.any() else 1

    sparse_vis = np.zeros_like(depth_sparse)
    sparse_vis[valid_mask] = depth_sparse[valid_mask]

    im1 = axes[1].imshow(sparse_vis, cmap='turbo', vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Sparse Depth ({valid_mask.sum()/valid_mask.size*100:.1f}% valid)")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # Completed depth - handle -1 values (no LiDAR coverage) as black
    completed_vis = depth_completed.copy()
    no_lidar_mask = depth_completed < 0  # -1 indicates no LiDAR coverage

    # Create RGBA image for proper black handling
    # Normalize valid depth to [0, 1]
    completed_normalized = np.zeros_like(completed_vis)
    valid_completed = (depth_completed > 0) & np.isfinite(depth_completed)
    if valid_completed.any():
        completed_normalized[valid_completed] = np.clip(
            (depth_completed[valid_completed] - vmin) / (vmax - vmin + 1e-8), 0, 1
        )

    # Apply turbo colormap
    cmap = plt.cm.turbo
    completed_rgba = cmap(completed_normalized)

    # Set -1 regions to black
    completed_rgba[no_lidar_mask] = [0, 0, 0, 1]  # Black with full opacity

    im2 = axes[2].imshow(completed_rgba)

    # Calculate valid percentage (excluding -1 regions)
    valid_pct = valid_completed.sum() / depth_completed.size * 100
    no_lidar_pct = no_lidar_mask.sum() / depth_completed.size * 100

    if no_lidar_mask.any():
        axes[2].set_title(f"Completed Depth ({valid_pct:.1f}% valid, {no_lidar_pct:.1f}% no-LiDAR)")
    else:
        axes[2].set_title(f"Completed Depth ({valid_pct:.1f}% valid)")
    axes[2].axis("off")

    # Add colorbar with proper range
    sm = plt.cm.ScalarMappable(cmap='turbo', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    plt.colorbar(sm, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Depth completion for ETH3D and Waymo datasets")
    parser.add_argument("--dataset", type=str, required=True, choices=["eth3d", "waymo", "both"],
                        help="Dataset to process")
    parser.add_argument("--eth3d-root", type=str, default="/home/cvlab/hsy/Datasets/eth3d",
                        help="Path to ETH3D dataset root")
    parser.add_argument("--waymo-root", type=str, default="/home/cvlab/hsy/Datasets/waymo_seg",
                        help="Path to Waymo dataset root")
    parser.add_argument("--waymo-split", type=str, default="val",
                        help="Waymo split to process (train/val)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of parallel workers")
    parser.add_argument("--visualize", action="store_true",
                        help="Create visualization samples")

    args = parser.parse_args()

    if args.dataset in ["eth3d", "both"]:
        print("=" * 50)
        print("Processing ETH3D dataset...")
        print("=" * 50)
        process_eth3d_dataset(args.eth3d_root, num_workers=args.num_workers)

    if args.dataset in ["waymo", "both"]:
        print("=" * 50)
        print("Processing Waymo dataset...")
        print("=" * 50)
        process_waymo_dataset(args.waymo_root, split=args.waymo_split,
                              num_workers=args.num_workers)

    print("Done!")


if __name__ == "__main__":
    main()
