#!/usr/bin/env python3
"""
NYU Depth V2 Preprocessing Script

Converts NYU H5 files to FlashDepth-compatible format:
- RGB images: PNG format (640x480)
- Depth: PNG format encoded as uint16 millimeters (640x480)
- Organizes into pseudo-sequences based on file indices
- Generates fixed intrinsics file

Dataset info:
- 654 validation images
- Indoor scenes
- Metric depth range: ~1m to ~10m
- Fixed camera intrinsics for all frames
"""

import os
import h5py
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import argparse
from pathlib import Path


# NYU Depth V2 fixed camera intrinsics
NYU_FX = 518.86
NYU_FY = 519.47
NYU_CX = 325.58
NYU_CY = 253.74


def load_h5_file(h5_path):
    """
    Load RGB and depth from NYU H5 file

    Returns:
        rgb: (H, W, 3) uint8 array
        depth: (H, W) float32 array in meters
    """
    with h5py.File(h5_path, 'r') as f:
        # RGB: (3, H, W) -> (H, W, 3)
        rgb = f['rgb'][:].transpose(1, 2, 0)  # uint8

        # Depth: (H, W) in meters
        depth = f['depth'][:]  # float32

    return rgb, depth


def encode_depth_to_png(depth_meters):
    """
    Convert depth in meters to uint16 PNG (millimeters)

    Args:
        depth_meters: (H, W) float32 depth in meters

    Returns:
        depth_mm: (H, W) uint16 depth in millimeters
    """
    # Convert to millimeters and clip to uint16 range
    depth_mm = (depth_meters * 1000).astype(np.float32)
    depth_mm = np.clip(depth_mm, 0, 65535)
    depth_mm = depth_mm.astype(np.uint16)

    return depth_mm


def save_intrinsics(output_dir):
    """Save fixed intrinsics to text file"""
    intrinsics_path = output_dir / 'intrinsics.txt'
    with open(intrinsics_path, 'w') as f:
        f.write(f"# NYU Depth V2 Fixed Camera Intrinsics\n")
        f.write(f"# Format: fx fy cx cy\n")
        f.write(f"{NYU_FX} {NYU_FY} {NYU_CX} {NYU_CY}\n")
    print(f"Saved intrinsics to {intrinsics_path}")


def group_into_sequences(h5_files, sequence_length=10):
    """
    Group H5 files into pseudo-sequences

    Args:
        h5_files: List of H5 file paths
        sequence_length: Number of frames per sequence

    Returns:
        sequences: List of (sequence_name, file_list) tuples
    """
    sequences = []

    for i in range(0, len(h5_files), sequence_length):
        sequence_files = h5_files[i:i+sequence_length]
        sequence_name = f"seq_{i//sequence_length:03d}"
        sequences.append((sequence_name, sequence_files))

    return sequences


def process_nyu_dataset(input_dir, output_dir, sequence_length=10, visualize=False):
    """
    Process NYU Depth V2 dataset

    Args:
        input_dir: Directory containing H5 files
        output_dir: Output directory for processed data
        sequence_length: Number of frames per pseudo-sequence
        visualize: Whether to save visualization images
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Find all H5 files
    h5_files = sorted(input_path.glob('*.h5'))

    if len(h5_files) == 0:
        print(f"ERROR: No H5 files found in {input_dir}")
        return

    print(f"Found {len(h5_files)} H5 files")
    print(f"Output directory: {output_path}")
    print(f"Sequence length: {sequence_length}")
    print()

    # Save intrinsics
    save_intrinsics(output_path)

    # Group into sequences
    sequences = group_into_sequences(h5_files, sequence_length)
    print(f"Created {len(sequences)} pseudo-sequences")
    print()

    # Process each sequence
    total_frames = 0

    for seq_name, seq_files in tqdm(sequences, desc="Processing sequences"):
        # Create sequence directory
        seq_dir = output_path / 'val' / seq_name
        seq_dir.mkdir(parents=True, exist_ok=True)

        # Process each frame in sequence
        for frame_idx, h5_file in enumerate(seq_files):
            try:
                # Load H5 data
                rgb, depth = load_h5_file(h5_file)

                # Create frame directory
                frame_dir = seq_dir / f"{frame_idx:04d}"
                frame_dir.mkdir(parents=True, exist_ok=True)

                # Save RGB
                rgb_path = frame_dir / 'rgb.png'
                Image.fromarray(rgb).save(rgb_path)

                # Save depth as uint16 PNG (millimeters)
                depth_mm = encode_depth_to_png(depth)
                depth_path = frame_dir / 'depth.png'
                cv2.imwrite(str(depth_path), depth_mm)

                # Optional: Save visualization
                if visualize:
                    vis_dir = frame_dir / 'vis'
                    vis_dir.mkdir(exist_ok=True)

                    # Depth colormap
                    depth_vis = depth.copy()
                    depth_vis[depth_vis == 0] = np.nan
                    depth_vis = np.nan_to_num(depth_vis, nan=0)
                    depth_norm = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min() + 1e-8)
                    depth_color = (plt.cm.viridis(depth_norm)[:, :, :3] * 255).astype(np.uint8)
                    Image.fromarray(depth_color).save(vis_dir / 'depth_vis.png')

                total_frames += 1

            except Exception as e:
                print(f"ERROR processing {h5_file}: {e}")
                import traceback
                traceback.print_exc()
                continue

    print()
    print("="*60)
    print(f"Preprocessing complete!")
    print(f"Total sequences: {len(sequences)}")
    print(f"Total frames processed: {total_frames}")
    print(f"Output directory: {output_path}")
    print("="*60)
    print()
    print("Directory structure:")
    print(f"{output_path}/")
    print(f"  val/")
    print(f"    seq_000/")
    print(f"      0000/")
    print(f"        rgb.png")
    print(f"        depth.png")
    print(f"      0001/...")
    print(f"    seq_001/...")
    print(f"  intrinsics.txt")


def main():
    parser = argparse.ArgumentParser(description='Preprocess NYU Depth V2 H5 files to FlashDepth format')
    parser.add_argument('--input-dir', type=str,
                        default='/home/cvlab/hsy/Datasets/nyuv2/val',
                        help='Directory containing H5 files (default: /home/cvlab/hsy/Datasets/nyuv2/val)')
    parser.add_argument('--output-dir', type=str,
                        default='/home/cvlab/hsy/Datasets/nyuv2_preprocessed',
                        help='Output directory for processed data (default: /home/cvlab/hsy/Datasets/nyuv2_preprocessed)')
    parser.add_argument('--sequence-length', type=int, default=10,
                        help='Number of frames per pseudo-sequence (default: 10)')
    parser.add_argument('--visualize', action='store_true',
                        help='Save depth visualization images')

    args = parser.parse_args()

    process_nyu_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sequence_length=args.sequence_length,
        visualize=args.visualize
    )


if __name__ == '__main__':
    # Optional: Import matplotlib only if visualization is needed
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    main()
