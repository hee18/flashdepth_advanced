#!/usr/bin/env python3
"""
Resize UnrealStereo4K dataset from 3840×2160 to 2112×1188 (0.55× scale)

Usage:
    python scripts/resize_unreal4k.py

Input:  /home/cvlab/hsy/Datasets/unreal4k_original/
Output: /home/cvlab/hsy/Datasets/unreal4k/

Resizes:
- Images: Image0/*.png (RGBA → RGB, 3840×2160 → 2112×1188)
- Depth: Disp0/*.npy (3840×2160 → 2112×1188, nearest interpolation)
- Intrinsics: fx, fy, cx, cy scaled by 0.55

Note: Original intrinsics are fx=1920 for 3840×2160
      Resized intrinsics will be fx=1056 for 2112×1188
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
SCALE_FACTOR = 0.55
ORIGINAL_SIZE = (3840, 2160)  # W, H
TARGET_SIZE = (2112, 1188)    # W, H
ORIGINAL_FX = 1920.0
TARGET_FX = 1056.0  # 1920 * 0.55

SOURCE_ROOT = Path('/home/cvlab/hsy/Datasets/unreal4k_original')
TARGET_ROOT = Path('/home/cvlab/hsy/Datasets/unreal4k')


def resize_image(src_path, dst_path):
    """Resize image from 3840×2160 to 2112×1188"""
    try:
        # Load image (RGBA)
        img = Image.open(src_path)

        # Convert RGBA → RGB (UnrealStereo4K uses RGBA format)
        if img.mode == 'RGBA':
            # Create white background
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])  # Alpha channel as mask
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize with high-quality Lanczos filter
        img_resized = img.resize(TARGET_SIZE, Image.LANCZOS)

        # Save as PNG
        img_resized.save(dst_path, 'PNG', compress_level=6)
        return True
    except Exception as e:
        logger.error(f"Error resizing image {src_path}: {e}")
        return False


def resize_depth(src_path, dst_path):
    """Resize depth from 3840×2160 to 2112×1188 (nearest interpolation)"""
    try:
        # Load depth (metric depth in meters)
        depth = np.load(src_path)

        # Resize with nearest interpolation (preserve depth values)
        depth_resized = cv2.resize(
            depth,
            TARGET_SIZE,
            interpolation=cv2.INTER_NEAREST
        )

        # Save as .npy
        np.save(dst_path, depth_resized)
        return True
    except Exception as e:
        logger.error(f"Error resizing depth {src_path}: {e}")
        return False


def process_frame(frame_idx, scene_name, src_scene_dir, dst_scene_dir):
    """Process a single frame (image + depth)"""
    frame_name = f"{frame_idx:05d}"

    # Image: Image0/XXXXX.png
    src_img = src_scene_dir / 'Image0' / f"{frame_name}.png"
    dst_img = dst_scene_dir / 'Image0' / f"{frame_name}.png"

    # Depth: Disp0/XXXXX.npy
    src_depth = src_scene_dir / 'Disp0' / f"{frame_name}.npy"
    dst_depth = dst_scene_dir / 'Disp0' / f"{frame_name}.npy"

    # Skip if both files already exist (resume support)
    if dst_img.exists() and dst_depth.exists():
        return True

    success = True

    # Resize image
    if src_img.exists():
        if not resize_image(src_img, dst_img):
            success = False
    else:
        logger.warning(f"Missing image: {src_img}")
        success = False

    # Resize depth
    if src_depth.exists():
        if not resize_depth(src_depth, dst_depth):
            success = False
    else:
        logger.warning(f"Missing depth: {src_depth}")
        success = False

    return success


def process_scene(scene_name):
    """Process a single scene"""
    logger.info(f"Processing scene: {scene_name}")

    src_scene_dir = SOURCE_ROOT / scene_name
    dst_scene_dir = TARGET_ROOT / scene_name

    # Create output directories
    (dst_scene_dir / 'Image0').mkdir(parents=True, exist_ok=True)
    (dst_scene_dir / 'Disp0').mkdir(parents=True, exist_ok=True)

    # Get list of frames
    src_img_dir = src_scene_dir / 'Image0'
    frame_files = sorted([f for f in src_img_dir.glob('*.png')])
    frame_indices = [int(f.stem) for f in frame_files]

    logger.info(f"  Found {len(frame_indices)} frames in {scene_name}")

    # Process frames sequentially with progress bar
    success_count = 0
    with tqdm(total=len(frame_indices), desc=f"  {scene_name}", unit='frame') as pbar:
        for frame_idx in frame_indices:
            success = process_frame(frame_idx, scene_name, src_scene_dir, dst_scene_dir)
            if success:
                success_count += 1
            pbar.update(1)

    logger.info(f"  {scene_name}: {success_count}/{len(frame_indices)} frames processed successfully")
    return success_count, len(frame_indices)


def main():
    """Main preprocessing function"""
    logger.info("=" * 80)
    logger.info("UnrealStereo4K Dataset Resizing")
    logger.info("=" * 80)
    logger.info(f"Source: {SOURCE_ROOT}")
    logger.info(f"Target: {TARGET_ROOT}")
    logger.info(f"Scale factor: {SCALE_FACTOR}")
    logger.info(f"Original size: {ORIGINAL_SIZE[0]}×{ORIGINAL_SIZE[1]}")
    logger.info(f"Target size: {TARGET_SIZE[0]}×{TARGET_SIZE[1]}")
    logger.info(f"Original fx: {ORIGINAL_FX}")
    logger.info(f"Target fx: {TARGET_FX}")
    logger.info("=" * 80)

    # Check source directory exists
    if not SOURCE_ROOT.exists():
        logger.error(f"Source directory not found: {SOURCE_ROOT}")
        logger.error("Please ensure unreal4k is renamed to unreal4k_original")
        return

    # Create target root directory
    TARGET_ROOT.mkdir(parents=True, exist_ok=True)

    # Get list of scenes
    scenes = sorted([d.name for d in SOURCE_ROOT.iterdir() if d.is_dir()])
    logger.info(f"Found {len(scenes)} scenes: {scenes}")
    logger.info("")

    # Process each scene
    total_success = 0
    total_frames = 0

    for scene_name in scenes:
        success_count, frame_count = process_scene(scene_name)
        total_success += success_count
        total_frames += frame_count
        logger.info("")

    # Summary
    logger.info("=" * 80)
    logger.info("Processing Complete!")
    logger.info("=" * 80)
    logger.info(f"Total scenes: {len(scenes)}")
    logger.info(f"Total frames: {total_frames}")
    logger.info(f"Successful: {total_success}")
    logger.info(f"Failed: {total_frames - total_success}")
    logger.info(f"Success rate: {100.0 * total_success / total_frames:.2f}%")
    logger.info("")
    logger.info(f"Output directory: {TARGET_ROOT}")
    logger.info(f"Intrinsics updated: fx={TARGET_FX} (scaled from {ORIGINAL_FX})")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
