"""
Convert FlashDepth output mp4 files with inverse colormap.

Usage:
    python utils/convert_mp4_inverse.py <input_dir> [--recursive]
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def invert_colormap(frame):
    """
    Invert the colormap of a frame.

    Args:
        frame: RGB frame (H, W, 3) in uint8

    Returns:
        inverted_frame: RGB frame with inverted colors
    """
    # Invert RGB channels
    inverted = 255 - frame
    return inverted


def convert_video_inverse(input_path, output_path):
    """
    Convert video with inverse colormap.

    Args:
        input_path: Path to input mp4 file
        output_path: Path to output mp4 file
    """
    # Open input video
    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        logger.error(f"Cannot open video: {input_path}")
        return False

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not out.isOpened():
        logger.error(f"Cannot create output video: {output_path}")
        cap.release()
        return False

    # Process frames
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Invert colormap
        inverted_frame = invert_colormap(frame)

        # Write frame
        out.write(inverted_frame)
        frame_count += 1

    # Release resources
    cap.release()
    out.release()

    logger.info(f"✓ Converted {frame_count} frames: {output_path.name}")
    return True


def process_directory(input_dir, recursive=False):
    """
    Process all mp4 files in directory.

    Args:
        input_dir: Directory containing mp4 files
        recursive: If True, search recursively
    """
    input_dir = Path(input_dir)

    if not input_dir.exists():
        logger.error(f"Directory not found: {input_dir}")
        return

    # Find all mp4 files
    if recursive:
        mp4_files = list(input_dir.rglob("*.mp4"))
    else:
        mp4_files = list(input_dir.glob("*.mp4"))

    # Filter out files that already have "_inverse" suffix
    mp4_files = [f for f in mp4_files if "_inverse" not in f.stem]

    if not mp4_files:
        logger.warning(f"No mp4 files found in {input_dir}")
        return

    logger.info(f"Found {len(mp4_files)} mp4 files")

    # Process each file
    success_count = 0
    for mp4_file in mp4_files:
        # Create output path with _inverse suffix
        output_path = mp4_file.parent / f"{mp4_file.stem}_inverse{mp4_file.suffix}"

        if output_path.exists():
            logger.info(f"⊘ Skipping (already exists): {output_path.name}")
            continue

        logger.info(f"Processing: {mp4_file.relative_to(input_dir)}")
        if convert_video_inverse(mp4_file, output_path):
            success_count += 1

    logger.info(f"\n✓ Converted {success_count}/{len(mp4_files)} videos")


def main():
    parser = argparse.ArgumentParser(description="Convert FlashDepth mp4 files with inverse colormap")
    parser.add_argument("input_dir", type=str, help="Directory containing mp4 files")
    parser.add_argument("--recursive", action="store_true", help="Search recursively for mp4 files")

    args = parser.parse_args()

    process_directory(args.input_dir, args.recursive)


if __name__ == "__main__":
    main()
