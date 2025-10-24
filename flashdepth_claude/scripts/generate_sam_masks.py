"""
Generate segmentation masks using Segment Anything Model (SAM).

This script generates instance segmentation masks for datasets that don't have
segmentation ground truth. Useful for object-wise depth evaluation when only
depth GT is available.

Installation:
    pip install segment-anything
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

Usage:
    # Generate masks for KITTI dataset
    python scripts/generate_sam_masks.py \
        --input-dir /home/cvlab/hsy/Datasets/KITTI/raw \
        --output-dir /home/cvlab/hsy/Datasets/KITTI/segmentation \
        --checkpoint sam_vit_h_4b8939.pth \
        --model-type vit_h \
        --device cuda:0

    # Generate with class filtering (keep only large objects)
    python scripts/generate_sam_masks.py \
        --input-dir /home/cvlab/hsy/Datasets/sintel/final \
        --output-dir /home/cvlab/hsy/Datasets/sintel/segmentation \
        --checkpoint sam_vit_h_4b8939.pth \
        --min-mask-area 1000 \
        --max-masks 50
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import List, Optional

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_sam_installation():
    """Check if SAM is installed and provide installation instructions."""
    try:
        import segment_anything
        return True
    except ImportError:
        logger.error("Segment Anything Model (SAM) not installed!")
        logger.info("Install with: pip install segment-anything")
        logger.info("Download checkpoint: wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        return False


def generate_masks_for_image(
    image_path: Path,
    mask_generator,
    min_mask_area: int = 100,
    max_masks: int = 100,
    stability_score_thresh: float = 0.9
) -> np.ndarray:
    """
    Generate segmentation masks for a single image.

    Args:
        image_path: Path to input image
        mask_generator: SAM mask generator
        min_mask_area: Minimum mask area in pixels
        max_masks: Maximum number of masks to keep
        stability_score_thresh: Minimum stability score for masks

    Returns:
        Segmentation mask (H, W) with instance IDs
    """
    # Load image
    image = np.array(Image.open(image_path).convert('RGB'))
    H, W = image.shape[:2]

    # Generate masks
    masks = mask_generator.generate(image)

    # Filter masks by area and stability
    filtered_masks = []
    for mask in masks:
        area = mask['area']
        stability = mask['stability_score']

        if area >= min_mask_area and stability >= stability_score_thresh:
            filtered_masks.append(mask)

    # Sort by area (largest first)
    filtered_masks = sorted(filtered_masks, key=lambda x: x['area'], reverse=True)

    # Keep only top N masks
    filtered_masks = filtered_masks[:max_masks]

    # Create segmentation mask
    seg_mask = np.zeros((H, W), dtype=np.uint16)

    for i, mask in enumerate(filtered_masks):
        # Instance ID starts from 1 (0 is background)
        instance_id = i + 1
        seg_mask[mask['segmentation']] = instance_id

    return seg_mask


def process_directory(
    input_dir: Path,
    output_dir: Path,
    mask_generator,
    recursive: bool = True,
    min_mask_area: int = 100,
    max_masks: int = 100,
    image_extensions: List[str] = ['.png', '.jpg', '.jpeg']
) -> int:
    """
    Process all images in a directory.

    Args:
        input_dir: Input directory containing images
        output_dir: Output directory for segmentation masks
        mask_generator: SAM mask generator
        recursive: Process subdirectories recursively
        min_mask_area: Minimum mask area
        max_masks: Maximum masks per image
        image_extensions: Valid image file extensions

    Returns:
        Number of images processed
    """
    # Find all image files
    if recursive:
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(input_dir.rglob(f'*{ext}'))
    else:
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(input_dir.glob(f'*{ext}'))

    image_paths = sorted(image_paths)
    logger.info(f"Found {len(image_paths)} images in {input_dir}")

    # Process each image
    num_processed = 0
    for image_path in tqdm(image_paths, desc="Generating masks"):
        # Compute relative path
        rel_path = image_path.relative_to(input_dir)

        # Create output path
        output_path = output_dir / rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip if already exists
        if output_path.exists():
            logger.debug(f"Skipping {rel_path} (already exists)")
            continue

        try:
            # Generate masks
            seg_mask = generate_masks_for_image(
                image_path,
                mask_generator,
                min_mask_area=min_mask_area,
                max_masks=max_masks
            )

            # Save as PNG
            Image.fromarray(seg_mask).save(output_path)
            num_processed += 1

        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            continue

    logger.info(f"Processed {num_processed} images")
    return num_processed


def main():
    parser = argparse.ArgumentParser(description='Generate segmentation masks using SAM')

    # Input/output arguments
    parser.add_argument('--input-dir', type=Path, required=True,
                        help='Input directory containing images')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Output directory for segmentation masks')

    # SAM model arguments
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='Path to SAM checkpoint (e.g., sam_vit_h_4b8939.pth)')
    parser.add_argument('--model-type', type=str, default='vit_h',
                        choices=['vit_h', 'vit_l', 'vit_b'],
                        help='SAM model type (default: vit_h)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run on (default: cuda:0)')

    # Mask generation arguments
    parser.add_argument('--min-mask-area', type=int, default=100,
                        help='Minimum mask area in pixels (default: 100)')
    parser.add_argument('--max-masks', type=int, default=100,
                        help='Maximum masks per image (default: 100)')
    parser.add_argument('--stability-score-thresh', type=float, default=0.9,
                        help='Minimum stability score (default: 0.9)')
    parser.add_argument('--points-per-side', type=int, default=32,
                        help='Grid points per side for mask sampling (default: 32)')
    parser.add_argument('--pred-iou-thresh', type=float, default=0.88,
                        help='Predicted IoU threshold (default: 0.88)')

    # Processing arguments
    parser.add_argument('--recursive', action='store_true',
                        help='Process subdirectories recursively')
    parser.add_argument('--image-extensions', type=str, nargs='+',
                        default=['.png', '.jpg', '.jpeg'],
                        help='Valid image extensions (default: .png .jpg .jpeg)')

    args = parser.parse_args()

    # Check SAM installation
    if not check_sam_installation():
        sys.exit(1)

    # Import SAM after checking installation
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    # Check inputs
    if not args.input_dir.exists():
        logger.error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)

    if not args.checkpoint.exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        logger.info("Download with: wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load SAM model
    logger.info(f"Loading SAM model ({args.model_type}) from {args.checkpoint}")
    sam = sam_model_registry[args.model_type](checkpoint=str(args.checkpoint))
    sam.to(device=args.device)

    # Create mask generator
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        min_mask_region_area=args.min_mask_area
    )

    logger.info("SAM model loaded successfully")

    # Process directory
    num_processed = process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        mask_generator=mask_generator,
        recursive=args.recursive,
        min_mask_area=args.min_mask_area,
        max_masks=args.max_masks,
        image_extensions=args.image_extensions
    )

    logger.info(f"Complete! Processed {num_processed} images")
    logger.info(f"Segmentation masks saved to {args.output_dir}")


if __name__ == "__main__":
    main()
