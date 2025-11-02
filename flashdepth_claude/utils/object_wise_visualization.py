"""
Object-wise visualization utilities for depth evaluation.

Creates 4x4 grid visualization showing:
- Top classes by pixel count
- Input image, GT depth, Predicted depth, Segmentation overlay
- Per-class metrics comparison
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from pathlib import Path
import cv2
import logging

logger = logging.getLogger(__name__)


def create_object_wise_grid(
    input_image,
    gt_depth,
    pred_depth,
    seg_mask,
    class_metrics,
    class_names_dict,
    output_path,
    top_k=16,
    cmap='turbo'
):
    """
    Create 4x4 grid visualization for object-wise evaluation.

    Args:
        input_image: RGB image (H, W, 3) in [0, 1] float or [0, 255] uint8
        gt_depth: Ground truth depth (H, W) in meters
        pred_depth: Predicted depth (H, W) in meters
        seg_mask: Segmentation mask (H, W) with class IDs
        class_metrics: Dict mapping class names to metrics
        class_names_dict: Dict mapping class IDs to class names
        output_path: Path to save visualization
        top_k: Number of top classes to visualize (default: 16 for 4x4 grid)
        cmap: Colormap for depth visualization
    """
    # Sort classes by pixel count (descending)
    sorted_classes = sorted(
        class_metrics.items(),
        key=lambda x: x[1].get('num_pixels', 0),
        reverse=True
    )[:top_k]

    # Create 4x4 grid
    fig = plt.figure(figsize=(24, 24))
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)

    # Normalize input image if needed
    if input_image.dtype == np.uint8:
        input_image = input_image.astype(np.float32) / 255.0

    # Get depth range for consistent colormap
    valid_gt = gt_depth[gt_depth > 0]
    valid_pred = pred_depth[pred_depth > 0]
    if len(valid_gt) > 0 and len(valid_pred) > 0:
        depth_min = min(valid_gt.min(), valid_pred.min())
        depth_max = max(valid_gt.max(), valid_pred.max())
    else:
        depth_min, depth_max = 0, 100

    # Create visualization for each class
    for idx, (class_name, metrics) in enumerate(sorted_classes):
        row = idx // 4
        col = idx % 4

        # Create subplot
        ax = fig.add_subplot(gs[row, col])

        # Get class mask
        class_id = None
        for cid, cname in class_names_dict.items():
            if cname == class_name:
                class_id = cid
                break

        if class_id is None:
            logger.warning(f"Could not find class ID for {class_name}")
            continue

        class_mask = (seg_mask == class_id)

        # Create overlay visualization
        overlay = create_class_overlay(
            input_image, gt_depth, pred_depth, class_mask,
            depth_min, depth_max, cmap
        )

        ax.imshow(overlay)
        ax.axis('off')

        # Add title with class name and key metrics
        num_pixels = metrics.get('num_pixels', 0)
        mae = metrics.get('mae', 0)
        abs_rel = metrics.get('abs_rel', 0)
        a1 = metrics.get('a1', 0)

        title = f"{class_name.upper()}\n"
        title += f"Pixels: {num_pixels:,}\n"
        title += f"MAE: {mae:.3f}m | AbsRel: {abs_rel:.3f}\n"
        title += f"δ1: {a1:.3f}"

        ax.set_title(title, fontsize=10, fontweight='bold')

    # Add overall title
    fig.suptitle(
        'Object-Wise Depth Evaluation (Top 16 Classes by Pixel Count)',
        fontsize=16,
        fontweight='bold',
        y=0.995
    )

    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved object-wise visualization to {output_path}")


def create_class_overlay(input_image, gt_depth, pred_depth, class_mask, depth_min, depth_max, cmap='turbo'):
    """
    Create 2x2 overlay for a single class:
    - Top-left: Input image (clean, no overlay)
    - Top-right: Object mask (white=object, black=background)
    - Bottom-left: GT depth (class region only)
    - Bottom-right: Predicted depth (class region only)

    Args:
        input_image: RGB image (H, W, 3) in [0, 1]
        gt_depth: Ground truth depth (H, W)
        pred_depth: Predicted depth (H, W)
        class_mask: Boolean mask (H, W) for this class
        depth_min: Minimum depth for colormap
        depth_max: Maximum depth for colormap
        cmap: Colormap name

    Returns:
        Combined overlay image (2H, 2W, 3)
    """
    h, w = input_image.shape[:2]

    # Get colormap
    cmap_func = plt.get_cmap(cmap)

    # 1. Input image (top-left) - clean, no overlay
    input_clean = input_image.copy()

    # 2. Object mask (top-right) - white for object, black for background
    mask_vis = np.zeros((h, w, 3), dtype=np.float32)
    mask_vis[class_mask] = [1.0, 1.0, 1.0]  # White for class region
    mask_vis[~class_mask] = [0.0, 0.0, 0.0]  # Black for background

    # 3. GT depth (bottom-left)
    gt_vis = np.zeros((h, w, 3))
    if class_mask.sum() > 0:
        gt_normalized = np.clip((gt_depth - depth_min) / (depth_max - depth_min + 1e-8), 0, 1)
        gt_colored = cmap_func(gt_normalized)[:, :, :3]
        # Only show class region
        gt_vis[class_mask] = gt_colored[class_mask]

    # 4. Predicted depth (bottom-right)
    pred_vis = np.zeros((h, w, 3))
    if class_mask.sum() > 0:
        pred_normalized = np.clip((pred_depth - depth_min) / (depth_max - depth_min + 1e-8), 0, 1)
        pred_colored = cmap_func(pred_normalized)[:, :, :3]
        # Only show class region
        pred_vis[class_mask] = pred_colored[class_mask]

    # Combine into 2x2 grid
    top_row = np.concatenate([input_clean, mask_vis], axis=1)
    bottom_row = np.concatenate([gt_vis, pred_vis], axis=1)
    combined = np.concatenate([top_row, bottom_row], axis=0)

    return combined


def create_per_class_comparison(
    class_metrics_baseline,
    class_metrics_improved,
    class_names_dict,
    output_path,
    baseline_name="Baseline",
    improved_name="Improved",
    top_k=20
):
    """
    Create bar chart comparing per-class metrics between two models.

    Args:
        class_metrics_baseline: Dict of baseline model metrics
        class_metrics_improved: Dict of improved model metrics
        class_names_dict: Dict mapping class IDs to names
        output_path: Path to save visualization
        baseline_name: Name of baseline model
        improved_name: Name of improved model
        top_k: Number of top classes to show
    """
    # Get common classes
    common_classes = set(class_metrics_baseline.keys()) & set(class_metrics_improved.keys())

    # Sort by baseline pixel count
    sorted_classes = sorted(
        common_classes,
        key=lambda x: class_metrics_baseline[x].get('num_pixels', 0),
        reverse=True
    )[:top_k]

    if len(sorted_classes) == 0:
        logger.warning("No common classes found for comparison")
        return

    # Prepare data
    class_names = [cn.replace('_', ' ').title() for cn in sorted_classes]
    mae_baseline = [class_metrics_baseline[cn]['mae'] for cn in sorted_classes]
    mae_improved = [class_metrics_improved[cn]['mae'] for cn in sorted_classes]
    abs_rel_baseline = [class_metrics_baseline[cn]['abs_rel'] for cn in sorted_classes]
    abs_rel_improved = [class_metrics_improved[cn]['abs_rel'] for cn in sorted_classes]

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    x = np.arange(len(class_names))
    width = 0.35

    # MAE comparison
    ax1.bar(x - width/2, mae_baseline, width, label=baseline_name, alpha=0.8)
    ax1.bar(x + width/2, mae_improved, width, label=improved_name, alpha=0.8)
    ax1.set_xlabel('Class', fontweight='bold')
    ax1.set_ylabel('MAE (meters)', fontweight='bold')
    ax1.set_title('Mean Absolute Error by Class', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # AbsRel comparison
    ax2.bar(x - width/2, abs_rel_baseline, width, label=baseline_name, alpha=0.8)
    ax2.bar(x + width/2, abs_rel_improved, width, label=improved_name, alpha=0.8)
    ax2.set_xlabel('Class', fontweight='bold')
    ax2.set_ylabel('AbsRel', fontweight='bold')
    ax2.set_title('Absolute Relative Error by Class', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved comparison chart to {output_path}")


if __name__ == "__main__":
    # Test visualization with dummy data
    logging.basicConfig(level=logging.INFO)

    # Create dummy data
    h, w = 512, 512
    input_image = np.random.rand(h, w, 3).astype(np.float32)
    gt_depth = np.random.rand(h, w) * 50
    pred_depth = gt_depth + np.random.randn(h, w) * 2  # Add noise
    seg_mask = np.random.randint(0, 10, (h, w))

    # Dummy metrics
    class_metrics = {
        'vehicle': {'mae': 1.5, 'abs_rel': 0.12, 'a1': 0.85, 'num_pixels': 50000},
        'pedestrian': {'mae': 2.1, 'abs_rel': 0.18, 'a1': 0.78, 'num_pixels': 30000},
        'road': {'mae': 0.8, 'abs_rel': 0.08, 'a1': 0.92, 'num_pixels': 100000},
        'building': {'mae': 3.2, 'abs_rel': 0.22, 'a1': 0.72, 'num_pixels': 80000},
    }

    class_names_dict = {
        0: 'undefined', 1: 'vehicle', 2: 'pedestrian', 3: 'road',
        4: 'building', 5: 'vegetation', 6: 'sky', 7: 'pole'
    }

    # Create visualization
    create_object_wise_grid(
        input_image, gt_depth, pred_depth, seg_mask,
        class_metrics, class_names_dict,
        'test_object_wise_grid.png'
    )

    logger.info("Test visualization complete!")
