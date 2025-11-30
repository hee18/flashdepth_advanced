"""
Foreground-wise depth evaluation utilities.

Evaluates depth estimation accuracy separately for foreground (FG) and background (BG)
regions based on ViT attention-based fg_masks.

Unlike object-wise evaluation (which uses dataset-provided segmentation),
fg_masks are generated from ViT attention weights and represent regions
where the model focuses attention.

This allows analyzing the correlation between model attention and depth accuracy
across all datasets, not just those with segmentation labels.
"""

import os
import cv2
import numpy as np
import torch
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# fg_mask path patterns for each dataset
# These match the output of scripts/generate_fg_masks.py
FG_MASK_PATTERNS = {
    'eth3d': '{data_root}/eth3d/{scene}/fg_masks/{frame}.png',
    'sintel': '{data_root}/sintel/fg_masks/training/clean/{scene}/{frame}.png',
    'waymo_seg': '{data_root}/waymo_seg/val/{segment}/FRONT/fg_masks/{frame}.png',
    'vkitti': '{data_root}/vkitti/{scene}/clone/frames/fg_masks/Camera_0/fg_{frame}.png',
    'unreal4k': '{data_root}/unreal4k/UnrealStereo4K_{scene}/fg_masks/{frame}.png',
    'urbansyn': '{data_root}/urbansyn/{scene}/fg_masks/{frame}.png',
    'tartanair': '{data_root}/tartanair/{scene}/fg_masks/{frame}.png',
    'bonn': '{data_root}/bonn/{scene}/fg_masks/{frame}.png',
}


class FGWiseMetrics:
    """Compute depth metrics separately for foreground and background regions."""

    def __init__(self, data_root: str, dataset_name: str):
        """
        Initialize FG-wise metrics calculator.

        Args:
            data_root: Root directory of datasets
            dataset_name: Name of the dataset (eth3d, sintel, waymo_seg, etc.)
                          Can include scene path like 'waymo_seg/segment-xxx'
        """
        self.data_root = data_root
        self.full_dataset_name = dataset_name.lower()
        # Extract base dataset name (e.g., 'waymo_seg/segment-xxx' -> 'waymo_seg')
        self.dataset_name = self.full_dataset_name.split('/')[0]
        self.fg_mask_cache = {}  # Cache loaded masks

        if self.dataset_name not in FG_MASK_PATTERNS:
            logger.warning(f"No fg_mask pattern defined for dataset: {self.dataset_name}")
            self.pattern = None
        else:
            self.pattern = FG_MASK_PATTERNS[self.dataset_name]

    def get_fg_mask_path(self, scene: str, frame: str) -> str:
        """
        Get the path to fg_mask file for a specific frame.

        Args:
            scene: Scene/sequence name
            frame: Frame filename (without extension for some datasets)

        Returns:
            Full path to fg_mask PNG file
        """
        if self.pattern is None:
            return None

        return self.pattern.format(
            data_root=self.data_root,
            scene=scene,
            segment=scene,  # For waymo_seg
            frame=frame
        )

    def load_fg_mask(self, scene: str, frame: str, target_shape: Tuple[int, int] = None) -> Optional[np.ndarray]:
        """
        Load fg_mask for a specific frame.

        Args:
            scene: Scene/sequence name
            frame: Frame filename
            target_shape: (H, W) to resize mask to if needed

        Returns:
            Binary mask array (H, W) with values 0 (BG) or 255 (FG), or None if not found
        """
        path = self.get_fg_mask_path(scene, frame)
        if path is None or not os.path.exists(path):
            return None

        # Check cache
        cache_key = (scene, frame)
        if cache_key in self.fg_mask_cache:
            mask = self.fg_mask_cache[cache_key]
        else:
            # Load grayscale mask
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                logger.warning(f"Failed to load fg_mask: {path}")
                return None
            self.fg_mask_cache[cache_key] = mask

        # Resize if needed
        if target_shape is not None and mask.shape != target_shape:
            mask = cv2.resize(mask, (target_shape[1], target_shape[0]),
                             interpolation=cv2.INTER_NEAREST)

        return mask

    def clear_cache(self):
        """Clear the fg_mask cache to free memory."""
        self.fg_mask_cache.clear()

    @staticmethod
    def compute_metrics_with_mask(
        pred_depth: np.ndarray,
        gt_depth: np.ndarray,
        region_mask: np.ndarray,
        valid_mask: np.ndarray,
        min_pixels: int = 100
    ) -> Dict[str, float]:
        """
        Compute depth metrics for a specific region.

        Args:
            pred_depth: Predicted depth [H, W]
            gt_depth: Ground truth depth [H, W]
            region_mask: Binary mask for region of interest [H, W] (>0 means included)
            valid_mask: Valid depth mask [H, W]
            min_pixels: Minimum valid pixels required to compute metrics

        Returns:
            Dictionary with metrics (mae, rmse, abs_rel, a1, a2, a3, num_pixels)
        """
        # Combine masks
        combined_mask = (region_mask > 0) & valid_mask

        num_valid = combined_mask.sum()
        if num_valid < min_pixels:
            return {
                'mae': float('nan'),
                'rmse': float('nan'),
                'abs_rel': float('nan'),
                'a1': float('nan'),
                'a2': float('nan'),
                'a3': float('nan'),
                'num_pixels': int(num_valid)
            }

        pred_valid = pred_depth[combined_mask]
        gt_valid = gt_depth[combined_mask]

        # Ensure positive values for ratio computation
        pred_valid = np.maximum(pred_valid, 1e-8)
        gt_valid = np.maximum(gt_valid, 1e-8)

        # Basic metrics
        abs_diff = np.abs(pred_valid - gt_valid)
        mae = np.mean(abs_diff)
        rmse = np.sqrt(np.mean(abs_diff ** 2))
        abs_rel = np.mean(abs_diff / gt_valid)

        # Threshold accuracies (δ1, δ2, δ3)
        thresh = np.maximum(gt_valid / pred_valid, pred_valid / gt_valid)
        a1 = np.mean(thresh < 1.25)
        a2 = np.mean(thresh < 1.25 ** 2)
        a3 = np.mean(thresh < 1.25 ** 3)

        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'abs_rel': float(abs_rel),
            'a1': float(a1),
            'a2': float(a2),
            'a3': float(a3),
            'num_pixels': int(num_valid)
        }

    def compute_fgbg_metrics(
        self,
        pred_depth: np.ndarray,
        gt_depth: np.ndarray,
        fg_mask: np.ndarray,
        valid_mask: np.ndarray = None,
        min_pixels: int = 100
    ) -> Dict[str, float]:
        """
        Compute metrics separately for FG and BG regions.

        Args:
            pred_depth: Predicted depth [H, W]
            gt_depth: Ground truth depth [H, W]
            fg_mask: Foreground mask [H, W], values > 0 are foreground
            valid_mask: Valid depth mask [H, W] (default: gt > 0)
            min_pixels: Minimum valid pixels per region

        Returns:
            Dictionary with fg_* and bg_* metrics
        """
        if valid_mask is None:
            valid_mask = gt_depth > 0

        # Ensure numpy arrays
        if isinstance(pred_depth, torch.Tensor):
            pred_depth = pred_depth.cpu().numpy()
        if isinstance(gt_depth, torch.Tensor):
            gt_depth = gt_depth.cpu().numpy()
        if isinstance(fg_mask, torch.Tensor):
            fg_mask = fg_mask.cpu().numpy()
        if isinstance(valid_mask, torch.Tensor):
            valid_mask = valid_mask.cpu().numpy()

        # Create FG and BG masks
        fg_region = fg_mask > 0
        bg_region = ~fg_region

        # Compute metrics for each region
        fg_metrics = self.compute_metrics_with_mask(
            pred_depth, gt_depth, fg_region, valid_mask, min_pixels
        )
        bg_metrics = self.compute_metrics_with_mask(
            pred_depth, gt_depth, bg_region, valid_mask, min_pixels
        )

        # Prefix with fg_ and bg_
        result = {}
        for key, value in fg_metrics.items():
            result[f'fg_{key}'] = value
        for key, value in bg_metrics.items():
            result[f'bg_{key}'] = value

        # Add ratio metrics (FG/BG performance gap)
        if not np.isnan(fg_metrics['abs_rel']) and not np.isnan(bg_metrics['abs_rel']):
            result['fg_bg_absrel_ratio'] = fg_metrics['abs_rel'] / max(bg_metrics['abs_rel'], 1e-8)

        return result

    def compute_frame_metrics(
        self,
        pred_depth: np.ndarray,
        gt_depth: np.ndarray,
        scene: str,
        frame: str,
        valid_mask: np.ndarray = None,
        min_pixels: int = 100
    ) -> Optional[Dict[str, float]]:
        """
        Compute FG/BG metrics for a single frame.

        Args:
            pred_depth: Predicted depth [H, W]
            gt_depth: Ground truth depth [H, W]
            scene: Scene name
            frame: Frame filename
            valid_mask: Valid depth mask
            min_pixels: Minimum valid pixels per region

        Returns:
            Dictionary with metrics, or None if fg_mask not found
        """
        # Load fg_mask with matching shape
        target_shape = pred_depth.shape if isinstance(pred_depth, np.ndarray) else pred_depth.shape
        fg_mask = self.load_fg_mask(scene, frame, target_shape)

        if fg_mask is None:
            return None

        return self.compute_fgbg_metrics(
            pred_depth, gt_depth, fg_mask, valid_mask, min_pixels
        )


def aggregate_fgwise_metrics(metrics_list: list) -> Dict[str, float]:
    """
    Aggregate FG-wise metrics across multiple frames/sequences.

    Args:
        metrics_list: List of dictionaries from compute_fgbg_metrics()

    Returns:
        Dictionary with averaged metrics
    """
    if not metrics_list:
        return {}

    # Filter out None values
    metrics_list = [m for m in metrics_list if m is not None]
    if not metrics_list:
        return {}

    # Get all keys
    keys = metrics_list[0].keys()

    result = {}
    for key in keys:
        values = [m[key] for m in metrics_list if not np.isnan(m.get(key, float('nan')))]
        if values:
            if 'num_pixels' in key:
                result[key] = int(np.sum(values))  # Sum for pixel counts
            else:
                result[key] = float(np.mean(values))  # Mean for metrics
        else:
            result[key] = float('nan')

    return result


# ============================================================================
# FG Visualization Functions
# ============================================================================

def draw_fg_contours(
    image: np.ndarray,
    fg_mask: np.ndarray,
    contour_color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw FG contours on an image.

    Args:
        image: Input image [H, W, 3] in RGB or [H, W] grayscale, uint8 or float
        fg_mask: FG mask [H, W], values > 0 are foreground
        contour_color: BGR color for contours (default: red)
        thickness: Contour line thickness

    Returns:
        Image with FG contours drawn, [H, W, 3] uint8
    """
    # Convert image to uint8 BGR for cv2
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

    if len(image.shape) == 2:
        # Grayscale to BGR
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 3:
        # Assume RGB, convert to BGR for cv2
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Make a copy to draw on
    result = image.copy()

    # Ensure fg_mask is uint8
    if fg_mask.dtype != np.uint8:
        fg_mask = ((fg_mask > 0) * 255).astype(np.uint8)

    # Resize mask if needed
    if fg_mask.shape[:2] != image.shape[:2]:
        fg_mask = cv2.resize(fg_mask, (image.shape[1], image.shape[0]),
                             interpolation=cv2.INTER_NEAREST)

    # Find contours
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours
    cv2.drawContours(result, contours, -1, contour_color, thickness)

    # Convert back to RGB
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    return result


def create_depth_with_fg_overlay(
    depth: np.ndarray,
    fg_mask: np.ndarray,
    vmin: float = None,
    vmax: float = None,
    cmap: str = 'plasma_r',
    contour_color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Create depth colormap with FG contours overlaid.

    Args:
        depth: Depth map [H, W]
        fg_mask: FG mask [H, W]
        vmin, vmax: Depth range for colormap normalization
        cmap: Matplotlib colormap name
        contour_color: BGR color for contours
        thickness: Contour line thickness

    Returns:
        Depth visualization with FG contours, [H, W, 3] uint8
    """
    import matplotlib.pyplot as plt

    # Get colormap
    colormap = plt.get_cmap(cmap)

    # Determine value range
    valid_mask = np.isfinite(depth) & (depth > 0)
    if valid_mask.sum() > 0:
        if vmin is None:
            vmin = np.percentile(depth[valid_mask], 2)
        if vmax is None:
            vmax = np.percentile(depth[valid_mask], 98)
    else:
        vmin = vmin or 0
        vmax = vmax or 1

    # Normalize depth
    depth_normalized = np.clip((depth - vmin) / (vmax - vmin + 1e-8), 0, 1)

    # Apply colormap
    depth_colored = colormap(depth_normalized)[:, :, :3]  # Remove alpha
    depth_colored = (depth_colored * 255).astype(np.uint8)

    # Draw FG contours
    result = draw_fg_contours(depth_colored, fg_mask, contour_color, thickness)

    return result


def create_fgwise_comparison_figure(
    image: np.ndarray,
    gt_depth: np.ndarray,
    pred_depth: np.ndarray,
    fg_mask: np.ndarray,
    fg_metrics: Dict[str, float] = None,
    title: str = "FG-wise Depth Evaluation"
) -> 'matplotlib.figure.Figure':
    """
    Create a comprehensive FG-wise comparison figure.

    Layout:
        Row 1: Input Image | GT Depth (FG contours) | Pred Depth (FG contours)
        Row 2: FG Mask Overlay | FG/BG Error Comparison | FG/BG Metrics

    Args:
        image: Input image [H, W, 3]
        gt_depth: Ground truth depth [H, W]
        pred_depth: Predicted depth [H, W]
        fg_mask: FG mask [H, W]
        fg_metrics: Optional dict with fg_* and bg_* metrics
        title: Figure title

    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    # Convert tensors to numpy
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
        if image.shape[0] == 3:  # [3, H, W] -> [H, W, 3]
            image = np.transpose(image, (1, 2, 0))
    if isinstance(gt_depth, torch.Tensor):
        gt_depth = gt_depth.cpu().numpy()
    if isinstance(pred_depth, torch.Tensor):
        pred_depth = pred_depth.cpu().numpy()
    if isinstance(fg_mask, torch.Tensor):
        fg_mask = fg_mask.cpu().numpy()

    # Squeeze if needed
    if len(gt_depth.shape) == 3:
        gt_depth = gt_depth.squeeze()
    if len(pred_depth.shape) == 3:
        pred_depth = pred_depth.squeeze()

    # Normalize image to [0, 1]
    if image.max() > 1:
        image = image / 255.0
    image = np.clip(image, 0, 1)

    # Create figure
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.2)

    # Determine depth range from GT
    valid_mask = (gt_depth > 0) & np.isfinite(gt_depth)
    if valid_mask.sum() > 0:
        vmin = np.percentile(gt_depth[valid_mask], 2)
        vmax = np.percentile(gt_depth[valid_mask], 98)
    else:
        vmin, vmax = 0, 70

    # Row 1: Input, GT with FG contours, Pred with FG contours
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image)
    ax1.set_title('Input Image', fontsize=14, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    gt_with_contours = create_depth_with_fg_overlay(gt_depth, fg_mask, vmin, vmax)
    ax2.imshow(gt_with_contours)
    ax2.set_title('GT Depth + FG Contours', fontsize=14, fontweight='bold')
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    pred_with_contours = create_depth_with_fg_overlay(pred_depth, fg_mask, vmin, vmax)
    ax3.imshow(pred_with_contours)
    ax3.set_title('Pred Depth + FG Contours', fontsize=14, fontweight='bold')
    ax3.axis('off')

    # Row 2: FG Mask overlay, Error map, Metrics
    ax4 = fig.add_subplot(gs[1, 0])
    # Create FG overlay (red for FG, blue for BG)
    fg_overlay = np.zeros((*image.shape[:2], 3), dtype=np.float32)
    fg_region = fg_mask > 0
    fg_overlay[fg_region, 0] = 0.5  # Red for FG
    fg_overlay[~fg_region, 2] = 0.3  # Blue for BG
    ax4.imshow(image)
    ax4.imshow(fg_overlay, alpha=0.4)
    fg_ratio = fg_region.sum() / fg_region.size * 100
    ax4.set_title(f'FG Mask Overlay\nFG: {fg_ratio:.1f}% (red) | BG: {100-fg_ratio:.1f}% (blue)',
                  fontsize=12, fontweight='bold')
    ax4.axis('off')

    ax5 = fig.add_subplot(gs[1, 1])
    # Error map with FG/BG distinction
    error = np.abs(pred_depth - gt_depth)
    error_masked = np.where(valid_mask, error, np.nan)
    cmap = plt.cm.hot.copy()
    cmap.set_bad(color='black')
    im5 = ax5.imshow(error_masked, cmap=cmap, vmin=0, vmax=np.nanpercentile(error_masked, 95))
    ax5.set_title('Absolute Error', fontsize=14, fontweight='bold')
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    ax6.set_title('FG/BG Metrics', fontsize=14, fontweight='bold')

    if fg_metrics is not None:
        y_pos = 0.9
        # FG metrics
        ax6.text(0.1, y_pos, 'Foreground (FG):', fontsize=12, fontweight='bold',
                 transform=ax6.transAxes, color='red')
        y_pos -= 0.1
        ax6.text(0.1, y_pos, f"  AbsRel: {fg_metrics.get('fg_abs_rel', float('nan')):.4f}",
                 fontsize=11, transform=ax6.transAxes)
        y_pos -= 0.08
        ax6.text(0.1, y_pos, f"  δ1: {fg_metrics.get('fg_a1', float('nan')):.4f}",
                 fontsize=11, transform=ax6.transAxes)
        y_pos -= 0.08
        ax6.text(0.1, y_pos, f"  Pixels: {fg_metrics.get('fg_num_pixels', 0):,}",
                 fontsize=11, transform=ax6.transAxes)
        y_pos -= 0.15

        # BG metrics
        ax6.text(0.1, y_pos, 'Background (BG):', fontsize=12, fontweight='bold',
                 transform=ax6.transAxes, color='blue')
        y_pos -= 0.1
        ax6.text(0.1, y_pos, f"  AbsRel: {fg_metrics.get('bg_abs_rel', float('nan')):.4f}",
                 fontsize=11, transform=ax6.transAxes)
        y_pos -= 0.08
        ax6.text(0.1, y_pos, f"  δ1: {fg_metrics.get('bg_a1', float('nan')):.4f}",
                 fontsize=11, transform=ax6.transAxes)
        y_pos -= 0.08
        ax6.text(0.1, y_pos, f"  Pixels: {fg_metrics.get('bg_num_pixels', 0):,}",
                 fontsize=11, transform=ax6.transAxes)
        y_pos -= 0.15

        # FG/BG ratio
        if 'fg_bg_absrel_ratio' in fg_metrics:
            ratio = fg_metrics['fg_bg_absrel_ratio']
            color = 'green' if ratio < 1.0 else 'orange'
            ax6.text(0.1, y_pos, f'FG/BG AbsRel Ratio: {ratio:.3f}',
                     fontsize=12, fontweight='bold', transform=ax6.transAxes, color=color)

    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    return fig


def save_fgwise_visualization(
    save_path: str,
    image: np.ndarray,
    gt_depth: np.ndarray,
    pred_depth: np.ndarray,
    fg_mask: np.ndarray,
    fg_metrics: Dict[str, float] = None,
    title: str = "FG-wise Depth Evaluation",
    dpi: int = 150
):
    """
    Save FG-wise comparison visualization to file.

    Args:
        save_path: Output file path
        image: Input image [H, W, 3]
        gt_depth: Ground truth depth [H, W]
        pred_depth: Predicted depth [H, W]
        fg_mask: FG mask [H, W]
        fg_metrics: Optional dict with fg_* and bg_* metrics
        title: Figure title
        dpi: Output DPI
    """
    import matplotlib.pyplot as plt

    fig = create_fgwise_comparison_figure(
        image, gt_depth, pred_depth, fg_mask, fg_metrics, title
    )
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved FG-wise visualization to {save_path}")
