import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
import cv2
from PIL import Image
import logging
from .metric_depth_metrics import MetricDepthMetrics
from .sparse_depth_visualization import create_sparse_depth_vis_no_inpaint


class Gear2Visualizer:
    """
    Visualization utilities for Gear2 training (Ablation: No FG/BG separation)
    """

    # Sparse depth datasets that need inpainting for visualization
    SPARSE_DATASETS = ['waymo', 'waymo_seg', 'nuscenes']

    def __init__(self, save_dir="./visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.logger = logging.getLogger(__name__)

        # Set up matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")

    def create_validation_summary(self, sample_batch, model_outputs, step, save_name=None, prefix="validation", fps=None, loss_dict=None, dataset_name=None, config=None):
        """
        Create a comprehensive validation summary for Gear2

        Layout (4 rows × 3 columns) - UNIFIED WITH GEAR3 UPGRADE:
        Row 1: Input, GT Depth, Pred Depth
        Row 2: N/A (No Importance), N/A (No FG Mask), N/A (No BG Mask)
        Row 3: Valid Mask, Error Map, Metrics & Training Info
        Row 4: Depth Distribution (colspan=2), Empty (colspan=1)

        Args:
            sample_batch: Validation batch (images, gt_depth, dataset_idx)
            model_outputs: Dictionary with 'pred_depth' and 'importance_map' (importance_map will be None for Gear2)
            step: Training step number
            save_name: Optional custom save name
            prefix: Prefix for the visualization
            fps: Forward pass FPS (optional)
            loss_dict: Dictionary with loss values (optional)
                - 'depth_loss': Depth loss value
            dataset_name: Name of the dataset (for sparse depth detection)

        Returns:
            fig: Matplotlib figure object
        """
        try:
            images, gt_depth, dataset_idx, focal_lengths = sample_batch
            pred_depth = model_outputs['pred_depth']
            importance_map = model_outputs.get('importance_map', None)  # Will be None for Gear2

            # Debug shapes
            if not hasattr(images, 'cpu'):
                images = torch.from_numpy(images) if isinstance(images, np.ndarray) else images
            if not hasattr(gt_depth, 'cpu'):
                gt_depth = torch.from_numpy(gt_depth) if isinstance(gt_depth, np.ndarray) else gt_depth
            if not hasattr(pred_depth, 'cpu'):
                pred_depth = torch.from_numpy(pred_depth) if isinstance(pred_depth, np.ndarray) else pred_depth

            # Use first batch and first frame for visualization
            # Handle both [B, T, ...] and [B, ...] formats
            if images.ndim == 5:  # [B, T, C, H, W]
                input_img = images[0, 0].float().cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
            else:  # [B, C, H, W]
                input_img = images[0].float().cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]

            # Min-Max normalization (FlashDepth original method)
            # This automatically stretches contrast without manual denormalization
            input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min() + 1e-8)
            input_img = np.clip(input_img, 0, 1)

            if gt_depth.ndim == 4:  # [B, T, H, W] or [B, 1, H, W]
                gt_depth_frame = gt_depth[0, 0].cpu().numpy()  # [H, W]
            else:  # [B, H, W]
                gt_depth_frame = gt_depth[0].cpu().numpy()  # [H, W]

            if pred_depth.ndim == 4:  # [B, T, H, W] or [B, 1, H, W]
                pred_depth_frame = pred_depth[0, 0].cpu().numpy()  # [H, W]
            else:  # [B, H, W]
                pred_depth_frame = pred_depth[0].cpu().numpy()  # [H, W]

            # Ensure all frames are 2D
            while gt_depth_frame.ndim > 2:
                gt_depth_frame = gt_depth_frame[0]
            while pred_depth_frame.ndim > 2:
                pred_depth_frame = pred_depth_frame[0]

            # Create valid masks based on canonical space (70m threshold)
            if 'canonical_gt_valid' in model_outputs:
                canonical_gt_valid = model_outputs['canonical_gt_valid'][0, 0].cpu().numpy()
                canonical_pred_valid = model_outputs['canonical_pred_valid'][0, 0].cpu().numpy()

                # Metrics: GT valid + Pred outlier filtering (Option 1)
                # Filter out unreasonable predictions (NaN, Inf, >200m)
                MAX_DEPTH_OUTLIER = 200.0
                pred_outlier_mask = (pred_depth_frame > 0) & (pred_depth_frame < MAX_DEPTH_OUTLIER)
                valid_mask_metrics = canonical_gt_valid & pred_outlier_mask

                # Visualization: GT uses GT valid
                valid_mask_gt_vis = canonical_gt_valid

                # Check if dataset is sparse (< 50% valid GT pixels)
                gt_exists = (gt_depth_frame > 0)
                gt_density = gt_exists.sum() / gt_exists.size
                is_sparse = gt_density < 0.5

                if is_sparse:
                    # Sparse dataset (waymo_seg): Apply height mask + fill sparse gaps
                    # 1. Find valid scan height range from GT
                    valid_pixels_per_row = gt_exists.sum(axis=1)  # [H]
                    min_valid_pixels_threshold = 10  # At least 10 GT pixels per row
                    valid_rows = valid_pixels_per_row >= min_valid_pixels_threshold
                    valid_row_indices = np.where(valid_rows)[0]

                    if len(valid_row_indices) > 0:
                        min_valid_row = valid_row_indices.min()
                        max_valid_row = valid_row_indices.max()
                        height_mask = np.zeros_like(gt_depth_frame, dtype=bool)
                        height_mask[min_valid_row:max_valid_row+1, :] = True
                    else:
                        height_mask = np.ones_like(gt_depth_frame, dtype=bool)

                    # 2. GT missing mask (sparse LiDAR gaps)
                    gt_missing = ~gt_exists

                    # 3. Final: Within height AND (GT valid OR (GT missing AND Pred valid))
                    valid_mask_pred_vis = height_mask & (canonical_gt_valid | (gt_missing & canonical_pred_valid))
                else:
                    # Dense dataset (sintel): Just use GT valid mask
                    valid_mask_pred_vis = canonical_gt_valid

                # Valid Mask visualization: GT valid only (shows actual evaluation region)
                valid_mask_vis = canonical_gt_valid
            else:
                # Backward compatibility: create masks in actual space
                MAX_DEPTH = 70.0
                MAX_DEPTH_OUTLIER = 200.0
                gt_valid = (gt_depth_frame > 0) & (gt_depth_frame < MAX_DEPTH)
                pred_valid = (pred_depth_frame > 0) & (pred_depth_frame < MAX_DEPTH)
                pred_outlier_mask = (pred_depth_frame > 0) & (pred_depth_frame < MAX_DEPTH_OUTLIER)
                valid_mask_metrics = gt_valid & pred_outlier_mask
                valid_mask_gt_vis = gt_valid

                # Check if dataset is sparse
                gt_exists = (gt_depth_frame > 0)
                gt_density = gt_exists.sum() / gt_exists.size
                is_sparse = gt_density < 0.5

                if is_sparse:
                    # Sparse: height mask + fill sparse gaps
                    valid_pixels_per_row = gt_exists.sum(axis=1)
                    min_valid_pixels_threshold = 10
                    valid_rows = valid_pixels_per_row >= min_valid_pixels_threshold
                    valid_row_indices = np.where(valid_rows)[0]

                    if len(valid_row_indices) > 0:
                        min_valid_row = valid_row_indices.min()
                        max_valid_row = valid_row_indices.max()
                        height_mask = np.zeros_like(gt_depth_frame, dtype=bool)
                        height_mask[min_valid_row:max_valid_row+1, :] = True
                    else:
                        height_mask = np.ones_like(gt_depth_frame, dtype=bool)

                    gt_missing = ~gt_exists
                    valid_mask_pred_vis = height_mask & (gt_valid | (gt_missing & pred_valid))
                else:
                    # Dense: just use GT valid mask
                    valid_mask_pred_vis = gt_valid

                valid_mask_vis = gt_valid

            if valid_mask_metrics.sum() > 0:
                gt_valid = gt_depth_frame[valid_mask_metrics]
                pred_valid = pred_depth_frame[valid_mask_metrics]

            # Create figure with subplots - NEW LAYOUT: 4 rows × 3 columns
            fig = plt.figure(figsize=(15, 16))
            gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)

            # Calculate valid ratio and error BEFORE visualization
            # Use metrics mask for statistics (70m threshold)
            num_valid_metrics = valid_mask_metrics.sum()
            valid_ratio_metrics = num_valid_metrics / valid_mask_metrics.size
            valid_ratio_gt_vis = valid_mask_gt_vis.sum() / valid_mask_gt_vis.size
            valid_ratio_pred_vis = valid_mask_pred_vis.sum() / valid_mask_pred_vis.size
            abs_error = np.abs(pred_depth_frame - gt_depth_frame)
            abs_error_masked = np.where(valid_mask_metrics, abs_error, np.nan)  # Metrics use 70m threshold

            # Check if we have valid pixels for metrics calculation
            has_valid_pixels = num_valid_metrics > 0

            # ==================== Row 1: Input, GT, Pred ====================

            # 1. Input Image
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(input_img)
            ax1.set_title('Input Image', fontsize=14, fontweight='bold')
            ax1.axis('off')

            # 2. Ground Truth Depth (with sparse depth handling)
            ax2 = fig.add_subplot(gs[0, 1])

            # Use inpainting ONLY for sparse datasets (waymo, nuscenes)
            is_sparse_dataset = dataset_name in self.SPARSE_DATASETS if dataset_name else False

            if is_sparse_dataset:
                # Sparse depth: show valid pixels only (no inpainting)
                # Use GT vis mask (canonical_gt_valid with 70m limit)
                _, gt_dense_vis, gt_info = create_sparse_depth_vis_no_inpaint(
                    gt_depth_frame, valid_mask_gt_vis, colormap='plasma_r', percentile_range=(2, 98)
                )
                im2 = ax2.imshow(gt_dense_vis)
                ax2.set_title(f'GT Depth (Sparse)\n{valid_ratio_gt_vis*100:.1f}% valid\n(Metrics: {valid_ratio_metrics*100:.1f}%)',
                             fontsize=12, fontweight='bold')
                vmin, vmax = gt_info['vmin'], gt_info['vmax']
            else:
                # Dense depth: use standard visualization (invalid pixels = black)
                gt_display = np.where(valid_mask_gt_vis, gt_depth_frame, np.nan)  # Invalid = NaN (will be black)
                if valid_mask_gt_vis.sum() > 0:
                    vmin = np.nanpercentile(gt_display, 2)
                    vmax = np.nanpercentile(gt_display, 98)
                else:
                    vmin, vmax = 0, 1
                cmap = plt.cm.plasma_r.copy()
                cmap.set_bad(color='black')  # NaN pixels = black
                im2 = ax2.imshow(gt_display, cmap=cmap, vmin=vmin, vmax=vmax)  # plasma_r: near=bright, far=dark
                ax2.set_title(f'Ground Truth Depth (m)\n(Metrics: {valid_ratio_metrics*100:.1f}%)',
                             fontsize=12, fontweight='bold')

            ax2.axis('off')
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

            # 3. Predicted Metric Depth (with sparse depth handling)
            ax3 = fig.add_subplot(gs[0, 2])

            # Apply valid_mask_pred_vis to pred visualization (both sparse and dense)
            pred_display = np.where(valid_mask_pred_vis, pred_depth_frame, np.nan)  # Invalid = NaN (will be black)
            cmap_pred = plt.cm.plasma_r.copy()
            cmap_pred.set_bad(color='black')  # NaN pixels = black
            im3 = ax3.imshow(pred_display, cmap=cmap_pred, vmin=vmin, vmax=vmax)  # plasma_r: near=bright, far=dark
            mae_str = f'{np.nanmean(abs_error_masked):.3f}m' if has_valid_pixels else 'N/A'

            if is_sparse_dataset:
                ax3.set_title(f'Pred Depth\nMAE: {mae_str}',
                             fontsize=12, fontweight='bold')
            else:
                ax3.set_title(f'Predicted Metric Depth (m)\nMAE: {mae_str}',
                             fontsize=12, fontweight='bold')

            ax3.axis('off')
            plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

            # ==================== Row 2: N/A (No Importance Map for Gear2) ====================

            # 4. Importance Map → N/A for Gear2
            ax4 = fig.add_subplot(gs[1, 0])
            ax4.text(0.5, 0.5, 'No Importance Map\n\nUniform Modulation',
                    ha='center', va='center', transform=ax4.transAxes,
                    fontsize=16, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            ax4.set_title('Importance Map (N/A)', fontsize=14, fontweight='bold')
            ax4.axis('off')

            # 5. FG Mask → N/A for Gear2
            ax5 = fig.add_subplot(gs[1, 1])
            ax5.text(0.5, 0.5, 'No FG/BG Separation\n\n(Gear2: Uniform)',
                    ha='center', va='center', transform=ax5.transAxes,
                    fontsize=16, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            ax5.set_title('FG Mask (N/A)', fontsize=14, fontweight='bold')
            ax5.axis('off')

            # 6. BG Mask → N/A for Gear2
            ax6 = fig.add_subplot(gs[1, 2])
            ax6.text(0.5, 0.5, 'No FG/BG Separation\n\n(Gear2: Uniform)',
                    ha='center', va='center', transform=ax6.transAxes,
                    fontsize=16, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            ax6.set_title('BG Mask (N/A)', fontsize=14, fontweight='bold')
            ax6.axis('off')

            # ==================== Row 3: Valid Mask, Error, Metrics & Training Info ====================

            # 7. Valid Mask (GT valid only, shows actual evaluation region)
            # valid=white, invalid=black
            ax7 = fig.add_subplot(gs[2, 0])
            ax7.imshow(valid_mask_vis.astype(np.uint8), cmap='gray', vmin=0, vmax=1)
            valid_ratio_pct = (valid_mask_vis.sum() / valid_mask_vis.size) * 100
            ax7.set_title(f'Valid Mask ({valid_ratio_pct:.1f}%)\nGT valid only',
                         fontsize=12, fontweight='bold')
            ax7.axis('off')

            # 8. Absolute Error Map
            ax8 = fig.add_subplot(gs[2, 1])
            if has_valid_pixels:
                error_vmax = np.nanpercentile(abs_error_masked, 95)
                im8 = ax8.imshow(abs_error_masked, cmap='hot', vmin=0, vmax=error_vmax)
                ax8.set_title(f'Absolute Error (m)\nMean: {np.nanmean(abs_error_masked):.3f}',
                             fontsize=14, fontweight='bold')
                plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)
            else:
                # No valid pixels - show placeholder
                ax8.text(0.5, 0.5, 'No Valid Pixels\n(All depths > 70m or invalid)',
                        ha='center', va='center', transform=ax8.transAxes,
                        fontsize=12, color='red', fontweight='bold')
                ax8.set_title('Absolute Error (m)\nN/A', fontsize=14, fontweight='bold')
            ax8.axis('off')

            # 9. Depth Metrics & Training Info (COMBINED)
            ax9 = fig.add_subplot(gs[2, 2])

            y_pos = 0.95
            # Step info
            ax9.text(0.05, y_pos, f'Step: {step}', fontsize=11,
                    transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'),
                    fontweight='bold')
            y_pos -= 0.12

            # Dataset info
            if isinstance(dataset_idx, str):
                dataset_str = dataset_idx
            elif isinstance(dataset_idx, (list, tuple)):
                dataset_str = str(dataset_idx[0])
            elif torch.is_tensor(dataset_idx):
                dataset_str = str(dataset_idx[0].item() if dataset_idx.dim() > 0 else dataset_idx.item())
            else:
                dataset_str = str(dataset_idx)
            ax9.text(0.05, y_pos, f'Dataset: {dataset_str}', fontsize=11, transform=ax9.transAxes)
            y_pos -= 0.10

            # Focal length and depth range info (resized)
            if focal_lengths is not None and torch.is_tensor(focal_lengths):
                # Extract first batch, first/middle frame
                fx_value = focal_lengths[0, 0].item() if focal_lengths.ndim >= 2 else focal_lengths[0].item()

                # Show valid GT range (canonical 70m in actual space)
                use_canonical = config.get('use_canonical_space', False) if config is not None else False
                if use_canonical:
                    # Get canonical focal length from config
                    canonical_fx_config = config.get('canonical_focal_length', 1000.0)
                    if hasattr(canonical_fx_config, 'get'):
                        resolution = config['dataset']['resolution']
                        CANONICAL_FX = float(canonical_fx_config.get(resolution, canonical_fx_config.get('base', 500.0)))
                    else:
                        CANONICAL_FX = float(canonical_fx_config)
                    # depth_canonical = depth_actual * (CANONICAL_FX / fx_actual) = 70
                    # Therefore: depth_actual = 70 * (fx_actual / CANONICAL_FX)
                    valid_gt_max = 70.0 * (fx_value / CANONICAL_FX)
                    ax9.text(0.05, y_pos, f'resized_fx: {fx_value:.1f}, valid_gt_max: {valid_gt_max:.3f}m', fontsize=10,
                            transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightyellow'))
                else:
                    # No canonical space
                    ax9.text(0.05, y_pos, f'resized_fx: {fx_value:.1f}, valid_gt_max: 70.000m', fontsize=10,
                            transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightyellow'))
                y_pos -= 0.10

            # FG:BG ratio → N/A for Gear2
            ax9.text(0.05, y_pos, f'FG:BG = N/A', fontsize=10,
                    transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightgray'))
            y_pos -= 0.10

            # FPS if available
            if fps is not None:
                ax9.text(0.05, y_pos, f'FPS: {fps:.1f}', fontsize=10,
                        transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen'))
                y_pos -= 0.10

            # Depth metrics
            if valid_mask_metrics.sum() > 0:
                pred_tensor = torch.from_numpy(pred_depth_frame).float()
                gt_tensor = torch.from_numpy(gt_depth_frame).float()
                valid_tensor = torch.from_numpy(valid_mask_metrics).bool()

                metrics = MetricDepthMetrics.compute_metric_depth_metrics(
                    pred_tensor, gt_tensor, valid_tensor
                )

                ax9.text(0.05, y_pos, f'AbsRel: {metrics["abs_rel"]:.4f}', fontsize=10,
                        transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightcoral'))
                y_pos -= 0.08
                ax9.text(0.05, y_pos, f'Delta_1: {metrics["a1"]:.4f}', fontsize=10,
                        transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen'))
                y_pos -= 0.08
                ax9.text(0.05, y_pos, f'Delta_2: {metrics["a2"]:.4f}', fontsize=10,
                        transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen'))
                y_pos -= 0.08
                ax9.text(0.05, y_pos, f'Delta_3: {metrics["a3"]:.4f}', fontsize=10,
                        transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen'))
                y_pos -= 0.08
                ax9.text(0.05, y_pos, f'RMSE: {metrics["rmse"]:.3f}m', fontsize=9,
                        transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
                y_pos -= 0.08
                ax9.text(0.05, y_pos, f'MAE: {metrics.get("mae", 0):.3f}m', fontsize=9,
                        transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightblue'))
                y_pos -= 0.08

            # Loss values if available
            if loss_dict is not None:
                # Validation loss (for validation visualization)
                if 'val_loss' in loss_dict:
                    ax9.text(0.05, y_pos, f'Val Loss: {loss_dict["val_loss"]:.4f}', fontsize=9,
                            transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightcoral'))
                    y_pos -= 0.08

                # Depth loss (for training visualization)
                if 'depth_loss' in loss_dict:
                    ax9.text(0.05, y_pos, f'Log L1: {loss_dict["depth_loss"]:.4f}', fontsize=9,
                            transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightcoral'))
                    y_pos -= 0.08

            ax9.set_title('Depth Metrics & Training Info', fontsize=14, fontweight='bold')
            ax9.axis('off')

            # ==================== Row 4: Depth Distribution ====================

            # 10. Depth Distribution Histogram (70m threshold for metrics)
            ax10 = fig.add_subplot(gs[3, :2])
            if valid_mask_metrics.sum() > 0:
                gt_valid = gt_depth_frame[valid_mask_metrics]
                pred_valid = pred_depth_frame[valid_mask_metrics]

                bins = np.linspace(min(gt_valid.min(), pred_valid.min()),
                                  max(gt_valid.max(), pred_valid.max()), 50)

                ax10.hist(gt_valid, bins=bins, alpha=0.6, label='Ground Truth',
                        color='blue', density=True)
                ax10.hist(pred_valid, bins=bins, alpha=0.6, label='Predicted',
                        color='red', density=True)
                ax10.set_xlabel('Depth (meters)', fontsize=12)
                ax10.set_ylabel('Density', fontsize=12)
                ax10.set_title('Depth Distribution', fontsize=14, fontweight='bold')
                ax10.legend(fontsize=12)
                ax10.grid(True, alpha=0.3)

            # 11. Empty (reserved for importance distribution in Gear3/3_upgrade)
            ax11 = fig.add_subplot(gs[3, 2])
            ax11.text(0.5, 0.5, 'No Importance Map\n\n(Uniform Modulation)',
                     ha='center', va='center', transform=ax11.transAxes,
                     fontsize=14, fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            ax11.set_title('Importance Distribution (N/A)', fontsize=14, fontweight='bold')
            ax11.axis('off')

            # Save figure
            if save_name:
                save_path = self.save_dir / f"{save_name}.png"
            else:
                save_path = self.save_dir / f"{prefix}_step_{step:06d}.png"

            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved visualization to {save_path}")

            plt.close(fig)
            return fig

        except Exception as e:
            self.logger.error(f"Failed to create validation summary: {e}")
            import traceback
            traceback.print_exc()
            if 'fig' in locals():
                plt.close(fig)
            return None
