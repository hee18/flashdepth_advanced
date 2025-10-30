import torch
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
from .sparse_depth_visualization import create_dual_sparse_depth_vis


class Gear3Visualizer:
    """
    Visualization utilities for Gear3 training
    """

    # Sparse depth datasets that need inpainting for visualization
    SPARSE_DATASETS = ['waymo', 'nuscenes']

    def __init__(self, save_dir="./visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.logger = logging.getLogger(__name__)

        # Set up matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")

    def create_validation_summary(self, sample_batch, model_outputs, step, save_name=None, prefix="validation", fps=None, loss_dict=None, dataset_name=None):
        """
        Create a comprehensive validation summary for Gear3

        Args:
            sample_batch: Validation batch (images, gt_depth, dataset_idx)
            model_outputs: Dictionary with 'pred_depth' and 'importance_map'
            step: Training step number
            save_name: Optional custom save name
            prefix: Prefix for the visualization
            fps: Forward pass FPS (optional)
            loss_dict: Dictionary with loss values (optional)
                - 'depth_loss': Depth loss value
                - 'depth_variance_loss': Variance loss value (if enabled)
                - 'edge_aware_loss': Edge-aware loss value (if enabled)
                - 'contrastive_fgbg_loss': Contrastive FG/BG loss value (if enabled)
            dataset_name: Name of the dataset (for sparse depth detection)

        Returns:
            fig: Matplotlib figure object
        """
        try:
            images, gt_depth, dataset_idx = sample_batch
            pred_depth = model_outputs['pred_depth']
            importance_map = model_outputs['importance_map']

            # Debug shapes
            if not hasattr(images, 'cpu'):
                images = torch.from_numpy(images) if isinstance(images, np.ndarray) else images
            if not hasattr(gt_depth, 'cpu'):
                gt_depth = torch.from_numpy(gt_depth) if isinstance(gt_depth, np.ndarray) else gt_depth
            if not hasattr(pred_depth, 'cpu'):
                pred_depth = torch.from_numpy(pred_depth) if isinstance(pred_depth, np.ndarray) else pred_depth
            if not hasattr(importance_map, 'cpu'):
                importance_map = torch.from_numpy(importance_map) if isinstance(importance_map, np.ndarray) else importance_map

            # Use first batch and first frame for visualization
            # Handle both [B, T, ...] and [B, ...] formats
            if images.ndim == 5:  # [B, T, C, H, W]
                input_img = images[0, 0].cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
            else:  # [B, C, H, W]
                input_img = images[0].cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]

            if gt_depth.ndim == 4:  # [B, T, H, W] or [B, 1, H, W]
                gt_depth_frame = gt_depth[0, 0].cpu().numpy()  # [H, W]
            else:  # [B, H, W]
                gt_depth_frame = gt_depth[0].cpu().numpy()  # [H, W]

            if pred_depth.ndim == 4:  # [B, T, H, W] or [B, 1, H, W]
                pred_depth_frame = pred_depth[0, 0].cpu().numpy()  # [H, W]
            else:  # [B, H, W]
                pred_depth_frame = pred_depth[0].cpu().numpy()  # [H, W]

            if importance_map.ndim == 4:  # [B, T, H, W] or [B, 1, H, W]
                importance_frame = importance_map[0, 0].cpu().numpy()  # [H, W]
            else:  # [B, H, W]
                importance_frame = importance_map[0].cpu().numpy()  # [H, W]

            # Normalize input image for display
            input_img = np.clip((input_img + 1) / 2, 0, 1)  # Assuming normalized input

            # Debug: Check final shapes
            print(f"DEBUG Visualization shapes:")
            print(f"  input_img: {input_img.shape}")
            print(f"  gt_depth_frame: {gt_depth_frame.shape}")
            print(f"  pred_depth_frame: {pred_depth_frame.shape}")
            print(f"  importance_frame: {importance_frame.shape}")

            # Ensure all frames are 2D
            while gt_depth_frame.ndim > 2:
                gt_depth_frame = gt_depth_frame[0]
            while pred_depth_frame.ndim > 2:
                pred_depth_frame = pred_depth_frame[0]
            while importance_frame.ndim > 2:
                importance_frame = importance_frame[0]

            print(f"DEBUG After squeeze:")
            print(f"  gt_depth_frame: {gt_depth_frame.shape}")
            print(f"  pred_depth_frame: {pred_depth_frame.shape}")
            print(f"  importance_frame: {importance_frame.shape}")

            # Create two separate masks:
            # 1. Metrics mask: 70m threshold (same as training loss)
            # 2. Visualization mask: No upper limit (show all depths)
            MAX_DEPTH_METRICS = 70.0  # For metrics calculation (100/70 = 1.43 inverse depth threshold)

            # Metrics mask: 70m threshold for fair comparison with training
            gt_valid_metrics = (gt_depth_frame > 0) & (gt_depth_frame < MAX_DEPTH_METRICS)
            pred_valid_metrics = (pred_depth_frame > 0) & (pred_depth_frame < MAX_DEPTH_METRICS)
            valid_mask_metrics = gt_valid_metrics & pred_valid_metrics

            # Visualization mask: Filter out invalid values (<=0) and extreme outliers (>1000m)
            MAX_DEPTH_VIS = 1000.0  # Same as TartanAir's maximum valid depth
            gt_valid_vis = (gt_depth_frame > 0) & (gt_depth_frame < MAX_DEPTH_VIS)
            pred_valid_vis = (pred_depth_frame > 0) & (pred_depth_frame < MAX_DEPTH_VIS)
            valid_mask_vis = gt_valid_vis & pred_valid_vis

            # Debug: Print statistics
            print(f"DEBUG Step {step}:")
            print(f"  GT raw range: {gt_depth_frame.min():.3f} - {gt_depth_frame.max():.3f}")
            print(f"  Pred raw range: {pred_depth_frame.min():.3f} - {pred_depth_frame.max():.3f}")
            print(f"  Invalid GT pixels: {((gt_depth_frame <= 0) | (gt_depth_frame >= MAX_DEPTH_METRICS)).sum()}")
            print(f"  Valid for metrics: {valid_mask_metrics.sum()} / {valid_mask_metrics.size}")

            if valid_mask_metrics.sum() > 0:
                gt_valid = gt_depth_frame[valid_mask_metrics]
                pred_valid = pred_depth_frame[valid_mask_metrics]
                print(f"  GT valid range: {gt_valid.min():.3f} - {gt_valid.max():.3f}")
                print(f"  Pred valid range: {pred_valid.min():.3f} - {pred_valid.max():.3f}")
                print(f"  Valid pixels: {valid_mask_metrics.sum()} / {gt_depth_frame.size}")

            # Create figure with subplots
            fig = plt.figure(figsize=(20, 12))
            gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

            # Calculate valid ratio and error BEFORE visualization
            num_valid_metrics = valid_mask_metrics.sum()
            valid_ratio_metrics = num_valid_metrics / valid_mask_metrics.size
            valid_ratio_vis = valid_mask_vis.sum() / valid_mask_vis.size
            abs_error = np.abs(pred_depth_frame - gt_depth_frame)
            abs_error_masked = np.where(valid_mask_metrics, abs_error, np.nan)  # Metrics use 70m threshold

            # Check if we have valid pixels for metrics calculation
            has_valid_pixels = num_valid_metrics > 0

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
                # Sparse depth: use dual visualization (inpainted)
                _, gt_dense_vis, gt_info = create_dual_sparse_depth_vis(
                    gt_depth_frame, valid_mask_vis, colormap='plasma', percentile_range=(2, 98)
                )
                im2 = ax2.imshow(gt_dense_vis)
                ax2.set_title(f'GT Depth (Inpainted)\n{valid_ratio_vis*100:.1f}% valid',
                             fontsize=14, fontweight='bold')
                vmin, vmax = gt_info['vmin'], gt_info['vmax']
            else:
                # Dense depth: use standard visualization
                gt_display = np.where(valid_mask_vis, gt_depth_frame, np.nan)
                vmin, vmax = np.nanpercentile(gt_display, [2, 98])
                im2 = ax2.imshow(gt_display, cmap='plasma', vmin=vmin, vmax=vmax)
                ax2.set_title('Ground Truth Depth (m)', fontsize=14, fontweight='bold')

            ax2.axis('off')
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

            # 3. Predicted Metric Depth (with sparse depth handling)
            ax3 = fig.add_subplot(gs[0, 2])

            if is_sparse_dataset:
                # Sparse depth: use dual visualization (inpainted)
                _, pred_dense_vis, pred_info = create_dual_sparse_depth_vis(
                    pred_depth_frame, valid_mask_vis, colormap='plasma', percentile_range=(2, 98)
                )
                im3 = ax3.imshow(pred_dense_vis)
                mae_str = f'{np.nanmean(abs_error_masked):.2f}m' if has_valid_pixels else 'N/A'
                ax3.set_title(f'Pred Depth (Inpainted)\nMAE: {mae_str}',
                             fontsize=14, fontweight='bold')
            else:
                # Dense depth: use standard visualization
                pred_display = np.where(valid_mask_vis, pred_depth_frame, np.nan)
                im3 = ax3.imshow(pred_display, cmap='plasma', vmin=vmin, vmax=vmax)
                ax3.set_title('Predicted Metric Depth (m)', fontsize=14, fontweight='bold')

            ax3.axis('off')
            plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

            # 4. Importance Map (NEW!)
            ax4 = fig.add_subplot(gs[0, 3])
            im4 = ax4.imshow(importance_frame, cmap='jet', vmin=0, vmax=1)
            imp_mean = importance_frame.mean()
            imp_std = importance_frame.std()
            ax4.set_title(f'Importance Map\nmean={imp_mean:.3f}, std={imp_std:.3f}',
                         fontsize=14, fontweight='bold')
            ax4.axis('off')
            plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

            # 5. Valid Mask
            ax5 = fig.add_subplot(gs[1, 0])
            ax5.imshow(valid_mask_metrics.astype(np.uint8), cmap='gray_r', vmin=0, vmax=1)
            ax5.set_title(f'Valid Mask\n({valid_mask_metrics.sum():,} pixels)', fontsize=14, fontweight='bold')
            ax5.axis('off')

            # 6. Absolute Error Map
            ax6 = fig.add_subplot(gs[1, 1])
            if has_valid_pixels:
                error_vmax = np.nanpercentile(abs_error_masked, 95)
                im6 = ax6.imshow(abs_error_masked, cmap='hot', vmin=0, vmax=error_vmax)
                ax6.set_title(f'Absolute Error (m)\nMean: {np.nanmean(abs_error_masked):.3f}',
                             fontsize=14, fontweight='bold')
                plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
            else:
                # No valid pixels - show placeholder
                ax6.text(0.5, 0.5, 'No Valid Pixels\n(All depths > 70m or invalid)',
                        ha='center', va='center', transform=ax6.transAxes,
                        fontsize=12, color='red', fontweight='bold')
                ax6.set_title('Absolute Error (m)\nN/A', fontsize=14, fontweight='bold')
            ax6.axis('off')

            # 7. Metric Evaluation
            ax7 = fig.add_subplot(gs[1, 2])
            if valid_mask_metrics.sum() > 0:
                pred_tensor = torch.from_numpy(pred_depth_frame).float()
                gt_tensor = torch.from_numpy(gt_depth_frame).float()
                valid_tensor = torch.from_numpy(valid_mask_metrics).bool()

                metrics = MetricDepthMetrics.compute_metric_depth_metrics(
                    pred_tensor, gt_tensor, valid_tensor
                )

                ax7.text(0.1, 0.85, f'AbsRel: {metrics["abs_rel"]:.4f}', fontsize=14,
                        transform=ax7.transAxes, bbox=dict(boxstyle="round", facecolor='lightcoral'))
                ax7.text(0.1, 0.7, f'Delta_1: {metrics["a1"]:.4f}', fontsize=14,
                        transform=ax7.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen'))
                ax7.text(0.1, 0.55, f'Delta_2: {metrics["a2"]:.4f}', fontsize=14,
                        transform=ax7.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen'))
                ax7.text(0.1, 0.4, f'Delta_3: {metrics["a3"]:.4f}', fontsize=14,
                        transform=ax7.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen'))
                ax7.text(0.1, 0.25, f'RMSE: {metrics["rmse"]:.3f}m', fontsize=12,
                        transform=ax7.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
                ax7.text(0.1, 0.1, f'MAE: {metrics.get("mae", 0):.3f}m', fontsize=12,
                        transform=ax7.transAxes, bbox=dict(boxstyle="round", facecolor='lightblue'))
            ax7.set_title('Depth Metrics', fontsize=14, fontweight='bold')
            ax7.axis('off')

            # 8. Training Info
            ax8 = fig.add_subplot(gs[1, 3])
            y_pos = 0.9  # Start from top

            ax8.text(0.1, y_pos, f'Step: {step}', fontsize=16,
                    transform=ax8.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
            y_pos -= 0.15

            # Handle dataset_idx (can be string or tensor)
            if isinstance(dataset_idx, str):
                dataset_str = dataset_idx
            elif isinstance(dataset_idx, (list, tuple)):
                dataset_str = str(dataset_idx[0])
            elif torch.is_tensor(dataset_idx):
                dataset_str = str(dataset_idx[0].item() if dataset_idx.dim() > 0 else dataset_idx.item())
            else:
                dataset_str = str(dataset_idx)

            ax8.text(0.1, y_pos, f'Dataset: {dataset_str}', fontsize=14, transform=ax8.transAxes)
            y_pos -= 0.12

            # Show FG:BG ratio based on importance map mean
            fg_ratio = (importance_frame >= imp_mean).sum() / importance_frame.size * 100
            bg_ratio = 100.0 - fg_ratio
            ax8.text(0.1, y_pos, f'FG:BG = {fg_ratio:.1f}:{bg_ratio:.1f}', fontsize=12,
                    transform=ax8.transAxes, bbox=dict(boxstyle="round", facecolor='lightcyan'))
            y_pos -= 0.12

            # Show FPS if available
            if fps is not None:
                ax8.text(0.1, y_pos, f'FPS: {fps:.1f}', fontsize=12,
                        transform=ax8.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen'))
                y_pos -= 0.12

            # Show loss values if available
            if loss_dict is not None:
                # Validation loss (for validation visualization)
                if 'val_loss' in loss_dict:
                    ax8.text(0.1, y_pos, f'Val Loss: {loss_dict["val_loss"]:.4f}', fontsize=11,
                            transform=ax8.transAxes, bbox=dict(boxstyle="round", facecolor='lightcoral'))
                    y_pos -= 0.10

                # Depth loss (for training visualization)
                if 'depth_loss' in loss_dict:
                    ax8.text(0.1, y_pos, f'Log L1 Loss: {loss_dict["depth_loss"]:.4f}', fontsize=11,
                            transform=ax8.transAxes, bbox=dict(boxstyle="round", facecolor='lightcoral'))
                    y_pos -= 0.10

                # Variance loss (if enabled)
                if 'depth_variance_loss' in loss_dict and loss_dict['depth_variance_loss'] > 0:
                    ax8.text(0.1, y_pos, f'Variance: {loss_dict["depth_variance_loss"]:.4f}', fontsize=11,
                            transform=ax8.transAxes, bbox=dict(boxstyle="round", facecolor='lightyellow'))
                    y_pos -= 0.10

                # Edge-aware loss (if enabled)
                if 'edge_aware_loss' in loss_dict and loss_dict['edge_aware_loss'] > 0:
                    ax8.text(0.1, y_pos, f'Edge: {loss_dict["edge_aware_loss"]:.4f}', fontsize=11,
                            transform=ax8.transAxes, bbox=dict(boxstyle="round", facecolor='lightblue'))
                    y_pos -= 0.10

                # Contrastive FG/BG loss (if enabled)
                if 'contrastive_fgbg_loss' in loss_dict and loss_dict['contrastive_fgbg_loss'] > 0:
                    ax8.text(0.1, y_pos, f'Contrast: {loss_dict["contrastive_fgbg_loss"]:.4f}', fontsize=11,
                            transform=ax8.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen'))
                    y_pos -= 0.10

            ax8.set_title('Training Info', fontsize=14, fontweight='bold')
            ax8.axis('off')

            # 9. Depth Distribution Histogram
            ax9 = fig.add_subplot(gs[2, :2])
            if valid_mask_metrics.sum() > 0:
                gt_valid = gt_depth_frame[valid_mask_metrics]
                pred_valid = pred_depth_frame[valid_mask_metrics]

                bins = np.linspace(min(gt_valid.min(), pred_valid.min()),
                                  max(gt_valid.max(), pred_valid.max()), 50)

                ax9.hist(gt_valid, bins=bins, alpha=0.6, label='Ground Truth',
                        color='blue', density=True)
                ax9.hist(pred_valid, bins=bins, alpha=0.6, label='Predicted',
                        color='red', density=True)
                ax9.set_xlabel('Depth (meters)', fontsize=12)
                ax9.set_ylabel('Density', fontsize=12)
                ax9.set_title('Depth Distribution', fontsize=14, fontweight='bold')
                ax9.legend(fontsize=12)
                ax9.grid(True, alpha=0.3)

            # 10. Importance Distribution
            ax10 = fig.add_subplot(gs[2, 2:])
            importance_flat = importance_frame.flatten()

            # Handle case where all values are identical (std=0)
            if imp_std < 1e-6:
                # Just show a vertical line at the constant value
                ax10.axvline(imp_mean, color='purple', linestyle='-', linewidth=3,
                            label=f'Constant: {imp_mean:.3f}')
                ax10.set_xlim(max(0, imp_mean - 0.1), min(1, imp_mean + 0.1))
                ax10.text(0.5, 0.5, f'All pixels = {imp_mean:.3f}\n(std = {imp_std:.6f})',
                         ha='center', va='center', transform=ax10.transAxes,
                         fontsize=14, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
            else:
                # Normal histogram
                ax10.hist(importance_flat, bins=50, alpha=0.7, color='purple', density=True)
                ax10.axvline(imp_mean, color='red', linestyle='--', linewidth=2,
                            label=f'Mean: {imp_mean:.3f}')

            ax10.set_xlabel('Importance Value', fontsize=12)
            ax10.set_ylabel('Density', fontsize=12)
            ax10.set_title('Importance Map Distribution', fontsize=14, fontweight='bold')
            ax10.legend(fontsize=12)
            ax10.grid(True, alpha=0.3)

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
