"""
Gear3 Upgrade Visualization Utilities

Based on gear3_visualization.py with FG/BG mask visualization added.
FG/BG visualization matches visualize_attention_weights.py exactly.
"""

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
from .sparse_depth_visualization import create_dual_sparse_depth_vis


class Gear3UpgradeVisualizer:
    """
    Visualization utilities for Gear3 Upgrade training

    Based on Gear3Visualizer with FG/BG mask visualization added.
    FG/BG overlay matches visualize_attention_weights.py implementation.
    """

    def __init__(self, save_dir="./visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.logger = logging.getLogger(__name__)

        # Set up matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")

    def create_validation_summary(self, sample_batch, model_outputs, step, save_name=None, prefix="validation", fps=None, loss_dict=None):
        """
        Create a comprehensive validation summary for Gear3 Upgrade

        Layout (4 rows × 3 columns):
        Row 1: Input, GT Depth, Pred Depth
        Row 2: Importance Map, FG Mask, BG Mask
        Row 3: Valid Mask, Error Map, Metrics & Training Info
        Row 4: Depth Distribution (colspan=2), Importance Distribution (colspan=1)

        Args:
            sample_batch: Validation batch (images, gt_depth, dataset_idx)
            model_outputs: Dictionary with 'pred_depth', 'importance_map', 'fg_mask', 'bg_mask'
            step: Training step number
            save_name: Optional custom save name
            prefix: Prefix for the visualization
            fps: Forward pass FPS (optional)
            loss_dict: Dictionary with loss values (optional)

        Returns:
            fig: Matplotlib figure object
        """
        try:
            images, gt_depth, dataset_idx = sample_batch
            pred_depth = model_outputs['pred_depth']
            importance_map = model_outputs['importance_map']
            fg_mask = model_outputs.get('fg_mask', None)
            bg_mask = model_outputs.get('bg_mask', None)

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

            # FG/BG masks (Gear3 Upgrade specific)
            if fg_mask is not None:
                if not hasattr(fg_mask, 'cpu'):
                    fg_mask = torch.from_numpy(fg_mask) if isinstance(fg_mask, np.ndarray) else fg_mask
                if fg_mask.ndim == 4:
                    fg_mask_frame = fg_mask[0, 0].cpu().numpy()
                else:
                    fg_mask_frame = fg_mask[0].cpu().numpy()
            else:
                fg_mask_frame = None

            if bg_mask is not None:
                if not hasattr(bg_mask, 'cpu'):
                    bg_mask = torch.from_numpy(bg_mask) if isinstance(bg_mask, np.ndarray) else bg_mask
                if bg_mask.ndim == 4:
                    bg_mask_frame = bg_mask[0, 0].cpu().numpy()
                else:
                    bg_mask_frame = bg_mask[0].cpu().numpy()
            else:
                bg_mask_frame = None

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
            if fg_mask_frame is not None:
                while fg_mask_frame.ndim > 2:
                    fg_mask_frame = fg_mask_frame[0]
            if bg_mask_frame is not None:
                while bg_mask_frame.ndim > 2:
                    bg_mask_frame = bg_mask_frame[0]

            print(f"DEBUG After squeeze:")
            print(f"  gt_depth_frame: {gt_depth_frame.shape}")
            print(f"  pred_depth_frame: {pred_depth_frame.shape}")
            print(f"  importance_frame: {importance_frame.shape}")

            # Create valid mask (IDENTICAL to Gear3)
            MAX_DEPTH = 200.0  # Maximum valid depth in meters
            gt_valid_mask = (gt_depth_frame > 0) & (gt_depth_frame < MAX_DEPTH)
            pred_valid_mask = (pred_depth_frame > 0) & (pred_depth_frame < MAX_DEPTH)
            valid_mask = gt_valid_mask & pred_valid_mask

            # Debug: Print statistics (IDENTICAL to Gear3)
            print(f"DEBUG Step {step}:")
            print(f"  GT raw range: {gt_depth_frame.min():.3f} - {gt_depth_frame.max():.3f}")
            print(f"  Pred raw range: {pred_depth_frame.min():.3f} - {pred_depth_frame.max():.3f}")
            print(f"  Invalid GT pixels (>200m or <0): {((gt_depth_frame <= 0) | (gt_depth_frame >= MAX_DEPTH)).sum()}")

            if valid_mask.sum() > 0:
                gt_valid = gt_depth_frame[valid_mask]
                pred_valid = pred_depth_frame[valid_mask]
                print(f"  GT valid range: {gt_valid.min():.3f} - {gt_valid.max():.3f}")
                print(f"  Pred valid range: {pred_valid.min():.3f} - {pred_valid.max():.3f}")
                print(f"  Valid pixels: {valid_mask.sum()} / {gt_depth_frame.size}")

            # Create figure with subplots
            # NEW LAYOUT: 4 rows × 3 columns
            fig = plt.figure(figsize=(15, 16))
            gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)

            # Calculate valid ratio and error BEFORE visualization (IDENTICAL to Gear3)
            valid_ratio = valid_mask.sum() / valid_mask.size
            abs_error = np.abs(pred_depth_frame - gt_depth_frame)
            abs_error_masked = np.where(valid_mask, abs_error, np.nan)

            # Calculate importance statistics
            imp_mean = importance_frame.mean()
            imp_std = importance_frame.std()

            # ==================== Row 1: Input, GT, Pred ====================

            # 1. Input Image
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(input_img)
            ax1.set_title('Input Image', fontsize=14, fontweight='bold')
            ax1.axis('off')

            # 2. Ground Truth Depth (with sparse depth handling - IDENTICAL to Gear3)
            ax2 = fig.add_subplot(gs[0, 1])

            # Use enhanced sparse visualization if valid_ratio < 50% (e.g., Waymo LiDAR)
            if valid_ratio < 0.5:
                # Sparse depth: use dual visualization (inpainted within valid row range only)
                _, gt_dense_vis, gt_info = create_dual_sparse_depth_vis(
                    gt_depth_frame, valid_mask, colormap='plasma', percentile_range=(2, 98)
                )
                im2 = ax2.imshow(gt_dense_vis)
                ax2.set_title(f'GT Depth (Inpainted)\n{valid_ratio*100:.1f}% valid',
                             fontsize=14, fontweight='bold')
                vmin, vmax = gt_info['vmin'], gt_info['vmax']
            else:
                # Dense depth: use standard visualization
                gt_display = np.where(valid_mask, gt_depth_frame, np.nan)
                vmin, vmax = np.nanpercentile(gt_display, [2, 98])
                im2 = ax2.imshow(gt_display, cmap='plasma', vmin=vmin, vmax=vmax)
                ax2.set_title('Ground Truth Depth (m)', fontsize=14, fontweight='bold')

            ax2.axis('off')
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

            # 3. Predicted Metric Depth (with sparse depth handling - IDENTICAL to Gear3)
            ax3 = fig.add_subplot(gs[0, 2])

            if valid_ratio < 0.5:
                # Sparse depth: use dual visualization (inpainted within valid row range only)
                _, pred_dense_vis, pred_info = create_dual_sparse_depth_vis(
                    pred_depth_frame, valid_mask, colormap='plasma', percentile_range=(2, 98)
                )
                im3 = ax3.imshow(pred_dense_vis)
                ax3.set_title(f'Pred Depth (Inpainted)\nMAE: {np.nanmean(abs_error_masked):.2f}m',
                             fontsize=14, fontweight='bold')
            else:
                # Dense depth: use standard visualization
                pred_display = np.where(valid_mask, pred_depth_frame, np.nan)
                im3 = ax3.imshow(pred_display, cmap='plasma', vmin=vmin, vmax=vmax)
                ax3.set_title('Predicted Metric Depth (m)', fontsize=14, fontweight='bold')

            ax3.axis('off')
            plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

            # ==================== Row 2: Importance, FG, BG ====================

            # 4. Importance Map (IDENTICAL to Gear3)
            ax4 = fig.add_subplot(gs[1, 0])
            im4 = ax4.imshow(importance_frame, cmap='jet', vmin=0, vmax=1)
            ax4.set_title(f'Importance Map\nmean={imp_mean:.3f}, std={imp_std:.3f}',
                         fontsize=14, fontweight='bold')
            ax4.axis('off')
            plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

            # Calculate FG/BG from importance map (fallback if masks not provided)
            if fg_mask_frame is None or bg_mask_frame is None:
                fg_mask_frame = (importance_frame >= imp_mean).astype(np.float32)
                bg_mask_frame = (importance_frame < imp_mean).astype(np.float32)
            else:
                # IMPORTANT: Convert soft masks to binary (like visualize_attention_weights.py)
                # cls_seg and kmeans return soft probabilities, but we want binary visualization
                # Use mean threshold on the mask itself (or 0.5 for probabilistic masks)
                fg_mask_frame = (fg_mask_frame > 0.5).astype(np.float32)
                bg_mask_frame = (bg_mask_frame > 0.5).astype(np.float32)

            # Upsample FG/BG masks to match input image resolution (like visualize_attention_weights.py)
            img_h, img_w = input_img.shape[:2]
            fg_mask_upsampled = F.interpolate(
                torch.from_numpy(fg_mask_frame).unsqueeze(0).unsqueeze(0),
                size=(img_h, img_w),
                mode='nearest'
            ).squeeze().numpy()

            bg_mask_upsampled = F.interpolate(
                torch.from_numpy(bg_mask_frame).unsqueeze(0).unsqueeze(0),
                size=(img_h, img_w),
                mode='nearest'
            ).squeeze().numpy()

            # 5. FG Mask (visualize_attention_weights.py style)
            ax5 = fig.add_subplot(gs[1, 1])
            ax5.imshow(input_img)
            # Create FG overlay (Red channel only, as in visualize_attention_weights.py)
            fg_overlay = np.zeros((*fg_mask_upsampled.shape, 3))
            fg_overlay[..., 0] = fg_mask_upsampled  # Red channel
            ax5.imshow(fg_overlay, alpha=0.5)  # Use imshow alpha parameter
            fg_ratio = fg_mask_frame.mean() * 100  # Use original for ratio (not upsampled)
            ax5.set_title(f'FG Mask (Red)\n{fg_ratio:.1f}%', fontsize=14, fontweight='bold')
            ax5.axis('off')

            # 6. BG Mask (visualize_attention_weights.py style)
            ax6 = fig.add_subplot(gs[1, 2])
            ax6.imshow(input_img)
            # Create BG overlay (Blue channel only, as in visualize_attention_weights.py)
            bg_overlay = np.zeros((*bg_mask_upsampled.shape, 3))
            bg_overlay[..., 2] = bg_mask_upsampled  # Blue channel
            ax6.imshow(bg_overlay, alpha=0.5)  # Use imshow alpha parameter
            bg_ratio = bg_mask_frame.mean() * 100  # Use original for ratio (not upsampled)
            ax6.set_title(f'BG Mask (Blue)\n{bg_ratio:.1f}%', fontsize=14, fontweight='bold')
            ax6.axis('off')

            # ==================== Row 3: Valid Mask, Error, Metrics & Training Info ====================

            # 7. Valid Mask
            ax7 = fig.add_subplot(gs[2, 0])
            ax7.imshow(valid_mask.astype(np.uint8), cmap='gray_r', vmin=0, vmax=1)
            ax7.set_title(f'Valid Mask\n({valid_mask.sum():,} pixels)', fontsize=14, fontweight='bold')
            ax7.axis('off')

            # 8. Absolute Error Map
            ax8 = fig.add_subplot(gs[2, 1])
            error_vmax = np.nanpercentile(abs_error_masked, 95)
            im8 = ax8.imshow(abs_error_masked, cmap='hot', vmin=0, vmax=error_vmax)
            ax8.set_title(f'Absolute Error (m)\nMean: {np.nanmean(abs_error_masked):.3f}',
                         fontsize=14, fontweight='bold')
            ax8.axis('off')
            plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)

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

            # FG:BG ratio
            ax9.text(0.05, y_pos, f'FG:BG = {fg_ratio:.1f}:{bg_ratio:.1f}', fontsize=10,
                    transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightcyan'))
            y_pos -= 0.10

            # FPS if available
            if fps is not None:
                ax9.text(0.05, y_pos, f'FPS: {fps:.1f}', fontsize=10,
                        transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen'))
                y_pos -= 0.10

            # Depth metrics
            if valid_mask.sum() > 0:
                pred_tensor = torch.from_numpy(pred_depth_frame).float()
                gt_tensor = torch.from_numpy(gt_depth_frame).float()
                valid_tensor = torch.from_numpy(valid_mask).bool()

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
                if 'val_loss' in loss_dict:
                    ax9.text(0.05, y_pos, f'Val Loss: {loss_dict["val_loss"]:.4f}', fontsize=9,
                            transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightcoral'))
                    y_pos -= 0.08

                if 'depth_loss' in loss_dict:
                    ax9.text(0.05, y_pos, f'Log L1: {loss_dict["depth_loss"]:.4f}', fontsize=9,
                            transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightcoral'))
                    y_pos -= 0.08

                if 'depth_variance_loss' in loss_dict and loss_dict['depth_variance_loss'] > 0:
                    ax9.text(0.05, y_pos, f'Var: {loss_dict["depth_variance_loss"]:.4f}', fontsize=9,
                            transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightyellow'))
                    y_pos -= 0.08

                if 'edge_aware_loss' in loss_dict and loss_dict['edge_aware_loss'] > 0:
                    ax9.text(0.05, y_pos, f'Edge: {loss_dict["edge_aware_loss"]:.4f}', fontsize=9,
                            transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightblue'))
                    y_pos -= 0.08

                if 'contrastive_fgbg_loss' in loss_dict and loss_dict['contrastive_fgbg_loss'] > 0:
                    ax9.text(0.05, y_pos, f'Contrast: {loss_dict["contrastive_fgbg_loss"]:.4f}', fontsize=9,
                            transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen'))

            ax9.set_title('Depth Metrics & Training Info', fontsize=14, fontweight='bold')
            ax9.axis('off')

            # ==================== Row 4: Depth Distribution, Importance Distribution ====================

            # 10. Depth Distribution Histogram
            ax10 = fig.add_subplot(gs[3, :2])
            if valid_mask.sum() > 0:
                gt_valid = gt_depth_frame[valid_mask]
                pred_valid = pred_depth_frame[valid_mask]

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

            # 11. Importance Distribution
            ax11 = fig.add_subplot(gs[3, 2])
            importance_flat = importance_frame.flatten()

            # Handle case where all values are identical (std=0)
            if imp_std < 1e-6:
                # Just show a vertical line at the constant value
                ax11.axvline(imp_mean, color='purple', linestyle='-', linewidth=3,
                            label=f'Constant: {imp_mean:.3f}')
                ax11.set_xlim(max(0, imp_mean - 0.1), min(1, imp_mean + 0.1))
                ax11.text(0.5, 0.5, f'All pixels = {imp_mean:.3f}\n(std = {imp_std:.6f})',
                         ha='center', va='center', transform=ax11.transAxes,
                         fontsize=14, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
            else:
                # Normal histogram
                ax11.hist(importance_flat, bins=50, alpha=0.7, color='purple', density=True)
                ax11.axvline(imp_mean, color='red', linestyle='--', linewidth=2,
                            label=f'Mean: {imp_mean:.3f}')

            ax11.set_xlabel('Importance Value', fontsize=12)
            ax11.set_ylabel('Density', fontsize=12)
            ax11.set_title('Importance Map Distribution', fontsize=14, fontweight='bold')
            ax11.legend(fontsize=12)
            ax11.grid(True, alpha=0.3)

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
