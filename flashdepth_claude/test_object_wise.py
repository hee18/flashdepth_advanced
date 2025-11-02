"""
Test script for object-wise depth evaluation.

Evaluates depth estimation accuracy per segmentation class to demonstrate
improvements on specific object types.

Usage:
    # Evaluate single model on KITTI
    python test_object_wise.py --model-checkpoint train_results/results_14/gear_3/best_checkpoint.pth \
                                --config-path configs/gear3 \
                                --dataset kitti \
                                --data-root /home/cvlab/hsy/Datasets/KITTI \
                                --results-dir test_results/object_wise/gear3_kitti

    # Compare two models
    python test_object_wise.py --model-checkpoint train_results/results_14/gear_3/best_checkpoint.pth \
                                --baseline-checkpoint train_results/baseline/best_checkpoint.pth \
                                --config-path configs/gear3 \
                                --dataset kitti \
                                --data-root /home/cvlab/hsy/Datasets/KITTI \
                                --results-dir test_results/object_wise/comparison
"""

import argparse
import logging
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.object_wise_evaluation import ObjectWiseMetrics, load_segmentation_mask
from flashdepth.model import FlashDepth

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ObjectWiseEvaluator:
    """Evaluator for object-wise depth metrics."""

    def __init__(
        self,
        model_checkpoint: Path,
        config_path: Path,
        dataset_type: str,
        device: str = 'cuda',
        baseline_checkpoint: Path = None
    ):
        """
        Initialize evaluator.

        Args:
            model_checkpoint: Path to model checkpoint
            config_path: Path to model config directory
            dataset_type: Dataset type ('kitti', 'cityscapes', 'nyu', 'vkitti2')
            device: Device to run on
            baseline_checkpoint: Optional baseline model checkpoint for comparison
        """
        self.device = device
        self.dataset_type = dataset_type

        # Load config
        config_file = config_path / 'config.yaml'
        self.config = OmegaConf.load(config_file)

        # Initialize metrics calculator
        self.metrics_calculator = ObjectWiseMetrics(dataset_type=dataset_type)

        # Load main model
        logger.info(f"Loading model from {model_checkpoint}")
        self.model = self._load_model(model_checkpoint)
        self.model_name = "Gear3"

        # Load baseline model if provided
        self.baseline_model = None
        self.baseline_name = None
        if baseline_checkpoint is not None:
            logger.info(f"Loading baseline model from {baseline_checkpoint}")
            self.baseline_model = self._load_model(baseline_checkpoint)
            self.baseline_name = "Baseline"

    def _load_model(self, checkpoint_path: Path) -> FlashDepth:
        """Load FlashDepth model from checkpoint."""
        # Initialize model
        model = FlashDepth(self.config)
        model = model.to(self.device)
        model.eval()

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if present (from DDP)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")

        return model

    def _predict_depth(
        self,
        model: FlashDepth,
        images: torch.Tensor,
        return_visualization_data: bool = False
    ):
        """
        Predict depth for a sequence of images using full sequence processing.
        Supports both original FlashDepth and Gear models (Gear2/3/3 Upgrade).

        Args:
            model: FlashDepth model (with or without Gear modules)
            images: Input images (B, T, C, H, W)
            return_visualization_data: If True, return additional data for visualization

        Returns:
            If return_visualization_data is False:
                Predicted depth map for last frame (H, W)
            If return_visualization_data is True:
                Tuple of (pred_depth, importance_map, fg_mask, bg_mask, images_last)
        """
        from einops import rearrange

        B_orig, T_orig, C, H, W = images.shape

        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # Initialize Mamba sequence (critical for temporal processing!)
                if hasattr(model, 'mamba'):
                    model.mamba.start_new_sequence()

                # Reshape video from (B, T, C, H, W) to (B*T, C, H, W) for encoder
                images_flat = rearrange(images, 'b t c h w -> (b t) c h w')
                patch_h, patch_w = H // model.patch_size, W // model.patch_size

                # Extract features from DINOv2 - all frames at once
                encoder_features = model.pretrained.get_intermediate_layers(
                    images_flat, model.intermediate_layer_idx[model.encoder]
                )

                # Get DPT features with Mamba temporal processing
                dpt_output = model.depth_head.forward_with_mamba(
                    encoder_features, patch_h, patch_w,
                    temporal_layer=model.mamba_in_dpt_layer,
                    mamba_fn=model.dpt_features_to_mamba,
                    shape_placeholder=(B_orig, T_orig, None, H, W)
                )  # Returns path_1 with Mamba applied, shape: (B*T, dpt_dim, h, w)

                # Initialize visualization data
                importance_map = None
                fg_mask = None
                bg_mask = None

                # Check if this is a Gear model (has gear2_head, gear3_head, or gear3_upgrade_head)
                if hasattr(model, 'gear2_head') or hasattr(model, 'gear3_head'):
                    # Gear model: apply FG/BG modulation
                    # Get attention weights and patch tokens for Gear head
                    last_block = model.pretrained.blocks[-1]
                    attention_weights = last_block.attn.attn_weights
                    patch_tokens = encoder_features[-1]

                    # Determine which Gear head to use
                    if hasattr(model, 'gear3_head'):
                        # Gear3 or Gear3 Upgrade
                        # Prepare inputs based on separation_method (if available)
                        attention_weights_multi_layer = None
                        cls_token = None

                        if hasattr(model.gear3_head, 'multi_layer_fusion'):
                            # Multi-layer separation method
                            attention_weights_multi_layer = [
                                model.pretrained.blocks[i].attn.attn_weights
                                for i in [3, 10, 16, 22]
                            ]
                        else:
                            # CLS token separation method
                            cls_token = patch_tokens[:, 0]  # (B*T, embed_dim)

                        # Apply Gear3 modulation
                        path_1_modulated, importance_map, fg_features, bg_features, fg_mask, bg_mask = model.gear3_head(
                            patch_tokens, attention_weights, [dpt_output], patch_h, patch_w,
                            attention_weights_multi_layer=attention_weights_multi_layer,
                            cls_token=cls_token
                        )
                    else:
                        # Gear2
                        path_1_modulated, importance_map, fg_features, bg_features = model.gear2_head(
                            patch_tokens, attention_weights, [dpt_output], patch_h, patch_w
                        )
                        # Gear2 doesn't have explicit fg_mask/bg_mask, create from importance
                        importance_flat = importance_map.flatten(2).squeeze(1)  # [B*T, H*W]
                        imp_mean = importance_flat.mean(dim=1, keepdim=True)  # [B*T, 1]
                        fg_mask_flat = (importance_flat >= imp_mean).float()
                        bg_mask_flat = (importance_flat < imp_mean).float()
                        fg_mask = fg_mask_flat.reshape(-1, 1, patch_h, patch_w)
                        bg_mask = bg_mask_flat.reshape(-1, 1, patch_h, patch_w)

                    # Use modulated features
                    out = model.depth_head.scratch.output_conv1(path_1_modulated)
                else:
                    # Original FlashDepth: use DPT output directly
                    out = model.depth_head.scratch.output_conv1(dpt_output)

                # Final depth prediction
                out = F.interpolate(out, (H, W), mode="bilinear", align_corners=True)
                out = model.depth_head.scratch.output_conv2(out)

                # Output is inverse depth (100/m) with Softplus activation
                pred_depth_inverse = out  # Shape: (B*T, 1, H, W)

        # Reshape to (B, T, 1, H, W) and get last frame
        pred_depth_inverse_seq = rearrange(pred_depth_inverse, '(b t) 1 h w -> b t 1 h w', b=B_orig, t=T_orig)
        pred_depth_inverse_last = pred_depth_inverse_seq[0, -1, 0]  # Last frame (H, W)

        # Convert inverse depth to metric depth: 100/m -> m
        pred_depth_metric = 100.0 / (pred_depth_inverse_last.float() + 1e-8)

        if not return_visualization_data:
            # Return only depth as numpy array (original behavior)
            return pred_depth_metric.cpu().numpy()  # (H, W)
        else:
            # Return visualization data as well
            # Get last frame for each output
            if importance_map is not None:
                importance_map_seq = rearrange(importance_map, '(b t) c h w -> b t c h w', b=B_orig, t=T_orig)
                importance_map_last = importance_map_seq[0, -1, 0]  # [patch_h, patch_w]
            else:
                importance_map_last = None

            if fg_mask is not None:
                fg_mask_seq = rearrange(fg_mask, '(b t) c h w -> b t c h w', b=B_orig, t=T_orig)
                fg_mask_last = fg_mask_seq[0, -1, 0]  # [patch_h, patch_w]
            else:
                fg_mask_last = None

            if bg_mask is not None:
                bg_mask_seq = rearrange(bg_mask, '(b t) c h w -> b t c h w', b=B_orig, t=T_orig)
                bg_mask_last = bg_mask_seq[0, -1, 0]  # [patch_h, patch_w]
            else:
                bg_mask_last = None

            # Get last frame image
            images_last = images[0, -1]  # [C, H, W]

            return (
                pred_depth_metric.cpu().numpy(),  # (H, W)
                importance_map_last.cpu().numpy() if importance_map_last is not None else None,
                fg_mask_last.cpu().numpy() if fg_mask_last is not None else None,
                bg_mask_last.cpu().numpy() if bg_mask_last is not None else None,
                images_last.cpu()  # [C, H, W]
            )  # (H, W)

    def _save_best_frame_visualizations(self, image, pred_depth, gt_depth, object_mask,
                                        importance_map, fg_mask, bg_mask, 
                                        sequence_id, class_name, class_metrics, save_dir):
        """
        Save visualization for a sequence with Object Mask focus.
        All visualizations (GT, pred, FG/BG masks, error) shown within Object Mask boundaries.

        Creates a 4x3 grid visualization:
            Row 1: Input Image | GT Depth (masked) | Pred Depth (masked)
            Row 2: Importance Map | FG Mask | BG Mask
            Row 3: Object Mask | Error Map (masked) | Metrics
            Row 4: Depth Distribution (2 cols) | Importance Distribution

        Args:
            image: [3, H, W] - RGB image
            pred_depth: [H, W] - Predicted metric depth
            gt_depth: [H, W] - Ground truth metric depth
            object_mask: [H, W] - Object mask for the target class (boolean)
            importance_map: [patch_h, patch_w] - Importance map (0-1 normalized) or None
            fg_mask: [patch_h, patch_w] - FG mask or None
            bg_mask: [patch_h, patch_w] - BG mask or None
            sequence_id: int - Sequence index
            class_name: str - Class name being visualized
            class_metrics: dict - Metrics for this class
            save_dir: Path - Directory to save visualizations
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import torch.nn.functional as F

        # Convert tensors to numpy and move to CPU
        if isinstance(image, torch.Tensor):
            if image.shape[0] == 3:  # [3, H, W]
                image = image.permute(1, 2, 0)  # [H, W, 3]
            image = image.float().cpu().numpy()

        if isinstance(pred_depth, torch.Tensor):
            pred_depth = pred_depth.float().cpu().numpy()
        if isinstance(gt_depth, torch.Tensor):
            gt_depth = gt_depth.float().cpu().numpy()
        if isinstance(object_mask, torch.Tensor):
            object_mask = object_mask.cpu().numpy()

        # Denormalize ImageNet normalization for image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = image * std + mean  # Reverse normalization
        image_np = np.clip(image_np, 0, 1)  # Clip to valid range

        # Get image size
        img_h, img_w = image_np.shape[:2]

        # Create valid mask within object boundaries
        MAX_DEPTH = 70.0  # Same as training
        gt_valid = (gt_depth > 0) & (gt_depth < MAX_DEPTH)
        pred_valid = (pred_depth > 0) & (pred_depth < 1000)
        valid_mask = gt_valid & pred_valid & object_mask

        # Calculate error only within object mask
        abs_error = np.abs(pred_depth - gt_depth)
        abs_error_masked = np.where(valid_mask, abs_error, np.nan)

        # Process importance map and masks if available
        if importance_map is not None and isinstance(importance_map, torch.Tensor):
            importance_map = importance_map.float().cpu().numpy()
        if fg_mask is not None and isinstance(fg_mask, torch.Tensor):
            fg_mask = fg_mask.float().cpu().numpy()
        if bg_mask is not None and isinstance(bg_mask, torch.Tensor):
            bg_mask = bg_mask.float().cpu().numpy()

        # Upsample importance/masks to image resolution if available
        if importance_map is not None:
            imp_mean = importance_map.mean()
            imp_std = importance_map.std()
            importance_upsampled = F.interpolate(
                torch.from_numpy(importance_map).unsqueeze(0).unsqueeze(0),
                size=(img_h, img_w),
                mode='bilinear',
                align_corners=True
            ).squeeze().numpy()
            # Mask by object_mask
            importance_upsampled = np.where(object_mask, importance_upsampled, np.nan)
        else:
            importance_upsampled = None
            imp_mean = 0
            imp_std = 0

        if fg_mask is not None:
            fg_mask_upsampled = F.interpolate(
                torch.from_numpy(fg_mask).unsqueeze(0).unsqueeze(0),
                size=(img_h, img_w),
                mode='bilinear',
                align_corners=True
            ).squeeze().numpy()
            # Mask by object_mask
            fg_mask_upsampled = np.where(object_mask, fg_mask_upsampled, 0)
            fg_ratio = (fg_mask >= fg_mask.mean()).mean() * 100 if fg_mask is not None else 0
        else:
            fg_mask_upsampled = None
            fg_ratio = 0

        if bg_mask is not None:
            bg_mask_upsampled = F.interpolate(
                torch.from_numpy(bg_mask).unsqueeze(0).unsqueeze(0),
                size=(img_h, img_w),
                mode='bilinear',
                align_corners=True
            ).squeeze().numpy()
            # Mask by object_mask
            bg_mask_upsampled = np.where(object_mask, bg_mask_upsampled, 0)
            bg_ratio = (bg_mask < bg_mask.mean()).mean() * 100 if bg_mask is not None else 0
        else:
            bg_mask_upsampled = None
            bg_ratio = 0

        # Create figure with 4x3 grid layout
        fig = plt.figure(figsize=(15, 16))
        gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)

        # ==================== Row 1: Input, GT, Pred ====================

        # 1. Input Image with Object Mask overlay
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image_np)
        # Show object boundary
        from matplotlib.patches import Rectangle
        from matplotlib import patches
        object_overlay = np.zeros((*object_mask.shape, 4))
        object_overlay[..., 1] = object_mask.astype(float)  # Green channel
        object_overlay[..., 3] = object_mask.astype(float) * 0.3  # Alpha
        ax1.imshow(object_overlay)
        ax1.set_title(f'Input Image\n(Object: {class_name})', fontsize=14, fontweight='bold')
        ax1.axis('off')

        # 2. Ground Truth Depth (masked by object)
        ax2 = fig.add_subplot(gs[0, 1])
        gt_display = np.where(object_mask, gt_depth, np.nan)
        if valid_mask.sum() > 0:
            vmin, vmax = np.nanpercentile(gt_display[object_mask], [2, 98])
        else:
            vmin, vmax = 0, 1
        im2 = ax2.imshow(gt_display, cmap='plasma', vmin=vmin, vmax=vmax)
        ax2.set_title('GT Depth (m)\n(Within Object Mask)', fontsize=14, fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # 3. Predicted Metric Depth (masked by object)
        ax3 = fig.add_subplot(gs[0, 2])
        pred_display = np.where(object_mask, pred_depth, np.nan)
        im3 = ax3.imshow(pred_display, cmap='plasma', vmin=vmin, vmax=vmax)
        ax3.set_title('Pred Depth (m)\n(Within Object Mask)', fontsize=14, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        # ==================== Row 2: Importance, FG, BG ====================

        # 4. Importance Map (masked by object)
        ax4 = fig.add_subplot(gs[1, 0])
        if importance_upsampled is not None:
            im4 = ax4.imshow(importance_upsampled, cmap='jet', vmin=0, vmax=1)
            ax4.set_title(f'Importance Map\n(Within Object)\nmean={imp_mean:.3f}, std={imp_std:.3f}',
                         fontsize=14, fontweight='bold')
            plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        else:
            ax4.text(0.5, 0.5, 'No Importance Map\n(Not a Gear model)',
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Importance Map', fontsize=14, fontweight='bold')
        ax4.axis('off')

        # 5. FG Mask (Red overlay, within object)
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.imshow(image_np)
        if fg_mask_upsampled is not None:
            # Create FG overlay (Red channel only)
            fg_overlay = np.zeros((*fg_mask_upsampled.shape, 3))
            fg_overlay[..., 0] = fg_mask_upsampled  # Red channel
            ax5.imshow(fg_overlay, alpha=0.5)
            ax5.set_title(f'FG Mask (Red)\n(Within Object)\n{fg_ratio:.1f}%', fontsize=14, fontweight='bold')
        else:
            ax5.text(0.5, 0.5, 'No FG Mask',
                    ha='center', va='center', transform=ax5.transAxes, fontsize=12)
            ax5.set_title('FG Mask', fontsize=14, fontweight='bold')
        ax5.axis('off')

        # 6. BG Mask (Blue overlay, within object)
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.imshow(image_np)
        if bg_mask_upsampled is not None:
            # Create BG overlay (Blue channel only)
            bg_overlay = np.zeros((*bg_mask_upsampled.shape, 3))
            bg_overlay[..., 2] = bg_mask_upsampled  # Blue channel
            ax6.imshow(bg_overlay, alpha=0.5)
            ax6.set_title(f'BG Mask (Blue)\n(Within Object)\n{bg_ratio:.1f}%', fontsize=14, fontweight='bold')
        else:
            ax6.text(0.5, 0.5, 'No BG Mask',
                    ha='center', va='center', transform=ax6.transAxes, fontsize=12)
            ax6.set_title('BG Mask', fontsize=14, fontweight='bold')
        ax6.axis('off')

        # ==================== Row 3: Object Mask, Error, Metrics ====================

        # 7. Object Mask
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.imshow(object_mask.astype(np.uint8), cmap='gray_r', vmin=0, vmax=1)
        object_ratio = object_mask.sum() / object_mask.size
        ax7.set_title(f'Object Mask: {class_name}\n{object_ratio*100:.1f}% ({object_mask.sum():,} pixels)',
                     fontsize=14, fontweight='bold')
        ax7.axis('off')

        # 8. Absolute Error Map (within object mask)
        ax8 = fig.add_subplot(gs[2, 1])
        if valid_mask.sum() > 0:
            error_vmax = np.nanpercentile(abs_error_masked, 95)
        else:
            error_vmax = 1
        im8 = ax8.imshow(abs_error_masked, cmap='hot', vmin=0, vmax=error_vmax)
        ax8.set_title(f'Absolute Error (m)\n(Within Object)\nMean: {np.nanmean(abs_error_masked):.3f}',
                     fontsize=14, fontweight='bold')
        ax8.axis('off')
        plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)

        # 9. Depth Metrics (from class_metrics)
        ax9 = fig.add_subplot(gs[2, 2])
        y_pos = 0.95

        # Sequence and class info
        ax9.text(0.05, y_pos, f'Seq {sequence_id+1} - {class_name}', fontsize=11,
                transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'),
                fontweight='bold')
        y_pos -= 0.12

        # Object coverage
        ax9.text(0.05, y_pos, f'Object pixels: {class_metrics["num_pixels"]:,}', fontsize=9,
                transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightcyan'))
        y_pos -= 0.10

        # Depth metrics from class_metrics
        if 'abs_rel' in class_metrics:
            ax9.text(0.05, y_pos, f'AbsRel: {class_metrics["abs_rel"]:.4f}', fontsize=10,
                    transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightcoral'))
            y_pos -= 0.08
        if 'delta_1' in class_metrics:
            ax9.text(0.05, y_pos, f'Delta_1: {class_metrics["delta_1"]:.3f}', fontsize=10,
                    transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen'))
            y_pos -= 0.08
        if 'delta_2' in class_metrics:
            ax9.text(0.05, y_pos, f'Delta_2: {class_metrics["delta_2"]:.3f}', fontsize=10,
                    transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen'))
            y_pos -= 0.08
        if 'delta_3' in class_metrics:
            ax9.text(0.05, y_pos, f'Delta_3: {class_metrics["delta_3"]:.3f}', fontsize=10,
                    transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen'))
            y_pos -= 0.08
        if 'rmse' in class_metrics:
            ax9.text(0.05, y_pos, f'RMSE: {class_metrics["rmse"]:.3f}m', fontsize=9,
                    transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
            y_pos -= 0.08
        if 'mae' in class_metrics:
            ax9.text(0.05, y_pos, f'MAE: {class_metrics["mae"]:.3f}m', fontsize=9,
                    transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightblue'))

        ax9.set_title('Depth Metrics\n(Within Object)', fontsize=14, fontweight='bold')
        ax9.axis('off')

        # ==================== Row 4: Depth Distribution, Importance Distribution ====================

        # 10. Depth Distribution Histogram (within object mask)
        ax10 = fig.add_subplot(gs[3, :2])
        if valid_mask.sum() > 0:
            gt_valid = gt_depth[valid_mask]
            pred_valid = pred_depth[valid_mask]

            bins = np.linspace(min(gt_valid.min(), pred_valid.min()),
                              max(gt_valid.max(), pred_valid.max()), 50)

            ax10.hist(gt_valid, bins=bins, alpha=0.6, label='Ground Truth',
                    color='blue', density=True)
            ax10.hist(pred_valid, bins=bins, alpha=0.6, label='Predicted',
                    color='red', density=True)
            ax10.set_xlabel('Depth (meters)', fontsize=12)
            ax10.set_ylabel('Density', fontsize=12)
            ax10.set_title(f'Depth Distribution (Within {class_name})', fontsize=14, fontweight='bold')
            ax10.legend(fontsize=12)
            ax10.grid(True, alpha=0.3)

        # 11. Importance Distribution (within object mask)
        ax11 = fig.add_subplot(gs[3, 2])
        if importance_map is not None and importance_upsampled is not None:
            importance_in_object = importance_upsampled[object_mask & ~np.isnan(importance_upsampled)]
            
            if len(importance_in_object) > 0:
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
                    ax11.hist(importance_in_object, bins=50, alpha=0.7, color='purple', density=True)
                    ax11.axvline(imp_mean, color='red', linestyle='--', linewidth=2,
                                label=f'Mean: {imp_mean:.3f}')

                ax11.set_xlabel('Importance Value', fontsize=12)
                ax11.set_ylabel('Density', fontsize=12)
                ax11.set_title(f'Importance Distribution\n(Within {class_name})', fontsize=14, fontweight='bold')
                ax11.legend(fontsize=10)
                ax11.grid(True, alpha=0.3)
            else:
                ax11.text(0.5, 0.5, 'No valid importance values',
                         ha='center', va='center', transform=ax11.transAxes, fontsize=12)
        else:
            ax11.text(0.5, 0.5, 'No Importance Map',
                     ha='center', va='center', transform=ax11.transAxes, fontsize=12)
        ax11.set_title('Importance Distribution', fontsize=14, fontweight='bold')

        # Overall title
        plt.suptitle(f'Object-wise Evaluation: Sequence {sequence_id+1} - {class_name}',
                    fontsize=16, fontweight='bold')

        # Save visualization
        save_path = save_dir / f"seq{sequence_id+1}_{class_name}_absrel_{class_metrics.get('abs_rel', 0):.4f}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        logger.info(f"Saved object-wise visualization: {save_path}")

    def evaluate_sequence(
        self,
        images: torch.Tensor,
        gt_depth: np.ndarray,
        seg_mask: np.ndarray,
        save_visualization: bool = False
    ):
        """
        Evaluate a single sequence.

        Args:
            images: Input images (B, T, C, H, W)
            gt_depth: Ground truth depth (H, W)
            seg_mask: Segmentation mask (H, W)
            save_visualization: Whether to return visualization data

        Returns:
            Dictionary of per-class metrics, and optionally visualization data
        """
        # Predict depth with main model
        if save_visualization:
            pred_depth, importance_map, fg_mask, bg_mask, image_last = self._predict_depth(
                self.model, images, return_visualization_data=True
            )
            vis_data = {
                'pred_depth': pred_depth,
                'importance_map': importance_map,
                'fg_mask': fg_mask,
                'bg_mask': bg_mask,
                'image': image_last,
                'gt_depth': gt_depth,
                'seg_mask': seg_mask
            }
        else:
            pred_depth = self._predict_depth(self.model, images, return_visualization_data=False)
            vis_data = None

        # Compute per-class metrics
        class_metrics = self.metrics_calculator.compute_metrics_per_class(
            pred_depth, gt_depth, seg_mask
        )

        results = {self.model_name: class_metrics}

        # Evaluate baseline if available
        if self.baseline_model is not None:
            baseline_pred_depth = self._predict_depth(self.baseline_model, images, return_visualization_data=False)
            baseline_class_metrics = self.metrics_calculator.compute_metrics_per_class(
                baseline_pred_depth, gt_depth, seg_mask
            )
            results[self.baseline_name] = baseline_class_metrics

        if save_visualization:
            return results, vis_data
        else:
            return results

    def evaluate_dataset(
        self,
        dataloader: DataLoader,
        max_sequences: int = None,
        save_dir: Path = None,
        num_vis_sequences: int = 5
    ) -> dict:
        """
        Evaluate entire dataset.

        Args:
            dataloader: PyTorch dataloader
            max_sequences: Maximum number of sequences to evaluate (None = all)
            save_dir: Directory to save visualizations (None = no visualization)
            num_vis_sequences: Number of sequences to visualize

        Returns:
            Dictionary with aggregated metrics and comparison
        """
        all_metrics = {self.model_name: []}
        if self.baseline_model is not None:
            all_metrics[self.baseline_name] = []

        num_sequences = 0

        for batch in tqdm(dataloader, desc="Evaluating sequences"):
            # Add batch dimension if not present (dataset returns (T, C, H, W))
            images = batch['image']  # (T, C, H, W) or (B, T, C, H, W)
            if images.ndim == 4:
                images = images.unsqueeze(0)  # Add batch dim -> (1, T, C, H, W)
            images = images.to(self.device)  # (B, T, C, H, W)

            gt_depth = batch['depth'].cpu().numpy()[0, -1]  # Last frame (H, W)
            seg_mask = batch['segmentation'].cpu().numpy()[0]  # (H, W)

            # Evaluate this sequence (with visualization data if needed)
            save_vis = (save_dir is not None) and (num_sequences < num_vis_sequences)
            
            if save_vis:
                seq_metrics, vis_data = self.evaluate_sequence(
                    images, gt_depth, seg_mask, save_visualization=True
                )
            else:
                seq_metrics = self.evaluate_sequence(
                    images, gt_depth, seg_mask, save_visualization=False
                )
                vis_data = None

            # Accumulate metrics
            for model_name, class_metrics in seq_metrics.items():
                all_metrics[model_name].append(class_metrics)

            # Save visualization for this sequence (if enabled)
            if save_vis and vis_data is not None:
                # Get class metrics for main model
                class_metrics_main = seq_metrics[self.model_name]
                
                if len(class_metrics_main) > 0:
                    # Filter for object classes only (dynamic objects)
                    object_classes_to_visualize = []
                    
                    for class_name, class_metric in class_metrics_main.items():
                        # Check if this class is an object class
                        is_object_class = (class_name in self.metrics_calculator.object_classes)
                        
                        if is_object_class:
                            object_classes_to_visualize.append((class_name, class_metric))
                    
                    # Sort by number of pixels (descending) to visualize most prominent objects first
                    object_classes_to_visualize.sort(key=lambda x: x[1]['num_pixels'], reverse=True)
                    
                    # Visualize all object classes in this sequence
                    for class_name, class_metric in object_classes_to_visualize:
                        # Create object mask based on class ID
                        # Get class ID from class name
                        class_id = None
                        for cid, cname in self.metrics_calculator.classes.items():
                            if cname == class_name:
                                class_id = cid
                                break

                        if class_id is None:
                            logger.warning(f"Could not find class ID for {class_name}, skipping")
                                continue
                            
                            # Create object mask for this class
                            object_mask = (vis_data['seg_mask'] == class_id)
                        
                        # Skip if too few pixels (sanity check)
                        if object_mask.sum() < 100:
                            continue
                        
                        # Save visualization
                        try:
                            self._save_best_frame_visualizations(
                                image=vis_data['image'],
                                pred_depth=vis_data['pred_depth'],
                                gt_depth=vis_data['gt_depth'],
                                object_mask=object_mask,
                                importance_map=vis_data['importance_map'],
                                fg_mask=vis_data['fg_mask'],
                                bg_mask=vis_data['bg_mask'],
                                sequence_id=num_sequences,
                                class_name=class_name,
                                class_metrics=class_metric,
                                save_dir=save_dir
                            )
                        except Exception as e:
                            logger.warning(f"Failed to save visualization for seq {num_sequences}, class {class_name}: {e}")

            num_sequences += 1
            if max_sequences is not None and num_sequences >= max_sequences:
                break

            # Clear GPU cache
            torch.cuda.empty_cache()

        # Aggregate metrics across all sequences
        logger.info("Aggregating metrics across all sequences...")
        aggregated_metrics = {}
        for model_name, metrics_list in all_metrics.items():
            aggregated_metrics[model_name] = self.metrics_calculator.aggregate_metrics(
                metrics_list
            )

        # Compare models if baseline available
        comparison = None
        if self.baseline_model is not None:
            comparison = self.metrics_calculator.compare_models(
                aggregated_metrics[self.baseline_name],
                aggregated_metrics[self.model_name],
                model_a_name=self.baseline_name,
                model_b_name=self.model_name
            )

        return {
            'per_model_metrics': aggregated_metrics,
            'comparison': comparison,
            'num_sequences': num_sequences
        }


def create_dataloader(
    dataset_type: str,
    data_root: Path,
    batch_size: int = 1,
    video_length: int = 5,
    resolution: str = 'base'
):
    """
    Create dataloader for specified dataset.

    Args:
        dataset_type: Dataset type (accepts both 'waymo'/'waymo_seg')
        data_root: Root directory of dataset
        batch_size: Batch size
        video_length: Number of frames in sequence
        resolution: Resolution mode ('base' for 518 or '2k' for higher res)

    Returns:
        PyTorch DataLoader
    """
    # Normalize dataset type: waymo_seg -> waymo
    dataset_type_normalized = dataset_type.replace('_seg', '')

    # Convert resolution string to numeric value
    # Match CombinedDataset behavior for val/test split
    if resolution == 'base':
        # For base resolution, use dataset-specific resolutions (val/test behavior)
        res_map = {
            'waymo': (784, 518),
            'kitti': (784, 518)
        }
        res_value = res_map.get(dataset_type_normalized, 518)
    elif resolution == '2k':
        # For 2k, use dataset-specific high resolutions
        res_map = {
            'waymo': (1918, 1274),
            'kitti': (1918, 554)
        }
        res_value = res_map.get(dataset_type_normalized, 518)
    else:
        res_value = 518
    if dataset_type_normalized == 'kitti':
        from dataloaders.kitti_segmentation_dataset import KITTISegmentationDataset, collate_fn

        dataset = KITTISegmentationDataset(
            data_root=str(data_root),
            split='val',
            video_length=video_length,
            resolution=res_value
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
            pin_memory=True
        )

        logger.info(f"Created KITTI dataloader with {len(dataset)} sequences")
        return dataloader

    elif dataset_type_normalized == 'waymo':
        from dataloaders.waymo_segmentation_dataset import WaymoSegmentationDataset, collate_fn

        # Use sparse depth from preprocessed waymo dataset
        if 'waymo_seg' in str(data_root):
            # Point to preprocessed depth: waymo_seg -> waymo/val
            depth_root = str(data_root).replace('waymo_seg', 'waymo') + '/val'
        else:
            # Assume waymo_seg is at /home/cvlab/hsy/Datasets/waymo_seg
            depth_root = '/home/cvlab/hsy/Datasets/waymo/val'

        dataset = WaymoSegmentationDataset(
            data_root=str(data_root),
            split='val',  # Waymo uses 'val' not 'validation'
            video_length=video_length,
            resolution=res_value,
            use_depth=True,  # Load sparse depth from preprocessed dataset
            depth_root=depth_root
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
            pin_memory=True
        )

        logger.info(f"Created Waymo dataloader with {len(dataset)} sequences")
        return dataloader

    elif dataset_type_normalized in ['cityscapes', 'nyu', 'vkitti2']:
        # TODO: Implement other dataset loaders
        logger.warning(f"Dataset loader for {dataset_type_normalized} not yet implemented!")
        logger.info(f"Please implement dataset loader in dataloaders/{dataset_type_normalized}_dataset.py")
        logger.info("Dataset should return: images (B,T,C,H,W), depth (B,T,H,W), segmentation (B,H,W)")
        return None

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type} (normalized: {dataset_type_normalized})")


def main():
    parser = argparse.ArgumentParser(description='Object-wise depth evaluation')

    # Model arguments
    parser.add_argument('--model-checkpoint', type=Path, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--baseline-checkpoint', type=Path, default=None,
                        help='Path to baseline checkpoint for comparison')
    parser.add_argument('--config-path', type=Path, required=True,
                        help='Path to model config directory')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['kitti', 'cityscapes', 'nyu', 'vkitti2', 'waymo', 'waymo_seg'],
                        help='Dataset type (use *_seg variants for segmentation datasets)')
    parser.add_argument('--data-root', type=Path, required=True,
                        help='Root directory of dataset')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size (default: 1)')
    parser.add_argument('--video-length', type=int, default=5,
                        help='Video sequence length (default: 5)')
    parser.add_argument('--max-sequences', type=int, default=None,
                        help='Maximum sequences to evaluate (default: all)')
    parser.add_argument('--resolution', type=str, default='base',
                        choices=['base', '2k'],
                        help='Resolution mode: base (518x518) or 2k (1918x1078) (default: base)')

    # Output arguments
    parser.add_argument('--results-dir', type=Path, required=True,
                        help='Directory to save results')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (default: 0)')
    parser.add_argument('--num-vis-sequences', type=int, default=5,
                        help='Number of sequences to visualize (default: 5)')

    args = parser.parse_args()

    # Set device
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Create results directory
    args.results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualizations subdirectory
    vis_dir = args.results_dir / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Initialize evaluator
    evaluator = ObjectWiseEvaluator(
        model_checkpoint=args.model_checkpoint,
        config_path=args.config_path,
        dataset_type=args.dataset,
        device=device,
        baseline_checkpoint=args.baseline_checkpoint
    )

    # Create dataloader
    dataloader = create_dataloader(
        dataset_type=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        video_length=args.video_length,
        resolution=args.resolution
    )

    if dataloader is None:
        logger.error("Dataloader not implemented yet!")
        logger.info("Please implement dataset-specific loader before running evaluation.")
        sys.exit(1)

    # Run evaluation with visualization
    logger.info("Starting evaluation...")
    logger.info(f"Saving visualizations for first {args.num_vis_sequences} sequences to {vis_dir}")
    results = evaluator.evaluate_dataset(
        dataloader=dataloader,
        max_sequences=args.max_sequences,
        save_dir=vis_dir,
        num_vis_sequences=args.num_vis_sequences
    )

    # Print summary
    for model_name, metrics in results['per_model_metrics'].items():
        logger.info(f"\n{model_name} Results:")
        evaluator.metrics_calculator.print_summary(metrics)

    if results['comparison'] is not None:
        logger.info("\nModel Comparison:")
        evaluator.metrics_calculator.print_summary(
            results['per_model_metrics'][evaluator.model_name],
            comparison=results['comparison']
        )

    # Save results
    output_file = args.results_dir / f"{args.dataset}_object_wise_results.json"
    evaluator.metrics_calculator.save_results(
        results['per_model_metrics'][evaluator.model_name],
        output_file,
        comparison=results['comparison']
    )

    logger.info(f"\nEvaluation complete! Results saved to {output_file}")
    logger.info(f"Visualizations saved to {vis_dir}")


if __name__ == "__main__":
    main()
