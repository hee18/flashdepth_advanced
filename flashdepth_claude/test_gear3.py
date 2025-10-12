#!/usr/bin/env python3
"""
Test script for Gear3: Feature-level Metric Depth Learning

Key differences from baseline:
    - No relative depth visualization (features are modulated)
    - Test on inverse depth metrics
    - De-canonicalize outputs for final evaluation
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import logging
import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from PIL import Image
from einops import rearrange
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from flashdepth.model import FlashDepth
from flashdepth.gear3_modules import Gear3MetricHead
from dataloaders.combined_dataset import CombinedDataset
from utils.metric_depth_metrics import MetricDepthMetrics, format_metrics
from utils.helpers import save_gifs_as_grid, save_grid_to_mp4, depth_to_np_arr, torch_batch_to_np_arr

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Gear3Tester:
    """
    Test harness for Gear3 model.

    Evaluates on:
        - Inverse depth metrics (TAE, AbsRel, δ1/δ2/δ3)
        - Metric depth visualization (no relative depth)
        - Importance map visualization
    """
    def __init__(self, config):
        self.config = config
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Setup save directory
        self.save_dir = Path(config.eval.outfolder)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Save directory: {self.save_dir}")

        # Initialize model
        self.model = self._setup_model()

        # Setup test loader
        self.test_loader = self._setup_test_loader()

        # Setup metrics
        self.metrics = MetricDepthMetrics()

    def _setup_model(self):
        """Load trained Gear3 model"""
        # Create base FlashDepth model
        model_config = dict(self.config.model)
        model_config['batch_size'] = 1
        model_config['use_metric_head'] = False

        model = FlashDepth(**model_config)

        # Add Gear3 metric head
        embed_dim = 1024 if model.encoder == 'vitl' else 384
        dpt_dim = 256 if model.encoder == 'vitl' else 64
        num_heads = 16 if model.encoder == 'vitl' else 6

        model.gear3_head = Gear3MetricHead(
            embed_dim=embed_dim,
            dpt_dim=dpt_dim,
            num_heads=num_heads,
            num_dpt_layers=4
        )

        # Load checkpoint
        checkpoint_path = self.config.get('load')
        if checkpoint_path and checkpoint_path != 'true':
            if os.path.exists(checkpoint_path):
                logger.info(f"Loading checkpoint from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location='cpu')

                # Extract state dict
                if isinstance(checkpoint, dict) and 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

                # Load state dict
                model.load_state_dict(state_dict, strict=True)
                logger.info(f"Loaded checkpoint successfully")

                # Log training info if available
                if 'global_step' in checkpoint:
                    logger.info(f"Checkpoint step: {checkpoint['global_step']}")
                if 'phase' in checkpoint:
                    logger.info(f"Training phase: {checkpoint['phase']}")
            else:
                logger.warning(f"Checkpoint {checkpoint_path} not found")
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        else:
            logger.warning("No checkpoint specified!")

        model = model.to(self.device)
        model.eval()

        return model

    def _setup_test_loader(self):
        """Setup test data loader"""
        test_datasets = self.config.eval.test_datasets

        logger.info(f"Test datasets: {test_datasets}")

        test_dataset = CombinedDataset(
            root_dir=self.config.dataset.data_root,
            enable_dataset_flags=test_datasets,
            resolution=self.config.eval.test_dataset_resolution,
            split='test',
            video_length=50  # Longer sequences for testing
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,  # Single process for testing
            pin_memory=True
        )

        logger.info(f"Test dataset size: {len(test_dataset)}")

        return test_loader

    @torch.no_grad()
    def test(self):
        """Main testing loop"""
        logger.info("Starting testing...")

        all_metrics = []
        sequence_id = 0

        for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Testing")):
            try:
                metrics = self.test_sequence(batch, sequence_id)
                all_metrics.append(metrics)
                sequence_id += 1

            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Aggregate metrics
        if all_metrics:
            avg_metrics = self._aggregate_metrics(all_metrics)
            logger.info("\n" + "="*80)
            logger.info("FINAL RESULTS")
            logger.info("="*80)
            logger.info(format_metrics(avg_metrics))

            # Save results
            results_path = self.save_dir / "test_results.json"
            with open(results_path, 'w') as f:
                json.dump(avg_metrics, f, indent=2)
            logger.info(f"Results saved to {results_path}")

        else:
            logger.warning("No metrics computed!")

    @torch.no_grad()
    def test_sequence(self, batch, sequence_id):
        """Test on a single sequence"""
        images = batch['image'].to(self.device)  # [1, T, 3, H, W]
        gt_depth = batch['depth'].to(self.device)  # [1, T, 1, H, W]

        B, T = images.shape[:2]
        assert B == 1, "Batch size must be 1 for testing"

        # Dataloader gives inverse depth (1/m), scale to 100/m for training
        gt_depth_inverse_100 = gt_depth * 100.0  # [1, T, 1, H, W] in 100/m

        # Storage for predictions
        pred_depths = []
        importance_maps = []

        # Process each frame
        for t in range(T):
            img_t = images[0, t]  # [3, H, W]
            gt_t_inverse = gt_depth_inverse_100[0, t]  # [1, H, W] in 100/m

            # Extract features
            features = self.model.pretrained(img_t.unsqueeze(0), is_training=False)
            encoder_features = [features[idx] for idx in self.model.intermediate_layer_idx[self.model.encoder]]
            patch_tokens = features[-1][:, 1:]  # [1, num_patches, embed_dim]
            attention_weights = features[-2]  # [1, num_heads, num_patches+1, num_patches+1]

            # Get DPT features
            h, w = img_t.shape[1:]
            patch_h, patch_w = h // self.model.patch_size, w // self.model.patch_size
            dpt_features = self.model.depth_head.get_forward_features(encoder_features, patch_h, patch_w)

            # Apply Gear3 modulation
            modulated_dpt_features, importance_map = self.model.gear3_head(
                patch_tokens, attention_weights, dpt_features, patch_h, patch_w
            )

            # Get depth prediction (output is inverse depth in 100/m scale)
            path_1_modulated = modulated_dpt_features[-1]
            out = self.model.depth_head.scratch.output_conv1(path_1_modulated)
            out = F.interpolate(out, (h, w), mode="bilinear", align_corners=True)
            out = self.model.depth_head.scratch.output_conv2(out)  # [1, 1, H, W]

            # Prediction is already positive (Softplus activation in output_conv2)
            pred_depth_inverse_100 = out[0]  # [1, H, W] in 100/m

            # Convert to metric depth: 100/m -> m
            pred_depth_metric = 100.0 / (pred_depth_inverse_100 + 1e-8)

            pred_depths.append(pred_depth_metric)
            importance_maps.append(importance_map[0])

        # Stack predictions
        pred_depths = torch.stack(pred_depths, dim=0)  # [T, 1, H, W] in meters
        importance_maps = torch.stack(importance_maps, dim=0)  # [T, 1, patch_h, patch_w]

        # Convert GT to metric depth for evaluation: 100/m -> m
        gt_depth_metric = 100.0 / (gt_depth_inverse_100[0] + 1e-8)  # [T, 1, H, W] in meters

        # Compute metrics (both pred and GT are now in meters)
        valid_mask = (gt_depth_metric > 0).float()
        metrics = self.metrics.compute_metrics(
            pred_depths.squeeze(1),  # [T, H, W] in meters
            gt_depth_metric.squeeze(1),  # [T, H, W] in meters
            valid_mask.squeeze(1)  # [T, H, W]
        )

        # Visualize
        if self.config.eval.get('save_grid', True):
            self._visualize_sequence(
                images[0], pred_depths, gt_depth_metric, importance_maps,
                valid_mask, sequence_id, metrics
            )

        # Save video (GIF or MP4)
        if self.config.eval.get('out_video', True):
            self._save_video(
                images[0], pred_depths, gt_depth_metric, valid_mask, sequence_id
            )

        return metrics

    def _visualize_sequence(self, images, pred_depths, gt_depths, importance_maps,
                           valid_mask, sequence_id, metrics):
        """
        Create visualization grid for a sequence.

        Rows: Image, Metric Depth (Prediction), Metric Depth (GT), Importance Map
        """
        T = images.shape[0]
        frames_to_show = min(10, T)  # Show up to 10 frames
        interval = max(1, T // frames_to_show)
        frame_indices = list(range(0, T, interval))[:frames_to_show]

        # Create figure
        fig, axes = plt.subplots(4, frames_to_show, figsize=(frames_to_show * 3, 12))
        if frames_to_show == 1:
            axes = axes.reshape(-1, 1)

        for col, t in enumerate(frame_indices):
            # Row 0: Image
            img = images[t].permute(1, 2, 0).cpu().numpy()
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            axes[0, col].imshow(img)
            axes[0, col].set_title(f'Frame {t}')
            axes[0, col].axis('off')

            # Row 1: Predicted metric depth
            pred = pred_depths[t, 0].cpu().numpy()
            pred_valid = (pred > 0) & (pred < 1000)
            pred_display = np.full_like(pred, np.nan)
            if pred_valid.sum() > 0:
                pred_vmin, pred_vmax = np.nanpercentile(pred[pred_valid], [2, 98])
                pred_display[pred_valid] = pred[pred_valid]
            else:
                pred_vmin, pred_vmax = 0, 1
            axes[1, col].imshow(pred_display, cmap='plasma', vmin=pred_vmin, vmax=pred_vmax)
            axes[1, col].set_title(f'Pred (m)')
            axes[1, col].axis('off')

            # Row 2: GT metric depth
            gt = gt_depths[t, 0].cpu().numpy()
            gt_valid = valid_mask[t, 0].cpu().numpy().astype(bool)
            gt_display = np.full_like(gt, np.nan)
            if gt_valid.sum() > 0:
                gt_vmin, gt_vmax = np.nanpercentile(gt[gt_valid], [2, 98])
                gt_display[gt_valid] = gt[gt_valid]
            else:
                gt_vmin, gt_vmax = 0, 1
            axes[2, col].imshow(gt_display, cmap='plasma', vmin=gt_vmin, vmax=gt_vmax)
            axes[2, col].set_title(f'GT (m)')
            axes[2, col].axis('off')

            # Row 3: Importance map (matches train_gear3.py visualization)
            importance = importance_maps[t, 0].cpu().numpy()
            axes[3, col].imshow(importance, cmap='jet', vmin=0, vmax=1)
            axes[3, col].set_title(f'Importance')
            axes[3, col].axis('off')

        # Add overall title with metrics
        fig.suptitle(
            f"Sequence {sequence_id} | "
            f"TAE: {metrics.get('tae', 0):.4f} | "
            f"AbsRel: {metrics.get('abs_rel', 0):.4f} | "
            f"δ1: {metrics.get('delta_1', 0):.4f}",
            fontsize=14
        )

        plt.tight_layout()
        save_path = self.save_dir / f"sequence_{sequence_id:04d}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Saved visualization: {save_path}")

    def _save_video(self, images, pred_depths, gt_depths, valid_mask, sequence_id):
        """
        Create video files (GIF/MP4) similar to FlashDepth validation.

        Args:
            images: [T, 3, H, W] - RGB images
            pred_depths: [T, 1, H, W] - Predicted metric depth
            gt_depths: [T, 1, H, W] - GT metric depth
            valid_mask: [T, 1, H, W] - Valid mask
            sequence_id: int - Sequence index
        """
        T = images.shape[0]

        # Convert to numpy arrays for video creation
        video_frames = torch_batch_to_np_arr(images)  # [T, H, W, 3]

        # Convert depth to colorized numpy arrays
        pred_frames = depth_to_np_arr(pred_depths.squeeze(1))  # [T, H, W, 3]
        gt_frames = depth_to_np_arr(gt_depths.squeeze(1))  # [T, H, W, 3]

        # Generate video paths
        base_name = f"sequence_{sequence_id:04d}"
        gif_path = self.save_dir / f"{base_name}.gif"
        mp4_path = self.save_dir / f"{base_name}.mp4"

        # Save based on config
        if self.config.eval.get('out_mp4', False):
            # Save as MP4 (with separate pred-only video)
            logger.info(f"Saving MP4 videos for sequence {sequence_id}...")
            grid = save_grid_to_mp4(
                video_frames,
                gt_frames,
                pred_frames,
                output_path=str(mp4_path),
                fixed_height=self.config.eval.get('save_res', 256),
                fps=self.config.eval.get('video_fps', 10)
            )
            logger.info(f"Saved: {mp4_path}")
            logger.info(f"Saved: {grid['pred_video_path']}")
        else:
            # Save as GIF (default)
            logger.info(f"Saving GIF for sequence {sequence_id}...")
            grid = save_gifs_as_grid(
                video_frames,
                gt_frames,
                pred_frames,
                output_path=str(gif_path),
                fixed_height=self.config.eval.get('save_res', 256),
                duration=self.config.eval.get('gif_duration', 110)
            )
            logger.info(f"Saved: {gif_path}")

        return grid

    def _aggregate_metrics(self, all_metrics):
        """Aggregate metrics across sequences"""
        metric_keys = all_metrics[0].keys()
        aggregated = {}

        for key in metric_keys:
            values = [m[key] for m in all_metrics if key in m]
            if values:
                aggregated[key] = np.mean(values)

        return aggregated


@hydra.main(version_base=None, config_path="configs/gear3", config_name="config")
def main(config: DictConfig):
    """Main entry point"""
    import os

    # Override config for testing
    config.inference = True

    tester = Gear3Tester(config)
    tester.test()


if __name__ == "__main__":
    main()
