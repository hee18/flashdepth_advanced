#!/usr/bin/env python3
"""
Test script for Onepiece V3: Spatial Mamba + Dual-Stream Metric Depth Estimation

Tests on: sintel, waymo_seg, eth3d, urbansyn, unreal4k, bonn
Metrics: MAE, RMSE, AbsRel, d1/d2/d3, TAE (reprojection), FPS
Visualizations: depth sequence, error heatmaps, best/worst frames, video/gif
JSON outputs: test_results, per_sequence_results, best/worst_sequence,
              scale_shift_comparison, depth_range_analysis, temporal_analysis

Sea-RAFT NOT required for testing (flow only used in training loss).
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import logging
import json
import time
from pathlib import Path
from einops import rearrange
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from flashdepth.model import FlashDepth
from dataloaders.combined_dataset import CombinedDataset
from utils.metric_depth_metrics import MetricDepthMetrics, format_metrics
from utils.helpers import save_gifs_as_grid, save_grid_to_mp4, depth_to_np_arr, torch_batch_to_np_arr
from utils.reprojection_tae import ReprojectionTAECalculator
from utils.temporal_consistency import FlowTemporalConsistency

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_canonical_focal_length(config):
    """Get canonical focal length (fixed at 500.0 for all resolutions)."""
    return 500.0


class OnepieceTester:
    """
    Test harness for Onepiece metric depth model.
    Full evaluation with Gear5-equivalent feature set (no importance map / FG / BG).
    """

    def __init__(self, config):
        self.config = config
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Max depth for evaluation
        self.max_depth = config.get('max_depth', 80.0)

        # Save directory
        save_dir_str = config.get('results_dir', config.eval.outfolder)
        self.save_dir = Path(save_dir_str)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Save directory: {self.save_dir}")

        # Frame interval for visualization
        self.frame_interval = config.get('frame_interval', 1)

        # Figure export options (same as test_gear5.py)
        self.export_best_figure = config.get('best_figure', False)
        # Specific frame index(es) - supports single int or list of ints
        raw_frame = config.get('frame', None)
        if raw_frame is not None:
            if isinstance(raw_frame, (list, ListConfig)):
                self.export_frames = [int(f) for f in raw_frame]
            else:
                self.export_frames = [int(raw_frame)]
        else:
            self.export_frames = None
        if self.export_best_figure:
            logger.info(f"Best-figure export ENABLED (will save best_frame ±4 intervals as individual images, frame_interval={self.frame_interval or 1})")
        if self.export_frames is not None:
            logger.info(f"Frame-specific export ENABLED (will save frames {self.export_frames} ±4 intervals as individual images, frame_interval={self.frame_interval or 1})")

        # Enable visualization
        self.enable_visualization = True

        # Initialize model
        self.model = self._setup_model()

        # Setup test loader
        self.test_loader = self._setup_test_loader()

        # Setup metrics
        self.metrics = MetricDepthMetrics()

        # Setup reprojection TAE calculator
        data_root = config.dataset.get('data_root', '/home/cvlab/hsy/Datasets')
        self.reproj_tae_calculator = ReprojectionTAECalculator(data_root)
        logger.info(f"Reprojection TAE calculator initialized (supported: {self.reproj_tae_calculator.SUPPORTED_DATASETS})")

        # Flow-based temporal consistency (lazy-loaded)
        self.flow_tc = None
        self.tc_threshold = config.get('tc_threshold', 1.25)

        # Test mode: None (full), 'tc' (temporal consistency only), 'ea' (error & accuracy only)
        self.test_mode = config.get('test_mode', None)

        # Save depth maps option
        self.save_depth_maps = config.get('save_depth_maps', False)
        if self.save_depth_maps:
            logger.info("Depth map saving ENABLED (will save pred depth as .npy)")

    @staticmethod
    def _compute_excl_scd_metrics(tc_result, reset_frames, T):
        """
        Compute rTC and flickering metrics excluding SCD reset frame pairs.

        For a reset at frame t, pair indices t-1 (frames t-1→t) and t (frames t→t+1) are excluded.
        Returns dict with excl_scd versions of rTC, per_frame_rtc, and multi_threshold.
        """
        if not reset_frames or T < 2:
            return None

        # Build set of pair indices to exclude
        excl_pairs = set()
        for t in reset_frames:
            if t > 0:
                excl_pairs.add(t - 1)  # pair (t-1, t)
            if t < T - 1:
                excl_pairs.add(t)      # pair (t, t+1)

        per_frame_rtc = tc_result['per_frame_rtc']
        per_frame_rtc_gt = tc_result['per_frame_rtc_gt']

        if not excl_pairs or len(excl_pairs) >= len(per_frame_rtc):
            return None

        # Filter per_frame_rtc
        filtered_rtc = [v for i, v in enumerate(per_frame_rtc) if i not in excl_pairs]
        filtered_rtc_gt = [v for i, v in enumerate(per_frame_rtc_gt) if i not in excl_pairs]

        result = {
            'rtc_excl_scd': float(np.mean(filtered_rtc)) if filtered_rtc else 0.0,
            'rtc_gt_excl_scd': float(np.mean(filtered_rtc_gt)) if filtered_rtc_gt else 0.0,
            '_per_frame_rtc_excl_scd': filtered_rtc,
            '_per_frame_rtc_gt_excl_scd': filtered_rtc_gt,
            '_excl_scd_pair_indices': sorted(excl_pairs),
            '_excl_scd_num_excluded': len(excl_pairs),
            '_excl_scd_num_remaining': len(filtered_rtc),
        }

        # Filter multi_threshold
        mt = tc_result.get('multi_threshold', {})
        if mt:
            mt_excl = {}
            for thr_key, thr_data in mt.items():
                if thr_key.startswith('_'):
                    continue
                pf_rtc = thr_data.get('per_frame_rtc', [])
                filtered_pf = [v for i, v in enumerate(pf_rtc) if i not in excl_pairs]
                total_pairs = len(filtered_pf)
                mean_rtc = float(np.mean(filtered_pf)) if filtered_pf else 0.0

                flickering = {}
                for cutoff_key, cutoff_data in thr_data.get('flickering', {}).items():
                    # Re-detect flickering on filtered pairs only
                    flicker_frames = [i for i, rtc in enumerate(filtered_pf) if rtc < float(cutoff_key)]
                    flickering[cutoff_key] = {
                        'count': len(flicker_frames),
                        'rate': len(flicker_frames) / max(total_pairs, 1),
                        'frames': flicker_frames,
                    }

                mt_excl[thr_key] = {
                    'rtc': mean_rtc,
                    'per_frame_rtc': [float(x) for x in filtered_pf],
                    'total_pairs': total_pairs,
                    'flickering': flickering,
                }
            result['_multi_threshold_excl_scd'] = mt_excl

        return result

    def _setup_model(self):
        """Load trained Onepiece V3 model."""
        model_config = dict(self.config.model)
        model_config['batch_size'] = 1
        model_config['use_metric_head'] = False
        model_config['use_onepiece'] = True

        # Spatial Mamba config
        model_config['spatial_mamba_layers'] = self.config.model.get('spatial_mamba_layers', 4)
        model_config['spatial_mamba_d_state'] = self.config.model.get('spatial_mamba_d_state', 256)
        model_config['spatial_mamba_d_conv'] = self.config.model.get('spatial_mamba_d_conv', 4)
        model_config['spatial_mamba_downsample'] = self.config.model.get('spatial_mamba_downsample', 0.1)
        model_config['onepiece_train_mode'] = self.config.get('train_mode', 'metric')

        scene_cut_config = self.config.get('scene_cut', {})
        if self.config.get('no_scd', False):
            model_config['scene_cut_tau'] = float('inf')
        else:
            model_config['scene_cut_tau'] = scene_cut_config.get('tau', 0.05)
        model_config['scene_cut_k'] = scene_cut_config.get('k', 80)

        # Hybrid configs (top-level key, not under model)
        model_config['hybrid_configs'] = self.config.get('hybrid_configs', None)

        model = FlashDepth(**model_config)

        checkpoint_path = self.config.get('load')
        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                logger.warning(f"Missing keys: {missing[:10]}...")
            if unexpected:
                logger.warning(f"Unexpected keys: {unexpected[:10]}...")
            logger.info(f"Loaded checkpoint successfully")

            if isinstance(checkpoint, dict):
                if 'global_step' in checkpoint:
                    logger.info(f"  Checkpoint step: {checkpoint['global_step']}")
                if 'current_phase' in checkpoint:
                    logger.info(f"  Training phase: {checkpoint['current_phase']}")
                if 'best_val_loss' in checkpoint:
                    logger.info(f"  Best val loss: {checkpoint['best_val_loss']:.6f}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        model = model.to(self.device)
        model.eval()
        return model

    def _setup_test_loader(self):
        """Setup test data loader."""
        test_datasets = self.config.eval.get('test_datasets',
            ['sintel', 'waymo_seg', 'eth3d', 'urbansyn', 'unreal4k', 'bonn'])
        resolution = self.config.get('resolution',
            self.config.eval.get('test_dataset_resolution', 'base'))
        video_length = int(self.config.dataset.get('video_length', 50))

        if isinstance(test_datasets, (ListConfig, str)):
            if isinstance(test_datasets, str):
                test_datasets = [test_datasets]
            else:
                test_datasets = list(test_datasets)

        logger.info(f"Test datasets: {test_datasets}")
        logger.info(f"Resolution: {resolution}, video_length: {video_length}")

        seq_list = self.config.get('seq_list', None)
        if seq_list is not None:
            logger.info(f"Filtering to sequences: {seq_list}")

        test_dataset = CombinedDataset(
            root_dir=self.config.dataset.data_root,
            enable_dataset_flags=test_datasets,
            resolution=resolution,
            split='test',
            video_length=video_length,
            seq_list=seq_list,
            skip_gt_canonicalization=True
        )

        num_workers = self.config.training.get('workers', 0)
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self._collate_fn,
        )

        logger.info(f"Test dataset size: {len(test_dataset)}")
        return test_loader

    def _collate_fn(self, batch):
        """Custom collate function to filter out None values and convert tuple to dict."""
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None

        # CombinedDataset returns tuple for val/test splits - convert to dict
        if len(batch) > 0 and isinstance(batch[0], tuple):
            if len(batch[0]) == 9:
                images, depths, focal_lengths_canonical, focal_lengths_actual, actual_valid_masks, fx_ratios, resize_ratios, names, image_paths = zip(*batch)
                return {
                    'image': torch.stack(images, dim=0),
                    'depth': torch.stack(depths, dim=0),
                    'focal_lengths': torch.stack(focal_lengths_canonical, dim=0),
                    'focal_lengths_actual': torch.stack(focal_lengths_actual, dim=0),
                    'actual_valid_mask': torch.stack(actual_valid_masks, dim=0),
                    'fx_ratio': torch.stack(fx_ratios, dim=0),
                    'resize_ratio': torch.stack(resize_ratios, dim=0),
                    'dataset_name': names,
                    'image_paths': image_paths
                }
            elif len(batch[0]) == 8:
                images, depths, focal_lengths_canonical, focal_lengths_actual, actual_valid_masks, fx_ratios, resize_ratios, names = zip(*batch)
                return {
                    'image': torch.stack(images, dim=0),
                    'depth': torch.stack(depths, dim=0),
                    'focal_lengths': torch.stack(focal_lengths_canonical, dim=0),
                    'focal_lengths_actual': torch.stack(focal_lengths_actual, dim=0),
                    'actual_valid_mask': torch.stack(actual_valid_masks, dim=0),
                    'fx_ratio': torch.stack(fx_ratios, dim=0),
                    'resize_ratio': torch.stack(resize_ratios, dim=0),
                    'dataset_name': names
                }
            elif len(batch[0]) == 5:
                images, depths, focal_lengths, actual_valid_masks, names = zip(*batch)
                return {
                    'image': torch.stack(images, dim=0),
                    'depth': torch.stack(depths, dim=0),
                    'focal_lengths': torch.stack(focal_lengths, dim=0),
                    'actual_valid_mask': torch.stack(actual_valid_masks, dim=0),
                    'dataset_name': names
                }
            else:
                images, depths, focal_lengths, names = zip(*batch)
                return {
                    'image': torch.stack(images, dim=0),
                    'depth': torch.stack(depths, dim=0),
                    'focal_lengths': torch.stack(focal_lengths, dim=0),
                    'dataset_name': names
                }

        # Dict format (segmentation datasets)
        if len(batch) > 0 and isinstance(batch[0], dict):
            result = batch[0]
            if 'images' in result:
                result['image'] = result.pop('images')
            if 'focal_lengths_actual' in result and 'focal_lengths' not in result:
                result['focal_lengths'] = result['focal_lengths_actual']
            return result

        return batch

    @torch.no_grad()
    def test(self):
        """Main testing loop with full JSON output."""
        logger.info("Starting Onepiece testing...")

        all_metrics = []
        sequence_id = 0

        for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Testing")):
            try:
                metrics = self.test_sequence(batch, sequence_id)
                metrics['sequence_id'] = sequence_id
                all_metrics.append(metrics)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                sequence_id += 1

            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

        if not all_metrics:
            logger.warning("No valid sequences processed!")
            return

        # === Aggregate and save results ===
        avg_metrics = self._aggregate_metrics(all_metrics)

        logger.info("\n" + "=" * 80)
        logger.info("FINAL RESULTS")
        logger.info("=" * 80)
        for k, v in avg_metrics.items():
            if isinstance(v, float):
                logger.info(f"  {k}: {v:.4f}")

        # TC-only mode: save temporal_consistency.json + tc_summary.json and return
        if self.test_mode == 'tc':
            self._save_temporal_consistency(all_metrics)
            self._save_tc_summary(all_metrics)
            # Multi-threshold rTC + flickering count analysis
            from utils.temporal_consistency import FlowTemporalConsistency
            FlowTemporalConsistency.save_multi_threshold_json(all_metrics, self.save_dir)
            logger.info("TC-only mode: saved temporal_consistency.json, tc_summary.json, multi_threshold_rtc.json")
            return

        # 1. test_results.json (aggregated)
        metric_order = ['abs_rel', 'a1', 'a2', 'a3', 'fps', 'tae', 'tae_reproj', 'tae_reproj_gt',
                        'rtc', 'rtc_gt', 'rtc_excl_scd', 'rtc_gt_excl_scd',
                        'psr', 'psr_max',
                        'mae', 'rmse',
                        'pred_scale_mean', 'pred_shift_mean',
                        'optimal_scale_mean', 'optimal_shift_mean']
        ordered_results = {}
        for key in metric_order:
            if key in avg_metrics:
                ordered_results[key] = float(avg_metrics[key]) if isinstance(avg_metrics[key], (float, np.floating)) else avg_metrics[key]
        for key, value in avg_metrics.items():
            if key not in ordered_results:
                ordered_results[key] = float(value) if isinstance(value, (float, np.floating)) else value

        with open(self.save_dir / "test_results.json", 'w') as f:
            json.dump(ordered_results, f, indent=2, default=str)
        logger.info(f"Results saved to {self.save_dir / 'test_results.json'}")

        # 2. per_sequence_results.json
        per_seq_data = []
        for r in all_metrics:
            entry = {}
            for key in metric_order:
                if key in r:
                    entry[key] = float(r[key]) if isinstance(r[key], (float, np.floating)) else r[key]
            for key, value in r.items():
                if key not in entry and not key.startswith('_') and not isinstance(value, (dict, list)):
                    entry[key] = float(value) if isinstance(value, (float, np.floating)) else value
            # Include reset_frames list (SCD reset info)
            if 'reset_frames' in r:
                entry['reset_frames'] = r['reset_frames']
                entry['num_resets'] = r.get('num_resets', len(r['reset_frames']))
            per_seq_data.append(entry)

        with open(self.save_dir / "per_sequence_results.json", 'w') as f:
            json.dump(per_seq_data, f, indent=2, default=str)
        logger.info(f"Per-sequence results saved to {self.save_dir / 'per_sequence_results.json'}")

        # 3. best_sequence.json / worst_sequence.json
        valid_metrics = [m for m in all_metrics if m.get('abs_rel', float('inf')) != float('inf')]
        if valid_metrics:
            best = min(valid_metrics, key=lambda x: x.get('abs_rel', float('inf')))
            worst = max(valid_metrics, key=lambda x: x.get('abs_rel', 0))

            best_entry = {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                         for k, v in best.items() if not k.startswith('_') and not isinstance(v, (dict, list))}
            worst_entry = {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                          for k, v in worst.items() if not k.startswith('_') and not isinstance(v, (dict, list))}

            with open(self.save_dir / "best_sequence.json", 'w') as f:
                json.dump(best_entry, f, indent=2, default=str)
            with open(self.save_dir / "worst_sequence.json", 'w') as f:
                json.dump(worst_entry, f, indent=2, default=str)

            logger.info(f"Best sequence: #{best.get('sequence_id', -1)}, AbsRel={best.get('abs_rel', 0):.4f}")
            logger.info(f"Worst sequence: #{worst.get('sequence_id', -1)}, AbsRel={worst.get('abs_rel', 0):.4f}")

        # 4. scale_shift_comparison.json
        self._save_scale_shift_comparison(all_metrics)

        # 5. depth_range_analysis.json
        self._save_depth_range_analysis(all_metrics)

        # 6. temporal_analysis.json
        self._save_temporal_analysis(all_metrics)

        # 7. temporal_consistency.json (flow-based rTC)
        self._save_temporal_consistency(all_metrics)

        # 8. multi_threshold_rtc.json (multi-threshold rTC + flickering counts)
        from utils.temporal_consistency import FlowTemporalConsistency
        FlowTemporalConsistency.save_multi_threshold_json(all_metrics, self.save_dir)

    @torch.no_grad()
    def test_sequence(self, batch, sequence_id):
        """Test on a single sequence with full metrics and visualization."""
        # Handle dict batch format from CombinedDataset
        if isinstance(batch, dict):
            if 'images' in batch:
                images = batch['images'].to(self.device)
            else:
                images = batch['image'].to(self.device)
            if images.ndim == 4:
                images = images.unsqueeze(0)

            dataset_name = batch.get('dataset_name', 'unknown')
            if isinstance(dataset_name, (list, tuple)):
                dataset_name = dataset_name[0]
            dataset_name = dataset_name.lower() if isinstance(dataset_name, str) else 'unknown'

            if 'depths' in batch:
                gt_depth = batch['depths']
            else:
                gt_depth = batch['depth']

            # Focal lengths
            if 'focal_lengths_actual' in batch:
                fx_actual_tensor = batch['focal_lengths_actual'].to(self.device)
            elif 'fx_ratio' in batch:
                fx_ratio_tensor = batch['fx_ratio'].to(self.device)
                fx_actual_tensor = 500.0 / fx_ratio_tensor
            else:
                fx_actual_tensor = None

            actual_valid_mask = batch.get('actual_valid_mask', None)
            if actual_valid_mask is not None:
                actual_valid_mask = actual_valid_mask.to(self.device)

            fx_ratio = batch.get('fx_ratio', None)
            resize_ratio = batch.get('resize_ratio', None)
            if fx_ratio is not None:
                fx_ratio = fx_ratio.to(self.device)
            if resize_ratio is not None:
                resize_ratio = resize_ratio.to(self.device)

        elif isinstance(batch, (list, tuple)):
            # Gear5 8-element format
            images_raw, gt_depth_raw, focal_canonical, focal_actual, actual_valid, fx_ratio_raw, resize_ratio_raw, dataset_idx = batch
            images = images_raw.to(self.device)
            gt_depth = gt_depth_raw
            dataset_name = str(dataset_idx[0] if isinstance(dataset_idx, (list, tuple)) else dataset_idx)
            fx_actual_tensor = focal_actual.to(self.device) if focal_actual is not None else None
            actual_valid_mask = actual_valid.to(self.device) if actual_valid is not None else None
            fx_ratio = fx_ratio_raw.to(self.device) if fx_ratio_raw is not None else None
            resize_ratio = resize_ratio_raw.to(self.device) if resize_ratio_raw is not None else None
        else:
            raise TypeError(f"Unexpected batch type: {type(batch)}")

        # Normalize gt_depth shape
        if gt_depth.ndim == 3:
            gt_depth = gt_depth.unsqueeze(0)
        if gt_depth.ndim == 4:
            gt_depth = gt_depth.unsqueeze(2)

        gt_depth = gt_depth.to(self.device)
        B, T = images.shape[:2]
        H, W = images.shape[3], images.shape[4]
        assert B == 1, "Batch size must be 1 for testing"

        logger.info(f"Seq {sequence_id} [{dataset_name}]: {T} frames, {H}x{W}")

        # GT inverse depth -> 100/m
        gt_depth_inverse_100 = gt_depth * 100.0  # [1, T, 1, H, W]

        # De-canonicalization ratios
        # When use_dual_cstm=False (No-CSTM ablation), model was trained without canonical
        # transform → predictions are already in actual space → de-canon = identity
        use_dual_cstm = self.config.get('use_dual_cstm', True)
        if not use_dual_cstm:
            de_canonical_ratio_inverse = torch.ones(1, T, device=self.device)
            de_canonical_ratio_metric = torch.ones(1, T, device=self.device)
        else:
            CANONICAL_FX = get_canonical_focal_length(self.config)
            if fx_ratio is not None and resize_ratio is not None:
                de_canonical_ratio_inverse = fx_ratio / resize_ratio
                de_canonical_ratio_metric = 1.0 / de_canonical_ratio_inverse
            elif fx_actual_tensor is not None:
                de_canonical_ratio_inverse = CANONICAL_FX / fx_actual_tensor
                de_canonical_ratio_metric = 1.0 / de_canonical_ratio_inverse
            else:
                de_canonical_ratio_inverse = torch.ones(1, T, device=self.device)
                de_canonical_ratio_metric = torch.ones(1, T, device=self.device)

        # Valid mask
        if actual_valid_mask is not None:
            canonical_gt_valid = actual_valid_mask.unsqueeze(2)
        else:
            MIN_INVERSE_CANONICAL = 100.0 / self.max_depth
            canonical_gt_valid = (gt_depth_inverse_100 > MIN_INVERSE_CANONICAL)

        # === Per-frame streaming inference ===
        base_dataset_name = dataset_name.split('/')[0] if isinstance(dataset_name, str) else 'unknown'
        if base_dataset_name == 'eth3d':
            warmup_frames = min(5, T)
        else:
            warmup_frames = min(10, T)

        pred_depths_list = []
        rel_depths_list = []
        per_frame_scales_raw = []
        per_frame_shifts_raw = []
        per_frame_d_patch_mean = []
        reset_frames = []
        start_time = None
        prev_patch_mean = None

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # Reset Mamba state for new sequence
            self.model.spatial_mamba.start_new_sequence()

            for t in range(T):
                frame = images[0, t].unsqueeze(0)  # [1, 3, H, W]

                # Start timing after warmup
                if t == warmup_frames:
                    torch.cuda.synchronize()
                    start_time = time.time()

                outputs_t = self.model.forward_onepiece_single_frame(frame, prev_patch_mean=prev_patch_mean)
                prev_patch_mean = outputs_t['patch_mean']

                d_patch_mean_val = outputs_t.get('d_patch_mean', 0.0)
                per_frame_d_patch_mean.append(d_patch_mean_val)

                if outputs_t.get('is_reset', False):
                    reset_frames.append(t)
                    logger.info(f"  SCD reset at frame {t} (d_patch_mean={d_patch_mean_val:.4f}, tau={self.model.scene_cut_detector.tau})")

                # De-canonicalize: canonical → actual meters
                de_ratio_t = de_canonical_ratio_metric[0, t]
                pred_metric_t = outputs_t['metric_depth'].float() * de_ratio_t  # [1, H, W]
                rel_depth_t = outputs_t['relative_depth'].float()[0]  # [H, W]

                per_frame_scales_raw.append(float(outputs_t['scale']))
                per_frame_shifts_raw.append(float(outputs_t['shift']))

                # FPS window: keep on GPU for accurate timing; otherwise CPU
                if t >= warmup_frames:
                    pred_depths_list.append(pred_metric_t)  # GPU
                    rel_depths_list.append(rel_depth_t)  # GPU
                else:
                    pred_depths_list.append(pred_metric_t.cpu())
                    rel_depths_list.append(rel_depth_t.cpu())

                del outputs_t, pred_metric_t, rel_depth_t

        # End timing
        torch.cuda.synchronize()
        end_time = time.time()

        # FPS calculation
        if start_time is not None:
            inference_time = end_time - start_time
            timed_frames = T - warmup_frames
            fps = timed_frames / inference_time if timed_frames > 0 and inference_time > 0 else 0
            if fps > 0:
                logger.info(f"Inference time: {inference_time:.4f}s for {timed_frames} frames (warmup {warmup_frames} excluded)")
                logger.info(f"FPS: {fps:.2f} frames/second")
        else:
            fps = 0
            logger.warning(f"Too few frames ({T}) for FPS measurement (need > {warmup_frames})")

        # Stack results and move to CPU
        # Each pred_metric_t is [1, H, W] (B=1), stack → [T, 1, H, W]
        pred_depths_cpu = torch.stack(
            [p.cpu() if p.is_cuda else p for p in pred_depths_list], dim=0
        )  # [T, 1, pred_h, pred_w]
        dfr_canonical_cpu = 1.0 / (torch.stack(
            [r.cpu() if r.is_cuda else r for r in rel_depths_list], dim=0
        ) + 1e-8)  # [T, pred_h, pred_w]  (depth from relative in canonical meters)
        scale = torch.tensor(per_frame_scales_raw).unsqueeze(0)  # [1, T]
        shift = torch.tensor(per_frame_shifts_raw).unsqueeze(0)  # [1, T]
        del pred_depths_list, rel_depths_list
        torch.cuda.empty_cache()

        # Save predicted depth maps as .npy if enabled
        if self.save_depth_maps:
            depth_map_dir = self.save_dir / "depth_maps" / f"seq{sequence_id:04d}"
            depth_map_dir.mkdir(parents=True, exist_ok=True)
            for t in range(pred_depths_cpu.shape[0]):
                np.save(depth_map_dir / f"pred_{t:04d}.npy", pred_depths_cpu[t, 0].float().numpy())
            logger.info(f"Saved {pred_depths_cpu.shape[0]} depth maps to {depth_map_dir} "
                       f"(resolution: {pred_depths_cpu.shape[-2]}x{pred_depths_cpu.shape[-1]})")

        # GT in actual meters (already actual from skip_gt_canonicalization=True)
        gt_depth_metric = 100.0 / (gt_depth_inverse_100[0] + 1e-8)  # [T, 1, H, W]

        # === Resolution handling (upsample pred to GT resolution for metrics) ===
        MAX_DEPTH = self.max_depth
        pred_h, pred_w = pred_depths_cpu.shape[-2:]
        gt_h, gt_w = gt_depth_metric.shape[-2:]
        need_upsample = (gt_h, gt_w) != (pred_h, pred_w)

        if need_upsample:
            logger.info(f"Resolution mismatch: pred ({pred_h},{pred_w}) vs GT ({gt_h},{gt_w}). "
                        f"Will upsample pred to GT resolution for metrics.")

        # GT at original resolution (for per-frame metrics with upsampled pred)
        gt_depth_metric_cpu = gt_depth_metric.cpu()  # [T, 1, gt_h, gt_w]

        # GT at pred resolution (for TC, visualization, TAE, optimal scale)
        if need_upsample:
            ds_lower = dataset_name.lower() if isinstance(dataset_name, str) else 'unknown'
            is_sparse_ds = any(s in ds_lower for s in ['eth3d', 'waymo_seg'])
            interp_mode = 'nearest' if is_sparse_ds else 'bilinear'
            interp_kwargs = {} if is_sparse_ds else {'align_corners': False}
            gt_at_pred_res = F.interpolate(
                gt_depth_metric, size=(pred_h, pred_w), mode=interp_mode, **interp_kwargs
            )
            gt_at_pred_res_cpu = gt_at_pred_res.cpu()  # [T, 1, pred_h, pred_w]
            del gt_at_pred_res
            # Ensure images match pred resolution for TC flow estimation
            if images.shape[-2:] != (pred_h, pred_w):
                images = F.interpolate(
                    images[0], size=(pred_h, pred_w),
                    mode='bilinear', align_corners=False
                ).unsqueeze(0)
        else:
            gt_at_pred_res_cpu = gt_depth_metric_cpu  # Same tensor, no copy needed

        del gt_depth_metric  # Free GPU memory

        # === TC-only mode: skip per-frame metrics, depth range, TAE ===
        if self.test_mode == 'tc':
            metrics = {
                'fps': float(fps), 'dataset': str(dataset_name), 'num_frames': T,
                'abs_rel': 0.0, 'a1': 0.0, 'mae': 0.0, 'rmse': 0.0,
                'pred_scale_mean': float(scale[0].mean()), 'pred_shift_mean': float(shift[0].mean()),
                'tae': 0.0, 'tae_reproj': 0.0, 'tae_reproj_gt': 0.0,
                '_per_frame_tae': [], '_tae_spike_frames': [],
            }
            # Compute rTC only
            if T > 1:
                if self.flow_tc is None:
                    self.flow_tc = FlowTemporalConsistency(
                        device=self.device, thr=self.tc_threshold, max_depth=MAX_DEPTH
                    )
                tc_result = self.flow_tc.compute_rtc(
                    images[0], pred_depths_cpu, gt_depths=gt_at_pred_res_cpu
                )
                metrics['rtc'] = tc_result['rtc']
                metrics['rtc_gt'] = tc_result['rtc_gt']
                metrics['_per_frame_rtc'] = tc_result['per_frame_rtc']
                metrics['_per_frame_rtc_gt'] = tc_result['per_frame_rtc_gt']
                metrics['_rtc_ratio_stats'] = tc_result['ratio_stats']
                metrics['_rtc_per_frame_ratio_stats'] = tc_result['per_frame_ratio_stats']
                metrics['_rtc_best_frame_idx'] = tc_result['best_frame_idx']
                metrics['_rtc_worst_frame_idx'] = tc_result['worst_frame_idx']
                metrics['_multi_threshold'] = tc_result.get('multi_threshold', {})
                logger.info(f"Flow TC: rTC={metrics['rtc']:.4f}, rTC_gt={metrics['rtc_gt']:.4f}")

                # SCD-excluded rTC/flickering
                excl_scd = self._compute_excl_scd_metrics(tc_result, reset_frames, T)
                if excl_scd:
                    metrics.update(excl_scd)
                    logger.info(f"Flow TC (excl SCD): rTC={excl_scd['rtc_excl_scd']:.4f}, "
                                f"excluded {excl_scd['_excl_scd_num_excluded']} pairs")

                # TC visualizations
                if self.enable_visualization:
                    images_cpu = images[0].cpu()
                    rtc_best = tc_result['best_frame_idx']
                    rtc_worst = tc_result['worst_frame_idx']
                    per_frame_rtc = tc_result['per_frame_rtc']
                    per_frame_rtc_gt = tc_result['per_frame_rtc_gt']

                    self.flow_tc.save_visualization(
                        pred_depths_cpu, gt_at_pred_res_cpu, rtc_worst, sequence_id,
                        self.save_dir, per_frame_rtc[rtc_worst], label='worst',
                        dataset_name=dataset_name
                    )
                    self.flow_tc.save_visualization(
                        pred_depths_cpu, gt_at_pred_res_cpu, rtc_best, sequence_id,
                        self.save_dir, per_frame_rtc[rtc_best], label='best',
                        dataset_name=dataset_name
                    )
                    self.flow_tc.save_ratio_heatmap(
                        images_cpu, pred_depths_cpu, rtc_worst, sequence_id,
                        self.save_dir, per_frame_rtc[rtc_worst], label='worst',
                        dataset_name=dataset_name
                    )
                    self.flow_tc.save_ratio_heatmap(
                        images_cpu, pred_depths_cpu, rtc_best, sequence_id,
                        self.save_dir, per_frame_rtc[rtc_best], label='best',
                        dataset_name=dataset_name
                    )
                    self.flow_tc.save_rtc_plot(
                        per_frame_rtc, per_frame_rtc_gt, rtc_best, rtc_worst,
                        sequence_id, self.save_dir, dataset_name=dataset_name
                    )
            else:
                metrics['rtc'] = 0.0
                metrics['rtc_gt'] = 0.0
                metrics['_per_frame_rtc'] = []
                metrics['_per_frame_rtc_gt'] = []
                metrics['_rtc_ratio_stats'] = {}
                metrics['_rtc_per_frame_ratio_stats'] = []
                metrics['_rtc_best_frame_idx'] = 0
                metrics['_rtc_worst_frame_idx'] = 0
            # Offload SEA-RAFT to CPU for next sequence
            if self.flow_tc is not None:
                self.flow_tc.offload_to_cpu()

            # === TAE in TC mode (at GT resolution) ===
            if T > 1 and isinstance(batch, dict) and 'image_paths' in batch and self.reproj_tae_calculator.is_supported(dataset_name):
                try:
                    image_paths_for_tae = batch['image_paths'][0]
                    if need_upsample:
                        pred_at_gt_res = torch.stack([
                            F.interpolate(pred_depths_cpu[t:t+1], size=(gt_h, gt_w),
                                          mode='bilinear', align_corners=True)[0]
                            for t in range(T)
                        ], dim=0)
                    else:
                        pred_at_gt_res = pred_depths_cpu
                    reproj_tae_result = self.reproj_tae_calculator.compute_tae(
                        pred_at_gt_res[:, 0],
                        gt_depth_metric_cpu[:, 0],
                        dataset_name,
                        image_paths_for_tae,
                        max_depth=MAX_DEPTH
                    )
                    if need_upsample:
                        del pred_at_gt_res
                    metrics['tae_reproj'] = reproj_tae_result.get('tae_reproj', 0.0)
                    metrics['tae_reproj_gt'] = reproj_tae_result.get('tae_reproj_gt', 0.0)
                    metrics['tae'] = reproj_tae_result.get('tae', 0.0)
                    metrics['_per_frame_tae'] = reproj_tae_result.get('per_frame_tae', [])
                    per_frame_tae = metrics['_per_frame_tae']
                    valid_tae = [x for x in per_frame_tae if not np.isnan(x)]
                    if valid_tae:
                        tae_mean = np.mean(valid_tae)
                        metrics['_tae_spike_frames'] = [i for i, t in enumerate(per_frame_tae) if not np.isnan(t) and t > 2 * tae_mean]
                    else:
                        metrics['_tae_spike_frames'] = []
                    logger.info(f"Reprojection TAE: pred={metrics['tae_reproj']:.4f}%, gt={metrics['tae_reproj_gt']:.4f}%, diff={metrics['tae']:.4f}%")
                except Exception as e:
                    logger.warning(f"Failed to compute reprojection TAE: {e}")
                    metrics['tae_reproj'] = 0.0
                    metrics['tae_reproj_gt'] = 0.0
                    metrics['tae'] = 0.0
                    metrics['_per_frame_tae'] = []
                    metrics['_tae_spike_frames'] = []
            else:
                metrics['tae_reproj'] = 0.0
                metrics['tae_reproj_gt'] = 0.0
                metrics['tae'] = 0.0
                metrics['_per_frame_tae'] = []
                metrics['_tae_spike_frames'] = []

            # === PSR in TC mode ===
            per_frame_scale_ratios_tc = []
            for t in range(T):
                pred_frame = pred_depths_cpu[t, 0]
                gt_frame = gt_at_pred_res_cpu[t, 0]
                valid_mask = (gt_frame > 0) & (gt_frame < MAX_DEPTH) & (pred_frame > 0) & (pred_frame < MAX_DEPTH)
                if valid_mask.sum() > 0:
                    r_t = float(pred_frame[valid_mask].mean() / gt_frame[valid_mask].mean().clamp(min=1e-8))
                    per_frame_scale_ratios_tc.append(r_t)
                else:
                    per_frame_scale_ratios_tc.append(per_frame_scale_ratios_tc[-1] if per_frame_scale_ratios_tc else 1.0)
            if T > 1 and len(per_frame_scale_ratios_tc) > 1:
                psr_values = [abs(per_frame_scale_ratios_tc[i] - per_frame_scale_ratios_tc[i-1]) for i in range(1, len(per_frame_scale_ratios_tc))]
                metrics['psr'] = float(np.mean(psr_values))
                metrics['psr_max'] = float(np.max(psr_values))
                metrics['_per_frame_psr'] = [float(v) for v in psr_values]
                metrics['_per_frame_scale_ratio'] = [float(v) for v in per_frame_scale_ratios_tc]
            else:
                metrics['psr'] = 0.0
                metrics['psr_max'] = 0.0
                metrics['_per_frame_psr'] = []
                metrics['_per_frame_scale_ratio'] = []

            return metrics

        # === Per-frame metrics ===
        frame_metrics = []
        per_frame_scales = []
        per_frame_shifts = []
        per_frame_optimal_scales = []
        per_frame_optimal_shifts = []
        per_frame_scale_ratios = []
        best_frame_idx = 0
        best_frame_abs_rel = float('inf')
        worst_frame_idx = 0
        worst_frame_abs_rel = 0.0

        for t in range(T):
            # === Main metrics at GT resolution (upsample pred per-frame) ===
            gt_frame = gt_depth_metric_cpu[t, 0]  # [gt_h, gt_w]
            if need_upsample:
                pred_frame = F.interpolate(
                    pred_depths_cpu[t:t+1],  # [1, 1, pred_h, pred_w]
                    size=(gt_h, gt_w), mode='bilinear', align_corners=True
                )[0, 0]  # [gt_h, gt_w]
            else:
                pred_frame = pred_depths_cpu[t, 0]

            gt_valid_mask = (gt_frame > 0) & (gt_frame < MAX_DEPTH)
            pred_valid_mask = (pred_frame > 0) & (pred_frame < MAX_DEPTH)
            valid_mask = gt_valid_mask & pred_valid_mask

            # PSR: compute per-frame scale ratio (mean_pred / mean_gt) on valid pixels
            if valid_mask.sum() > 0:
                r_t = float(pred_frame[valid_mask].mean() / gt_frame[valid_mask].mean().clamp(min=1e-8))
                per_frame_scale_ratios.append(r_t)
            else:
                per_frame_scale_ratios.append(per_frame_scale_ratios[-1] if per_frame_scale_ratios else 1.0)

            # Store per-frame scale/shift
            per_frame_scales.append(float(scale[0, t]))
            per_frame_shifts.append(float(shift[0, t]))

            # Compute optimal (oracle) scale/shift via LSE in canonical space (GT resolution)
            if need_upsample:
                dfr_frame_gt_res = F.interpolate(
                    dfr_canonical_cpu[t:t+1].unsqueeze(0),  # [1, 1, pred_h, pred_w]
                    size=(gt_h, gt_w), mode='bilinear', align_corners=True
                )[0, 0]  # [gt_h, gt_w]
            else:
                dfr_frame_gt_res = dfr_canonical_cpu[t]  # [H, W]
            if valid_mask.sum() > 100:
                de_ratio_t = float(de_canonical_ratio_metric[0, t])
                dfr_valid = dfr_frame_gt_res[valid_mask].numpy()
                gt_canonical_valid = (gt_frame[valid_mask] / de_ratio_t).numpy()
                A = np.stack([dfr_valid, np.ones_like(dfr_valid)], axis=1)
                result = np.linalg.lstsq(A, gt_canonical_valid, rcond=None)
                opt_scale, opt_shift = result[0]
                per_frame_optimal_scales.append(float(opt_scale))
                per_frame_optimal_shifts.append(float(opt_shift))
            else:
                per_frame_optimal_scales.append(1.0)
                per_frame_optimal_shifts.append(0.0)

            if valid_mask.sum() > 0:
                frame_metric = self.metrics.compute_metric_depth_metrics(
                    pred_frame, gt_frame, valid_mask
                )
                frame_metrics.append(frame_metric)

                if frame_metric['abs_rel'] < best_frame_abs_rel:
                    best_frame_abs_rel = frame_metric['abs_rel']
                    best_frame_idx = t
                if frame_metric['abs_rel'] > worst_frame_abs_rel:
                    worst_frame_abs_rel = frame_metric['abs_rel']
                    worst_frame_idx = t

        if len(frame_metrics) == 0:
            logger.warning(f"No valid frames for sequence {sequence_id}")
            return {k: 0.0 for k in ["mae", "rmse", "abs_rel", "a1", "tae", "fps"]}

        # Average metrics
        metrics = {}
        for key in frame_metrics[0].keys():
            values = [m[key] for m in frame_metrics]
            metrics[key] = float(np.mean(values))
            if key in ['abs_rel', 'a1']:
                metrics[f'{key}_min'] = float(np.min(values))
                metrics[f'{key}_max'] = float(np.max(values))

        metrics['fps'] = float(fps)
        metrics['dataset'] = str(dataset_name)
        metrics['num_frames'] = T
        metrics['best_frame_idx'] = best_frame_idx
        metrics['worst_frame_idx'] = worst_frame_idx

        # Scale/shift stats
        metrics['pred_scale_mean'] = float(np.mean(per_frame_scales))
        metrics['pred_shift_mean'] = float(np.mean(per_frame_shifts))
        metrics['pred_scale_std'] = float(np.std(per_frame_scales)) if T > 1 else 0.0
        metrics['pred_shift_std'] = float(np.std(per_frame_shifts)) if T > 1 else 0.0
        metrics['pred_scale_min'] = float(np.min(per_frame_scales))
        metrics['pred_scale_max'] = float(np.max(per_frame_scales))
        metrics['pred_shift_min'] = float(np.min(per_frame_shifts))
        metrics['pred_shift_max'] = float(np.max(per_frame_shifts))
        metrics['optimal_scale_mean'] = float(np.mean(per_frame_optimal_scales))
        metrics['optimal_shift_mean'] = float(np.mean(per_frame_optimal_shifts))
        metrics['optimal_scale_min'] = float(np.min(per_frame_optimal_scales))
        metrics['optimal_scale_max'] = float(np.max(per_frame_optimal_scales))
        metrics['optimal_shift_min'] = float(np.min(per_frame_optimal_shifts))
        metrics['optimal_shift_max'] = float(np.max(per_frame_optimal_shifts))

        # Private data for JSON exports
        metrics['_per_frame_scales'] = per_frame_scales
        metrics['_per_frame_shifts'] = per_frame_shifts
        metrics['_per_frame_optimal_scales'] = per_frame_optimal_scales
        metrics['_per_frame_optimal_shifts'] = per_frame_optimal_shifts

        # SCD info (d_patch_mean per frame + reset frames)
        metrics['_per_frame_d_patch_mean'] = per_frame_d_patch_mean
        if reset_frames:
            metrics['reset_frames'] = reset_frames
            metrics['num_resets'] = len(reset_frames)
            logger.info(f"Seq {sequence_id}: SCD triggered {len(reset_frames)} resets at frames {reset_frames}")
        else:
            metrics['num_resets'] = 0
        if per_frame_d_patch_mean:
            metrics['d_patch_mean_max'] = float(max(per_frame_d_patch_mean))
            metrics['d_patch_mean_mean'] = float(np.mean(per_frame_d_patch_mean))

        # === Depth range analysis ===
        depth_ranges = [(0, 10), (10, 30), (30, int(self.max_depth))]
        depth_range_metrics = {}
        for depth_min, depth_max in depth_ranges:
            range_name = f"{depth_min}-{depth_max}m"
            range_abs_rels = []
            range_a1s = []
            range_pixel_counts = []

            for t in range(T):
                gt_frame = gt_depth_metric_cpu[t, 0]  # [gt_h, gt_w]
                if need_upsample:
                    pred_frame = F.interpolate(
                        pred_depths_cpu[t:t+1], size=(gt_h, gt_w),
                        mode='bilinear', align_corners=True
                    )[0, 0]
                else:
                    pred_frame = pred_depths_cpu[t, 0]
                range_mask = (gt_frame >= depth_min) & (gt_frame < depth_max) & (gt_frame > 0) & (pred_frame > 0) & (pred_frame < MAX_DEPTH)
                if range_mask.sum() > 0:
                    pred_valid = pred_frame[range_mask]
                    gt_valid = gt_frame[range_mask]
                    abs_rel = torch.abs(pred_valid - gt_valid) / gt_valid
                    range_abs_rels.append(abs_rel.mean().item())
                    thresh = torch.maximum(gt_valid / (pred_valid + 1e-8), pred_valid / (gt_valid + 1e-8))
                    a1 = (thresh < 1.25).float().mean().item()
                    range_a1s.append(a1)
                    range_pixel_counts.append(range_mask.sum().item())

            if range_abs_rels:
                depth_range_metrics[range_name] = {
                    'abs_rel': float(np.mean(range_abs_rels)),
                    'a1': float(np.mean(range_a1s)),
                    'pixel_count': int(np.sum(range_pixel_counts))
                }
            else:
                depth_range_metrics[range_name] = {'abs_rel': 0.0, 'a1': 0.0, 'pixel_count': 0}

        metrics['_depth_range_analysis'] = depth_range_metrics

        # === Reprojection TAE (at GT resolution) ===
        if self.test_mode == 'ea':
            # EA mode: skip TAE
            metrics['tae_reproj'] = 0.0
            metrics['tae_reproj_gt'] = 0.0
            metrics['tae'] = 0.0
            metrics['_per_frame_tae'] = []
            metrics['_tae_spike_frames'] = []
        elif T > 1 and isinstance(batch, dict) and 'image_paths' in batch and self.reproj_tae_calculator.is_supported(dataset_name):
            try:
                image_paths_for_tae = batch['image_paths'][0]
                # Upsample pred to GT resolution for TAE
                if need_upsample:
                    pred_at_gt_res = torch.stack([
                        F.interpolate(pred_depths_cpu[t:t+1], size=(gt_h, gt_w),
                                      mode='bilinear', align_corners=True)[0]
                        for t in range(T)
                    ], dim=0)  # [T, 1, gt_h, gt_w]
                else:
                    pred_at_gt_res = pred_depths_cpu
                reproj_tae_result = self.reproj_tae_calculator.compute_tae(
                    pred_at_gt_res[:, 0],
                    gt_depth_metric_cpu[:, 0],
                    dataset_name,
                    image_paths_for_tae,
                    max_depth=MAX_DEPTH
                )
                if need_upsample:
                    del pred_at_gt_res
                metrics['tae_reproj'] = reproj_tae_result.get('tae_reproj', 0.0)
                metrics['tae_reproj_gt'] = reproj_tae_result.get('tae_reproj_gt', 0.0)
                metrics['tae'] = reproj_tae_result.get('tae', 0.0)
                metrics['_per_frame_tae'] = reproj_tae_result.get('per_frame_tae', [])

                per_frame_tae = metrics['_per_frame_tae']
                valid_tae = [x for x in per_frame_tae if not np.isnan(x)]
                if valid_tae:
                    tae_mean = np.mean(valid_tae)
                    metrics['_tae_spike_frames'] = [i for i, t in enumerate(per_frame_tae) if not np.isnan(t) and t > 2 * tae_mean]
                else:
                    metrics['_tae_spike_frames'] = []

                logger.info(f"Reprojection TAE: pred={metrics['tae_reproj']:.4f}%, gt={metrics['tae_reproj_gt']:.4f}%, diff={metrics['tae']:.4f}%")
            except Exception as e:
                logger.warning(f"Failed to compute reprojection TAE: {e}")
                metrics['tae_reproj'] = 0.0
                metrics['tae_reproj_gt'] = 0.0
                metrics['tae'] = 0.0
                metrics['_per_frame_tae'] = []
                metrics['_tae_spike_frames'] = []
        else:
            metrics['tae_reproj'] = 0.0
            metrics['tae_reproj_gt'] = 0.0
            metrics['tae'] = 0.0
            metrics['_per_frame_tae'] = []
            metrics['_tae_spike_frames'] = []

        # === Flow-based Temporal Consistency (rTC) ===
        if self.test_mode == 'ea':
            # EA mode: skip rTC (avoids loading SEA-RAFT, saves ~200MB GPU)
            metrics['rtc'] = 0.0
            metrics['rtc_gt'] = 0.0
            metrics['_per_frame_rtc'] = []
            metrics['_per_frame_rtc_gt'] = []
            metrics['_rtc_ratio_stats'] = {}
            metrics['_rtc_per_frame_ratio_stats'] = []
            metrics['_rtc_best_frame_idx'] = 0
            metrics['_rtc_worst_frame_idx'] = 0
        elif T > 1:
            if self.flow_tc is None:
                self.flow_tc = FlowTemporalConsistency(
                    device=self.device, thr=self.tc_threshold, max_depth=MAX_DEPTH
                )
            tc_result = self.flow_tc.compute_rtc(
                images[0], pred_depths_cpu, gt_depths=gt_at_pred_res_cpu
            )
            metrics['rtc'] = tc_result['rtc']
            metrics['rtc_gt'] = tc_result['rtc_gt']
            metrics['_per_frame_rtc'] = tc_result['per_frame_rtc']
            metrics['_per_frame_rtc_gt'] = tc_result['per_frame_rtc_gt']
            metrics['_rtc_ratio_stats'] = tc_result['ratio_stats']
            metrics['_rtc_per_frame_ratio_stats'] = tc_result['per_frame_ratio_stats']
            metrics['_rtc_best_frame_idx'] = tc_result['best_frame_idx']
            metrics['_rtc_worst_frame_idx'] = tc_result['worst_frame_idx']
            metrics['_multi_threshold'] = tc_result.get('multi_threshold', {})
            logger.info(f"Flow TC: rTC={metrics['rtc']:.4f}, rTC_gt={metrics['rtc_gt']:.4f}")

            # SCD-excluded rTC/flickering
            excl_scd = self._compute_excl_scd_metrics(tc_result, reset_frames, T)
            if excl_scd:
                metrics.update(excl_scd)
                logger.info(f"Flow TC (excl SCD): rTC={excl_scd['rtc_excl_scd']:.4f}, "
                            f"excluded {excl_scd['_excl_scd_num_excluded']} pairs")
        else:
            metrics['rtc'] = 0.0
            metrics['rtc_gt'] = 0.0
            metrics['_per_frame_rtc'] = []
            metrics['_per_frame_rtc_gt'] = []
            metrics['_rtc_ratio_stats'] = {}
            metrics['_rtc_per_frame_ratio_stats'] = []
            metrics['_rtc_best_frame_idx'] = 0
            metrics['_rtc_worst_frame_idx'] = 0

        # === Prediction Stability Ratio (PSR) ===
        if self.test_mode == 'ea':
            # EA mode: skip PSR
            metrics['psr'] = 0.0
            metrics['psr_max'] = 0.0
            metrics['_per_frame_psr'] = []
            metrics['_per_frame_scale_ratio'] = []
        elif T > 1 and len(per_frame_scale_ratios) > 1:
            psr_values = [abs(per_frame_scale_ratios[i] - per_frame_scale_ratios[i-1]) for i in range(1, len(per_frame_scale_ratios))]
            metrics['psr'] = float(np.mean(psr_values))
            metrics['psr_max'] = float(np.max(psr_values))
            metrics['_per_frame_psr'] = [float(v) for v in psr_values]
            metrics['_per_frame_scale_ratio'] = [float(v) for v in per_frame_scale_ratios]
        else:
            metrics['psr'] = 0.0
            metrics['psr_max'] = 0.0
            metrics['_per_frame_psr'] = []
            metrics['_per_frame_scale_ratio'] = []

        logger.info(
            f"  AbsRel={metrics.get('abs_rel', 0):.4f}, MAE={metrics.get('mae', 0):.4f}, "
            f"RMSE={metrics.get('rmse', 0):.4f}, d1={metrics.get('a1', 0):.4f}, "
            f"TAE={metrics.get('tae', 0):.4f}, FPS={fps:.2f}, "
            f"Scale={metrics['pred_scale_mean']:.4f}, Shift={metrics['pred_shift_mean']:.6f}"
        )

        # === Visualizations ===
        if self.enable_visualization:
            images_cpu = images[0].cpu()

            # Video/GIF (skip for long-sequence datasets)
            skip_video_datasets = ['urbansyn', 'unreal4k']
            should_save_video = not any(s in dataset_name.lower() for s in skip_video_datasets)
            if should_save_video and self.config.eval.get('out_video', True):
                self._save_video(images_cpu, pred_depths_cpu, gt_at_pred_res_cpu,
                                sequence_id, dataset_name)

            # Frame PNGs
            self._visualize_sequence(images_cpu, pred_depths_cpu, gt_at_pred_res_cpu,
                                     sequence_id, metrics, dataset_name=dataset_name)

            # Error heatmaps
            self._save_error_heatmaps(pred_depths_cpu, gt_at_pred_res_cpu,
                                       sequence_id, metrics)

            # Best/Worst frame visualizations
            self._save_frame_visualizations(
                images_cpu, pred_depths_cpu, gt_at_pred_res_cpu,
                best_frame_idx, sequence_id, frame_metrics, metrics,
                frame_type='best', fps=fps, dataset_name=dataset_name
            )
            self._save_frame_visualizations(
                images_cpu, pred_depths_cpu, gt_at_pred_res_cpu,
                worst_frame_idx, sequence_id, frame_metrics, metrics,
                frame_type='worst', fps=fps, dataset_name=dataset_name
            )

            # Flow TC visualizations
            if self.flow_tc is not None and metrics.get('_per_frame_rtc'):
                try:
                    rtc_best = metrics['_rtc_best_frame_idx']
                    rtc_worst = metrics['_rtc_worst_frame_idx']
                    per_frame_rtc = metrics['_per_frame_rtc']
                    per_frame_rtc_gt = metrics['_per_frame_rtc_gt']

                    # Depth grids for best/worst TC pairs
                    self.flow_tc.save_visualization(
                        pred_depths_cpu, gt_at_pred_res_cpu, rtc_worst, sequence_id,
                        self.save_dir, per_frame_rtc[rtc_worst], label='worst',
                        dataset_name=dataset_name
                    )
                    self.flow_tc.save_visualization(
                        pred_depths_cpu, gt_at_pred_res_cpu, rtc_best, sequence_id,
                        self.save_dir, per_frame_rtc[rtc_best], label='best',
                        dataset_name=dataset_name
                    )

                    # Ratio heatmaps
                    self.flow_tc.save_ratio_heatmap(
                        images_cpu, pred_depths_cpu, rtc_worst, sequence_id,
                        self.save_dir, per_frame_rtc[rtc_worst], label='worst',
                        dataset_name=dataset_name
                    )
                    self.flow_tc.save_ratio_heatmap(
                        images_cpu, pred_depths_cpu, rtc_best, sequence_id,
                        self.save_dir, per_frame_rtc[rtc_best], label='best',
                        dataset_name=dataset_name
                    )

                    # rTC line plot
                    self.flow_tc.save_rtc_plot(
                        per_frame_rtc, per_frame_rtc_gt, rtc_best, rtc_worst,
                        sequence_id, self.save_dir, dataset_name=dataset_name
                    )
                except Exception as e:
                    logger.warning(f"Failed to save TC visualizations: {e}")

        # Export individual frames if --best-figure or --frame option is enabled
        # NOTE: This is independent of --visualization flag
        export_frame_indices = []
        if self.export_best_figure and len(frame_metrics) > 0:
            export_frame_indices.append(best_frame_idx)
            interval_info = f"interval={self.frame_interval}" if self.frame_interval else "interval=1"
            logger.info(f"Exporting best frame {best_frame_idx} ±4 intervals ({interval_info}) (--best-figure)")
        if self.export_frames is not None:
            for fidx in self.export_frames:
                if fidx < len(pred_depths_cpu):
                    if fidx not in export_frame_indices:
                        export_frame_indices.append(fidx)
                    interval_info = f"interval={self.frame_interval}" if self.frame_interval else "interval=1"
                    logger.info(f"Exporting user-specified frame {fidx} ±4 intervals ({interval_info}) (--frame)")
                else:
                    logger.warning(f"Requested frame {fidx} exceeds sequence length {len(pred_depths_cpu)}, skipping export")

        for export_frame_idx in export_frame_indices:
            self._export_figure_frames(
                images=images[0].cpu(),  # [T, 3, H, W]
                pred_depths=pred_depths_cpu,  # [T, 1, H, W]
                gt_depths=gt_at_pred_res_cpu,  # [T, 1, pred_h, pred_w]
                best_frame_idx=export_frame_idx,
                sequence_id=sequence_id,
                dataset_name=dataset_name,
            )

        # Offload SEA-RAFT to CPU for next sequence
        if self.flow_tc is not None:
            self.flow_tc.offload_to_cpu()

        return metrics

    def _save_video(self, images, pred_depths, gt_depths, sequence_id, dataset_name):
        """Save depth sequence as GIF/MP4."""
        try:
            from utils.gear_video_utils import save_video as save_video_util
            valid_mask = (gt_depths > 0) & (gt_depths < self.max_depth)
            save_video_util(
                images, pred_depths, gt_depths, valid_mask, sequence_id,
                save_dir=self.save_dir, config=self.config
            )
        except Exception as e:
            logger.warning(f"Failed to save video for seq {sequence_id}: {e}")

    def _visualize_sequence(self, images, pred_depths, gt_depths, sequence_id, metrics,
                            dataset_name='unknown'):
        """Save per-frame PNGs: 1x3 grid (Image, GT, Pred)."""
        T = images.shape[0]
        MAX_DEPTH = self.max_depth
        seq_dir = self.save_dir / "frames" / f"seq{sequence_id:04d}"
        seq_dir.mkdir(parents=True, exist_ok=True)

        if self.frame_interval is not None:
            frame_indices = list(range(0, T, self.frame_interval))
        else:
            frame_indices = list(range(T))

        # Sparse detection by dataset name
        ds_lower = dataset_name.lower() if isinstance(dataset_name, str) else 'unknown'
        is_sparse = any(s in ds_lower for s in ['eth3d', 'waymo_seg'])

        for t in frame_indices:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            img = images[t].permute(1, 2, 0).float().numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = np.clip(img, 0, 1)
            axes[0].imshow(img)
            axes[0].set_title(f'Image (Frame {t})')
            axes[0].axis('off')

            pred = pred_depths[t, 0].numpy()
            gt = gt_depths[t, 0].numpy()

            gt_valid = (gt > 0) & (gt < MAX_DEPTH)
            gt_display = np.where(gt_valid, gt, np.nan)
            if gt_valid.sum() > 0:
                vmin = np.nanpercentile(gt_display, 2)
                vmax = np.nanpercentile(gt_display, 98)
            else:
                vmin, vmax = 0, 1

            if is_sparse:
                gt_exists = (gt > 0)
                valid_pixels_per_row = gt_exists.sum(axis=1)
                valid_rows = valid_pixels_per_row >= 10
                valid_row_indices = np.where(valid_rows)[0]
                if len(valid_row_indices) > 0:
                    height_mask = np.zeros_like(gt, dtype=bool)
                    height_mask[valid_row_indices.min():valid_row_indices.max()+1, :] = True
                else:
                    height_mask = np.ones_like(gt, dtype=bool)
                pred_show_mask = height_mask & (pred > 0) & (pred < MAX_DEPTH)
            else:
                pred_show_mask = gt_valid

            cmap = plt.cm.plasma_r.copy()
            cmap.set_bad(color='black')
            axes[1].imshow(gt_display, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[1].set_title(f'GT Depth')
            axes[1].axis('off')

            pred_display = np.where(pred_show_mask, pred, np.nan)
            axes[2].imshow(pred_display, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[2].set_title(f'Pred Depth')
            axes[2].axis('off')

            plt.tight_layout()
            plt.savefig(seq_dir / f"frame_{t:04d}.png", dpi=100, bbox_inches='tight')
            plt.close(fig)

    def _save_error_heatmaps(self, pred_depths, gt_depths, sequence_id, metrics=None):
        """Save per-frame error heatmaps with scale/shift overlay."""
        T = pred_depths.shape[0]
        MAX_DEPTH = self.max_depth
        heatmap_dir = self.save_dir / "error_heatmaps" / f"seq{sequence_id:04d}"
        heatmap_dir.mkdir(parents=True, exist_ok=True)

        pred_scales = metrics.get('_per_frame_scales', []) if metrics else []
        pred_shifts = metrics.get('_per_frame_shifts', []) if metrics else []
        opt_scales = metrics.get('_per_frame_optimal_scales', []) if metrics else []
        opt_shifts = metrics.get('_per_frame_optimal_shifts', []) if metrics else []

        frame_indices = list(range(0, T, self.frame_interval)) if self.frame_interval else list(range(T))

        for t in frame_indices:
            pred = pred_depths[t, 0].numpy()
            gt = gt_depths[t, 0].numpy()
            valid = (gt > 0) & (gt < MAX_DEPTH) & (pred > 0) & (pred < MAX_DEPTH)

            if valid.sum() > 0:
                error = np.abs(pred - gt) / (gt + 1e-8)
                error_display = np.where(valid, error, np.nan)
                abs_rel = np.mean(error[valid])
                thresh = np.maximum(gt[valid] / (pred[valid] + 1e-8), pred[valid] / (gt[valid] + 1e-8))
                delta_1 = np.mean(thresh < 1.25)

                fig, ax = plt.subplots(figsize=(8, 6.8))
                cmap = plt.cm.hot.copy()
                cmap.set_bad(color='black')
                im = ax.imshow(error_display, cmap=cmap, vmin=0, vmax=1)
                plt.colorbar(im, ax=ax, label='AbsRel Error')
                ax.set_title(f'Seq {sequence_id} Frame {t} | AbsRel: {abs_rel:.3f} | d1: {delta_1:.3f}')
                ax.axis('off')

                if t < len(pred_scales) and t < len(opt_scales):
                    pred_s, pred_sh = pred_scales[t], pred_shifts[t]
                    opt_s, opt_sh = opt_scales[t], opt_shifts[t]
                    scale_ratio = pred_s / (opt_s + 1e-8)
                    shift_diff = pred_sh - opt_sh
                    info_text = (f'Pred: scale={pred_s:.3f}, shift={pred_sh:.3f}  |  '
                                f'Optimal: scale={opt_s:.3f}, shift={opt_sh:.3f}  |  '
                                f'D: scale={scale_ratio:.3f}x, shift={shift_diff:+.3f}')
                    fig.text(0.5, 0.02, info_text, ha='center', fontsize=8,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

                plt.tight_layout(rect=[0, 0.05, 1, 1])
                plt.savefig(heatmap_dir / f"error_{t:04d}.png", dpi=100, bbox_inches='tight')
                plt.close(fig)

    def _save_frame_visualizations(self, images, pred_depths, gt_depths,
                                    frame_idx, sequence_id, frame_metrics,
                                    seq_metrics, frame_type='best', fps=None,
                                    dataset_name='unknown'):
        """
        Save best/worst frame visualization.
        3x3 grid:
            Row 1: Input, GT Depth, Pred Depth
            Row 2: GT Valid Mask, Error Map, Metrics Panel
            Row 3: Depth Distribution (colspan=full)
        """
        MAX_DEPTH = self.max_depth
        vis_dir = self.save_dir / f"{frame_type}_frames"
        vis_dir.mkdir(parents=True, exist_ok=True)

        T = images.shape[0]
        t = frame_idx

        img = images[t].permute(1, 2, 0).float().numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = np.clip(img, 0, 1)

        pred = pred_depths[t, 0].numpy()
        gt = gt_depths[t, 0].numpy()

        gt_valid = (gt > 0) & (gt < MAX_DEPTH)
        pred_valid = (pred > 0) & (pred < MAX_DEPTH)
        error_valid_mask = gt_valid & pred_valid

        # Sparse detection by dataset name
        ds_lower = dataset_name.lower() if isinstance(dataset_name, str) else 'unknown'
        is_sparse = any(s in ds_lower for s in ['eth3d', 'waymo_seg'])

        if is_sparse:
            gt_exists = (gt > 0)
            valid_pixels_per_row = gt_exists.sum(axis=1)
            valid_rows = valid_pixels_per_row >= 10
            valid_row_indices = np.where(valid_rows)[0]
            if len(valid_row_indices) > 0:
                height_mask = np.zeros_like(gt, dtype=bool)
                height_mask[valid_row_indices.min():valid_row_indices.max()+1, :] = True
            else:
                height_mask = np.ones_like(gt, dtype=bool)
            pred_show_mask = height_mask & pred_valid
        else:
            pred_show_mask = gt_valid

        gt_display = np.where(gt_valid, gt, np.nan)
        pred_display = np.where(pred_show_mask, pred, np.nan)

        if gt_valid.sum() > 0:
            vmin = np.nanpercentile(gt_display, 2)
            vmax = np.nanpercentile(gt_display, 98)
        else:
            vmin, vmax = 0, 1

        abs_error = np.abs(pred - gt)
        abs_error_masked = np.where(error_valid_mask, abs_error, np.nan)
        abs_rel_frame = float(seq_metrics.get(f'{frame_type}_frame_abs_rel', seq_metrics.get('abs_rel', 0)))
        if t < len(frame_metrics):
            abs_rel_frame = frame_metrics[t].get('abs_rel', 0) if t < len(frame_metrics) else 0

        fig = plt.figure(figsize=(15, 14))
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

        # Row 1: Input, GT, Pred
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img)
        ax1.set_title(f'Input Image (Frame {t})', fontsize=12, fontweight='bold')
        ax1.axis('off')

        cmap = plt.cm.plasma_r.copy()
        cmap.set_bad(color='black')

        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(gt_display, cmap=cmap, vmin=vmin, vmax=vmax)
        ax2.set_title('GT Depth (m)', fontsize=12, fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(pred_display, cmap=cmap, vmin=vmin, vmax=vmax)
        ax3.set_title('Pred Depth (m)', fontsize=12, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        # Row 2: GT Valid Mask, Error Map, Metrics
        ax4 = fig.add_subplot(gs[1, 0])
        gt_valid_ratio = gt_valid.sum() / gt_valid.size
        ax4.imshow(gt_valid.astype(np.uint8), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        ax4.set_title(f'GT Valid Mask ({gt_valid_ratio*100:.1f}%)\ninvalid: black',
                     fontsize=12, fontweight='bold')
        ax4.axis('off')

        ax5 = fig.add_subplot(gs[1, 1])
        if error_valid_mask.sum() > 0:
            error_vmax = np.nanpercentile(abs_error_masked, 95)
            im5 = ax5.imshow(abs_error_masked, cmap='hot', vmin=0, vmax=max(error_vmax, 0.01))
            plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
            ax5.set_title(f'Error Map\nMean: {np.nanmean(abs_error_masked):.3f}m',
                         fontsize=12, fontweight='bold')
        else:
            ax5.text(0.5, 0.5, 'No Valid Pixels', ha='center', va='center',
                    transform=ax5.transAxes, fontsize=12, color='red')
        ax5.axis('off')

        ax6 = fig.add_subplot(gs[1, 2])
        y_pos = 0.95
        ax6.text(0.05, y_pos, f'Seq {sequence_id} | {frame_type.upper()} Frame {t}',
                fontsize=10, transform=ax6.transAxes,
                bbox=dict(boxstyle="round", facecolor='wheat'))
        y_pos -= 0.10
        scales = seq_metrics.get('_per_frame_scales', [])
        shifts = seq_metrics.get('_per_frame_shifts', [])
        if t < len(scales):
            ax6.text(0.05, y_pos, f'scale={scales[t]:.3f}, shift={shifts[t]:.3f}',
                    fontsize=9, transform=ax6.transAxes,
                    bbox=dict(boxstyle="round", facecolor='wheat'))
            y_pos -= 0.10
        if fps is not None:
            ax6.text(0.05, y_pos, f'FPS: {fps:.1f}', fontsize=9,
                    transform=ax6.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen'))
            y_pos -= 0.10
        if t < len(frame_metrics):
            fm = frame_metrics[t]
            for key, color in [('abs_rel', 'lightcoral'), ('a1', 'lightgreen'),
                                ('a2', 'lightgreen'), ('a3', 'lightgreen'),
                                ('rmse', 'wheat'), ('mae', 'lightblue')]:
                if key in fm:
                    label = key.upper() if key.startswith('a') else key.replace('_', ' ').title()
                    val = fm[key]
                    ax6.text(0.05, y_pos, f'{label}: {val:.4f}', fontsize=9,
                            transform=ax6.transAxes,
                            bbox=dict(boxstyle="round", facecolor=color))
                    y_pos -= 0.08
        ax6.set_title('Metrics', fontsize=12, fontweight='bold')
        ax6.axis('off')

        # Row 3: Depth Distribution
        ax7 = fig.add_subplot(gs[2, :])
        if error_valid_mask.sum() > 0:
            gt_vals = gt[error_valid_mask]
            pred_vals = pred[error_valid_mask]
            bins = np.linspace(min(gt_vals.min(), pred_vals.min()),
                              max(gt_vals.max(), pred_vals.max()), 50)
            ax7.hist(gt_vals, bins=bins, alpha=0.6, label='GT', color='blue', density=True)
            ax7.hist(pred_vals, bins=bins, alpha=0.6, label='Pred', color='red', density=True)
            ax7.set_xlabel('Depth (m)')
            ax7.set_ylabel('Density')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        ax7.set_title('Depth Distribution', fontsize=12, fontweight='bold')

        save_path = vis_dir / f"{frame_type}_frame_seq{sequence_id:04d}_{t}_absrel_{abs_rel_frame:.4f}.png"
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

    def _export_figure_frames(self, images, pred_depths, gt_depths, best_frame_idx,
                              sequence_id, dataset_name):
        """
        Export individual frames around best_frame (±4 intervals, total 9 frames).
        Saves: original image, GT depth (colormap), pred depth (colormap).

        Args:
            images: [T, 3, H, W] tensor
            pred_depths: [T, 1, H, W] tensor in meters
            gt_depths: [T, 1, H, W] tensor in meters
            best_frame_idx: int, index of center frame
            sequence_id: int, sequence identifier
            dataset_name: str, name of dataset
        """
        import cv2
        import matplotlib

        T = images.shape[0]
        MAX_DEPTH = self.max_depth

        # Determine frame range with frame_interval support
        frame_interval = self.frame_interval if self.frame_interval is not None else 1
        target_num_frames = 9
        half_count = (target_num_frames - 1) // 2  # 4

        # Generate candidate frame indices centered around best_frame_idx
        frame_indices = []
        for i in range(-half_count, half_count + 1):
            idx = best_frame_idx + i * frame_interval
            if 0 <= idx < T:
                frame_indices.append(idx)

        # Extend in available direction if not enough frames
        while len(frame_indices) < target_num_frames:
            first = frame_indices[0]
            new_back = first - frame_interval
            last = frame_indices[-1]
            new_forward = last + frame_interval

            extended = False
            if new_back >= 0:
                frame_indices.insert(0, new_back)
                extended = True
            elif new_forward < T:
                frame_indices.append(new_forward)
                extended = True

            if not extended:
                break

        # Ensure best_frame_idx is included
        if best_frame_idx not in frame_indices and best_frame_idx < T:
            frame_indices.append(best_frame_idx)
            frame_indices.sort()

        logger.info(f"Exporting figure frames for sequence {sequence_id}: frames {frame_indices} (center={best_frame_idx}, interval={frame_interval})")

        # Create figures directory
        figures_dir = self.save_dir / "figures" / f"seq{sequence_id:04d}"
        figures_dir.mkdir(parents=True, exist_ok=True)

        cmap = matplotlib.colormaps.get_cmap('plasma_r').copy()
        cmap.set_bad(color='black')

        for t in frame_indices:
            # 1. Save original image
            img = images[t].cpu().numpy()  # [3, H, W]
            mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            img = img * std + mean
            img = np.transpose(img, (1, 2, 0))  # [H, W, 3]
            img = np.clip(img, 0, 1)
            img = (img * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(figures_dir / f"frame_{t:04d}_image.png"), img)

            # 2. Save GT depth (colormap)
            gt_depth = gt_depths[t, 0].cpu().numpy()  # [H, W]
            pred_depth = pred_depths[t, 0].cpu().numpy()  # [H, W]

            gt_valid = (gt_depth > 0) & (gt_depth < MAX_DEPTH)
            gt_density = gt_valid.sum() / gt_valid.size
            is_sparse = gt_density < 0.5

            gt_display = np.where(gt_valid, gt_depth, np.nan)
            if gt_valid.any():
                gt_vmin = np.nanpercentile(gt_depth[gt_valid], 2)
                gt_vmax = np.nanpercentile(gt_depth[gt_valid], 98)
            else:
                gt_vmin, gt_vmax = 0, MAX_DEPTH

            gt_colored = cmap((gt_display - gt_vmin) / (gt_vmax - gt_vmin + 1e-8))
            gt_colored = (gt_colored[:, :, :3] * 255).astype(np.uint8)
            gt_colored = cv2.cvtColor(gt_colored, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(figures_dir / f"frame_{t:04d}_gt_depth.png"), gt_colored)

            # 3. Save pred depth (colormap) - use GT range for comparison
            if is_sparse:
                valid_pixels_per_row = gt_valid.sum(axis=1)
                valid_row_indices = np.where(valid_pixels_per_row >= 10)[0]
                if len(valid_row_indices) > 0:
                    height_mask = np.zeros_like(pred_depth, dtype=bool)
                    height_mask[valid_row_indices.min():valid_row_indices.max()+1, :] = True
                else:
                    height_mask = np.ones_like(pred_depth, dtype=bool)
                pred_show_mask = height_mask & (pred_depth > 0) & (pred_depth < MAX_DEPTH)
            else:
                pred_show_mask = gt_valid

            pred_display = np.where(pred_show_mask, pred_depth, np.nan)
            pred_colored = cmap((pred_display - gt_vmin) / (gt_vmax - gt_vmin + 1e-8))
            pred_colored = (pred_colored[:, :, :3] * 255).astype(np.uint8)
            pred_colored = cv2.cvtColor(pred_colored, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(figures_dir / f"frame_{t:04d}_pred_depth.png"), pred_colored)

        logger.info(f"Exported {len(frame_indices)} frames × 3 types = {len(frame_indices) * 3} images to {figures_dir}")

    def _save_scale_shift_comparison(self, all_metrics):
        """Save scale_shift_comparison.json."""
        scale_shift_comparison = []
        for result in all_metrics:
            seq_id = result.get('sequence_id', -1)
            pred_scales = result.get('_per_frame_scales', [])
            pred_shifts = result.get('_per_frame_shifts', [])
            opt_scales = result.get('_per_frame_optimal_scales', [])
            opt_shifts = result.get('_per_frame_optimal_shifts', [])

            per_frame_scale_ratios = []
            per_frame_shift_diffs = []
            if len(pred_scales) == len(opt_scales) and len(pred_scales) > 0:
                per_frame_scale_ratios = [p / (o + 1e-8) for p, o in zip(pred_scales, opt_scales)]
                per_frame_shift_diffs = [p - o for p, o in zip(pred_shifts, opt_shifts)]

            entry = {
                'sequence_id': seq_id,
                'abs_rel': result.get('abs_rel', 0.0),
                'pred_scale_mean': result.get('pred_scale_mean', 0),
                'pred_scale_std': result.get('pred_scale_std', 0),
                'pred_scale_min': result.get('pred_scale_min', 0),
                'pred_scale_max': result.get('pred_scale_max', 0),
                'pred_shift_mean': result.get('pred_shift_mean', 0),
                'pred_shift_std': result.get('pred_shift_std', 0),
                'pred_shift_min': result.get('pred_shift_min', 0),
                'pred_shift_max': result.get('pred_shift_max', 0),
                'optimal_scale_mean': result.get('optimal_scale_mean', 0),
                'optimal_scale_min': result.get('optimal_scale_min', 0),
                'optimal_scale_max': result.get('optimal_scale_max', 0),
                'optimal_shift_mean': result.get('optimal_shift_mean', 0),
                'optimal_shift_min': result.get('optimal_shift_min', 0),
                'optimal_shift_max': result.get('optimal_shift_max', 0),
            }

            if per_frame_scale_ratios:
                entry['scale_ratio_mean'] = float(np.mean(per_frame_scale_ratios))
                entry['scale_ratio_min'] = float(np.min(per_frame_scale_ratios))
                entry['scale_ratio_max'] = float(np.max(per_frame_scale_ratios))
                entry['shift_diff_mean'] = float(np.mean(per_frame_shift_diffs))
                entry['shift_diff_min'] = float(np.min(per_frame_shift_diffs))
                entry['shift_diff_max'] = float(np.max(per_frame_shift_diffs))

                # Per-frame details
                entry['pred_vs_optimal'] = [
                    [pred_scales[i], opt_scales[i], per_frame_scale_ratios[i],
                     pred_shifts[i], opt_shifts[i], per_frame_shift_diffs[i]]
                    for i in range(len(pred_scales))
                ]

                # Scale drift
                if len(pred_scales) > 1:
                    scale_changes = [abs(pred_scales[i+1] - pred_scales[i]) for i in range(len(pred_scales)-1)]
                    top_drift_idx = sorted(range(len(scale_changes)), key=lambda i: scale_changes[i], reverse=True)[:3]
                    entry['scale_drift'] = {
                        'max_change': float(max(scale_changes)),
                        'top_frames': [(i, float(scale_changes[i])) for i in top_drift_idx]
                    }

            scale_shift_comparison.append(entry)

        with open(self.save_dir / "scale_shift_comparison.json", 'w') as f:
            json.dump(scale_shift_comparison, f, indent=2, default=str)
        logger.info(f"Scale/shift comparison saved to {self.save_dir / 'scale_shift_comparison.json'}")

    def _save_depth_range_analysis(self, all_metrics):
        """Save depth_range_analysis.json."""
        depth_range_analysis = []
        for result in all_metrics:
            if '_depth_range_analysis' in result:
                entry = {
                    'sequence_id': result.get('sequence_id', -1),
                    'abs_rel': result.get('abs_rel', 0.0),
                    'depth_ranges': result['_depth_range_analysis']
                }
                depth_range_analysis.append(entry)

        if depth_range_analysis:
            aggregated_ranges = {}
            for range_name in ['0-10m', '10-30m', '30-70m']:
                range_abs_rels = [e['depth_ranges'].get(range_name, {}).get('abs_rel', 0) for e in depth_range_analysis]
                range_a1s = [e['depth_ranges'].get(range_name, {}).get('a1', 0) for e in depth_range_analysis]
                range_pixels = [e['depth_ranges'].get(range_name, {}).get('pixel_count', 0) for e in depth_range_analysis]
                aggregated_ranges[range_name] = {
                    'abs_rel': float(np.mean(range_abs_rels)) if range_abs_rels else 0.0,
                    'a1': float(np.mean(range_a1s)) if range_a1s else 0.0,
                    'total_pixels': int(np.sum(range_pixels))
                }

            result = {
                'aggregated': aggregated_ranges,
                'per_sequence': depth_range_analysis
            }
            with open(self.save_dir / "depth_range_analysis.json", 'w') as f:
                json.dump(result, f, indent=2, default=str)
            logger.info(f"Depth range analysis saved to {self.save_dir / 'depth_range_analysis.json'}")

    def _save_temporal_analysis(self, all_metrics):
        """Save temporal_analysis.json."""
        temporal_analysis = []
        for result in all_metrics:
            entry = {
                'sequence_id': result.get('sequence_id', -1),
                'tae_reproj': result.get('tae_reproj', 0.0),
                'tae_reproj_gt': result.get('tae_reproj_gt', 0.0),
                'tae': result.get('tae', 0.0),
                'per_frame_tae': result.get('_per_frame_tae', []),
                'tae_spike_frames': result.get('_tae_spike_frames', []),
                'tae_spike_count': len(result.get('_tae_spike_frames', []))
            }
            temporal_analysis.append(entry)

        if temporal_analysis:
            with open(self.save_dir / "temporal_analysis.json", 'w') as f:
                json.dump(temporal_analysis, f, indent=2, default=str)
            logger.info(f"Temporal analysis saved to {self.save_dir / 'temporal_analysis.json'}")

    def _save_temporal_consistency(self, all_metrics):
        """Save temporal_consistency.json (flow-based rTC metrics)."""
        has_rtc = any(m.get('rtc', 0) > 0 for m in all_metrics)
        if not has_rtc:
            return

        per_sequence = []
        for result in all_metrics:
            entry = {
                'sequence_id': result.get('sequence_id', -1),
                'rtc': result.get('rtc', 0.0),
                'rtc_gt': result.get('rtc_gt', 0.0),
                'per_frame_rtc': result.get('_per_frame_rtc', []),
                'per_frame_rtc_gt': result.get('_per_frame_rtc_gt', []),
                'per_frame_ratio_stats': result.get('_rtc_per_frame_ratio_stats', []),
                'ratio_stats': result.get('_rtc_ratio_stats', {}),
                'best_frame_idx': result.get('_rtc_best_frame_idx', 0),
                'worst_frame_idx': result.get('_rtc_worst_frame_idx', 0)
            }
            # SCD-excluded rTC (if available)
            if 'rtc_excl_scd' in result:
                entry['rtc_excl_scd'] = result['rtc_excl_scd']
                entry['rtc_gt_excl_scd'] = result.get('rtc_gt_excl_scd', 0.0)
                entry['excl_scd_num_excluded'] = result.get('_excl_scd_num_excluded', 0)
                entry['excl_scd_num_remaining'] = result.get('_excl_scd_num_remaining', 0)
                entry['per_frame_rtc_excl_scd'] = result.get('_per_frame_rtc_excl_scd', [])
            per_sequence.append(entry)

        # Aggregated
        rtc_values = [m.get('rtc', 0.0) for m in all_metrics if m.get('rtc', 0) > 0]
        rtc_gt_values = [m.get('rtc_gt', 0.0) for m in all_metrics if m.get('rtc_gt', 0) > 0]

        # Aggregate ratio stats
        all_ratio_stats = [m.get('_rtc_ratio_stats', {}) for m in all_metrics if m.get('_rtc_ratio_stats')]
        agg_ratio_stats = {}
        if all_ratio_stats:
            for key in ['avg', 'min', 'max', 'p90', 'p95']:
                values = [rs.get(key, 0.0) for rs in all_ratio_stats if key in rs]
                if values:
                    agg_ratio_stats[key] = float(np.mean(values))

        # Aggregate excl_scd rTC
        rtc_excl_scd_values = [m['rtc_excl_scd'] for m in all_metrics if 'rtc_excl_scd' in m]
        rtc_gt_excl_scd_values = [m['rtc_gt_excl_scd'] for m in all_metrics if 'rtc_gt_excl_scd' in m]

        aggregated = {
            'rtc': float(np.mean(rtc_values)) if rtc_values else 0.0,
            'rtc_gt': float(np.mean(rtc_gt_values)) if rtc_gt_values else 0.0,
            'ratio_stats': agg_ratio_stats
        }
        if rtc_excl_scd_values:
            aggregated['rtc_excl_scd'] = float(np.mean(rtc_excl_scd_values))
            aggregated['rtc_gt_excl_scd'] = float(np.mean(rtc_gt_excl_scd_values)) if rtc_gt_excl_scd_values else 0.0

        tc_output = {
            'config': {'threshold': self.tc_threshold, 'flow_model': 'sea_raft'},
            'aggregated': aggregated,
            'per_sequence': per_sequence
        }

        tc_path = self.save_dir / "temporal_consistency.json"
        with open(tc_path, 'w') as f:
            json.dump(tc_output, f, indent=2, default=str)
        logger.info(f"Temporal consistency saved to {tc_path}")

    def _save_tc_summary(self, all_metrics):
        """Save tc_summary.json with rTC + TAE + PSR (dataset aggregate + per-sequence)."""
        per_sequence = []
        for r in all_metrics:
            entry = {
                'sequence_id': r.get('sequence_id', -1),
                'rtc': r.get('rtc', 0.0),
                'rtc_gt': r.get('rtc_gt', 0.0),
                'tae': r.get('tae', 0.0),
                'tae_reproj': r.get('tae_reproj', 0.0),
                'tae_reproj_gt': r.get('tae_reproj_gt', 0.0),
                'psr': r.get('psr', 0.0),
                'psr_max': r.get('psr_max', 0.0),
                'fps': r.get('fps', 0.0),
            }
            if 'rtc_excl_scd' in r:
                entry['rtc_excl_scd'] = r['rtc_excl_scd']
                entry['rtc_gt_excl_scd'] = r.get('rtc_gt_excl_scd', 0.0)
            per_sequence.append(entry)

        # Aggregate (mean over sequences with valid values)
        agg = {}
        for key in ['rtc', 'rtc_gt', 'tae', 'tae_reproj', 'tae_reproj_gt', 'psr', 'psr_max', 'fps']:
            vals = [s[key] for s in per_sequence if s[key] != 0.0]
            agg[key] = float(np.mean(vals)) if vals else 0.0
        # Aggregate excl_scd
        excl_scd_vals = [s['rtc_excl_scd'] for s in per_sequence if 'rtc_excl_scd' in s]
        if excl_scd_vals:
            agg['rtc_excl_scd'] = float(np.mean(excl_scd_vals))

        summary = {
            'aggregated': agg,
            'per_sequence': per_sequence,
        }

        path = self.save_dir / "tc_summary.json"
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"TC summary saved to {path}")

    def _aggregate_metrics(self, all_metrics):
        """Aggregate metrics across sequences with metric_order."""
        metric_keys = set()
        for m in all_metrics:
            metric_keys.update(m.keys())

        aggregated_raw = {}
        for key in metric_keys:
            if key in ('sequence_id',) or key.startswith('_'):
                continue
            values = [m[key] for m in all_metrics if key in m]
            if values and all(isinstance(v, (int, float, np.number)) for v in values):
                aggregated_raw[key] = float(np.mean(values))

        # Reorder
        metric_order = ['abs_rel', 'a1', 'a2', 'a3', 'fps', 'tae', 'tae_reproj', 'tae_reproj_gt',
                        'rtc', 'rtc_gt', 'rtc_excl_scd', 'rtc_gt_excl_scd',
                        'psr', 'psr_max',
                        'mae', 'rmse',
                        'pred_scale_mean', 'pred_shift_mean',
                        'optimal_scale_mean', 'optimal_shift_mean']
        aggregated = {}
        for key in metric_order:
            if key in aggregated_raw:
                aggregated[key] = aggregated_raw[key]
        for key, value in aggregated_raw.items():
            if key not in aggregated:
                aggregated[key] = value

        # Per-dataset breakdown
        dataset_metrics = {}
        for m in all_metrics:
            ds = m.get('dataset', 'unknown')
            if ds not in dataset_metrics:
                dataset_metrics[ds] = []
            dataset_metrics[ds].append(m)

        aggregated['per_dataset'] = {}
        for ds, metrics_list in dataset_metrics.items():
            ds_result = {}
            for key in metric_order:
                values = [m[key] for m in metrics_list if key in m and isinstance(m[key], (int, float, np.number))]
                if values:
                    ds_result[key] = float(np.mean(values))
            ds_result['num_sequences'] = len(metrics_list)
            aggregated['per_dataset'][ds] = ds_result

        aggregated['num_sequences'] = len(all_metrics)

        return aggregated


@hydra.main(version_base=None, config_path="configs/onepiece", config_name="config")
def main(config: DictConfig):
    """Main entry point."""
    # Apply --test-mode if passed via sys.argv preprocessing
    test_mode = getattr(main, '_test_mode', None)
    if test_mode:
        OmegaConf.update(config, 'test_mode', test_mode, force_add=True)

    # Apply --frame if passed via sys.argv preprocessing (supports comma-separated: --frame 26,80)
    frame_arg = getattr(main, '_frame_arg', None)
    if frame_arg is not None:
        frame_indices = [int(f.strip()) for f in frame_arg.split(',')]
        if len(frame_indices) == 1:
            OmegaConf.update(config, 'frame', frame_indices[0], force_add=True)
        else:
            OmegaConf.update(config, 'frame', frame_indices, force_add=True)

    # Apply --max-depth if passed via sys.argv preprocessing
    max_depth_arg = getattr(main, '_max_depth_arg', None)
    if max_depth_arg is not None:
        OmegaConf.update(config, 'max_depth', float(max_depth_arg), force_add=True)

    # Apply --save-depth-maps if passed via sys.argv preprocessing
    save_depth_maps = getattr(main, '_save_depth_maps', False)
    if save_depth_maps:
        OmegaConf.update(config, 'save_depth_maps', True, force_add=True)

    # Apply --no-scd if passed via sys.argv preprocessing
    no_scd = getattr(main, '_no_scd', False)
    if no_scd:
        OmegaConf.update(config, 'no_scd', True, force_add=True)

    tester = OnepieceTester(config)
    tester.test()


if __name__ == "__main__":
    import sys

    # Handle --test-mode, --frame, --max-depth, --save-depth-maps flags BEFORE Hydra processes arguments
    test_mode = None
    frame_arg = None
    max_depth_arg = None
    save_depth_maps = False
    no_scd = False
    new_argv = []
    i = 0
    while i < len(sys.argv):
        if sys.argv[i] == '--test-mode' and i + 1 < len(sys.argv):
            test_mode = sys.argv[i + 1]
            i += 2
        elif sys.argv[i].startswith('--test-mode='):
            test_mode = sys.argv[i].split('=', 1)[1]
            i += 1
        elif sys.argv[i] == '--frame' and i + 1 < len(sys.argv):
            frame_arg = sys.argv[i + 1]
            i += 2
        elif sys.argv[i].startswith('--frame='):
            frame_arg = sys.argv[i].split('=', 1)[1]
            i += 1
        elif sys.argv[i] == '--max-depth' and i + 1 < len(sys.argv):
            max_depth_arg = sys.argv[i + 1]
            i += 2
        elif sys.argv[i].startswith('--max-depth='):
            max_depth_arg = sys.argv[i].split('=', 1)[1]
            i += 1
        elif sys.argv[i] == '--save-depth-maps':
            save_depth_maps = True
            i += 1
        elif sys.argv[i] == '--no-scd':
            no_scd = True
            i += 1
        else:
            new_argv.append(sys.argv[i])
            i += 1
    sys.argv = new_argv

    # Store as function attributes so main() can access them
    main._test_mode = test_mode
    main._frame_arg = frame_arg
    main._max_depth_arg = max_depth_arg
    main._save_depth_maps = save_depth_maps
    main._no_scd = no_scd

    main()
