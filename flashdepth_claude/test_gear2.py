#!/usr/bin/env python3
"""
Test script for Gear2: Ablation Study (No FG/BG Separation)

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
from flashdepth.gear2_modules import Gear2MetricHead
from dataloaders.combined_dataset import CombinedDataset
from dataloaders.sintel_segmentation_dataset import SintelSegmentationDataset, collate_fn as sintel_collate_fn
from dataloaders.waymo_segmentation_dataset import WaymoSegmentationDataset, collate_fn as waymo_collate_fn
from utils.metric_depth_metrics import MetricDepthMetrics, format_metrics
from utils.object_wise_evaluation import ObjectWiseMetrics
from utils.object_wise_visualization import create_object_wise_grid
from utils.helpers import save_gifs_as_grid, save_grid_to_mp4, depth_to_np_arr, torch_batch_to_np_arr

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Gear2Tester:
    """
    Test harness for Gear2 model.

    Evaluates on:
        - Inverse depth metrics (TAE, AbsRel, δ1/δ2/δ3)
        - Metric depth visualization (no relative depth)
        - Importance map visualization
    """
    def __init__(self, config):
        self.config = config
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Setup save directory - use results_dir if provided, otherwise use eval.outfolder
        save_dir_str = config.get('results_dir', config.eval.outfolder)
        self.save_dir = Path(save_dir_str)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Save directory: {self.save_dir}")

        # Object-wise evaluation configuration
        self.object_wise_enabled = config.get('object_wise', {}).get('enabled', False)
        self.object_wise_dataset = config.get('object_wise', {}).get('dataset', 'waymo')

        if self.object_wise_enabled:
            logger.info(f"Object-wise evaluation ENABLED for dataset: {self.object_wise_dataset}")
            self.object_wise_metrics = ObjectWiseMetrics(dataset_type=self.object_wise_dataset)
        else:
            self.object_wise_metrics = None

        # Initialize model
        self.model = self._setup_model()

        # Setup test loader
        self.test_loader = self._setup_test_loader()

        # Setup metrics
        self.metrics = MetricDepthMetrics()

    def _setup_model(self):
        """Load trained Gear2 model"""
        # Create base FlashDepth model
        model_config = dict(self.config.model)
        model_config['batch_size'] = 1
        model_config['use_metric_head'] = False

        model = FlashDepth(**model_config)

        # Add Gear2 metric head
        embed_dim = 1024 if model.encoder == 'vitl' else 384
        dpt_dim = 256 if model.encoder == 'vitl' else 64

        model.gear2_head = Gear2MetricHead(
            embed_dim=embed_dim,
            dpt_dim=dpt_dim
        )

        # Enable attention weights storage ONLY for last block (like train_gear2)
        for i, block in enumerate(model.pretrained.blocks):
            if i == len(model.pretrained.blocks) - 1:
                block.attn.store_attn_weights = True
                logger.info(f"Enabled attention weights storage for block {i} (last block)")
            else:
                block.attn.store_attn_weights = False

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
        # Check if single sequence mode
        single_seq_path = self.config.get('single_sequence', None)

        # Object-wise evaluation: use segmentation datasets
        if self.object_wise_enabled:
            video_length = int(self.config.get('vid_len', 50))  # Ensure integer (Hydra may pass string)
            resolution = self.config.eval.test_dataset_resolution
            data_root = self.config.dataset.data_root

            if self.object_wise_dataset == 'sintel':
                test_dataset = SintelSegmentationDataset(
                    data_root=data_root,
                    split='val',
                    video_length=video_length,
                    resolution=resolution
                )
                collate_fn = sintel_collate_fn
            elif self.object_wise_dataset == 'waymo':
                test_dataset = WaymoSegmentationDataset(
                    data_root=data_root,
                    split='val',
                    video_length=video_length,
                    resolution=resolution,
                    camera_name=1,  # FRONT camera
                    use_depth=False  # Use placeholder depth (complex to extract from LiDAR)
                )
                collate_fn = waymo_collate_fn
            else:
                raise ValueError(f"Unknown object-wise dataset: {self.object_wise_dataset}")

            logger.info(f"Object-wise dataset: {self.object_wise_dataset} (size: {len(test_dataset)})")

        elif single_seq_path:
            # Single sequence mode: create custom dataset
            logger.info(f"Single sequence mode: {single_seq_path}")
            test_dataset = self._create_single_sequence_dataset(single_seq_path)
            collate_fn = self._collate_fn
        else:
            # Normal mode: use CombinedDataset
            test_datasets = self.config.eval.test_datasets
            video_length = int(self.config.get('vid_len', 50))  # Ensure integer (Hydra may pass string)

            logger.info(f"Test datasets: {test_datasets}")
            logger.info(f"Video length: {video_length}")

            test_dataset = CombinedDataset(
                root_dir=self.config.dataset.data_root,
                enable_dataset_flags=test_datasets,
                resolution=self.config.eval.test_dataset_resolution,
                split='val',  # Use 'val' split which returns dict format
                video_length=video_length
            )
            collate_fn = self._collate_fn

        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,  # Single process for testing
            pin_memory=True,
            collate_fn=collate_fn
        )

        logger.info(f"Test dataset size: {len(test_dataset)}")

        return test_loader

    def _create_single_sequence_dataset(self, sequence_path):
        """Create a dataset from a single sequence directory"""
        from torch.utils.data import Dataset
        import glob
        from PIL import Image

        class SingleSequenceDataset(Dataset):
            def __init__(self, seq_path, resolution=518):
                self.seq_path = Path(seq_path)
                # Ensure resolution is an integer (config may pass 'base' string)
                if isinstance(resolution, str):
                    if resolution == 'base':
                        self.resolution = 518
                    else:
                        self.resolution = int(resolution)
                else:
                    self.resolution = int(resolution)

                # Find images and depths
                images_dir = self.seq_path / "images"
                depths_dir = self.seq_path / "depths"

                if not images_dir.exists():
                    raise FileNotFoundError(f"Images directory not found: {images_dir}")
                if not depths_dir.exists():
                    raise FileNotFoundError(f"Depths directory not found: {depths_dir}")

                # Get sorted image files
                self.image_files = sorted(glob.glob(str(images_dir / "*.png")))

                # Determine depth file pattern
                if len(self.image_files) > 0:
                    img_name = Path(self.image_files[0]).stem
                    # Try .geometric.png pattern (dynamicreplica)
                    depth_pattern = str(depths_dir / f"{img_name}_*.geometric.png")
                    depth_files = glob.glob(depth_pattern)

                    if len(depth_files) == 0:
                        # Try direct pattern matching
                        depth_files = sorted(glob.glob(str(depths_dir / "*.geometric.png")))

                    self.depth_files = sorted(depth_files)
                else:
                    self.depth_files = []

                assert len(self.image_files) > 0, f"No images found in {images_dir}"
                assert len(self.depth_files) > 0, f"No depth files found in {depths_dir}"
                assert len(self.image_files) == len(self.depth_files), \
                    f"Mismatch: {len(self.image_files)} images vs {len(self.depth_files)} depths"

                logger.info(f"Loaded single sequence: {len(self.image_files)} frames")

            def __len__(self):
                return 1  # Single sequence

            def __getitem__(self, idx):
                # Load all frames
                images = []
                depths = []

                for img_path, depth_path in zip(self.image_files, self.depth_files):
                    # Load image
                    img = Image.open(img_path).convert('RGB')
                    # Use integer constants for PIL compatibility (2=BILINEAR, 0=NEAREST)
                    img = img.resize((self.resolution, self.resolution), 2)  # 2 = BILINEAR (positional arg)
                    img_array = np.array(img).astype(np.float32) / 255.0
                    # Normalize (ImageNet stats)
                    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                    img_normalized = (img_array - mean) / std
                    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).float()  # [3, H, W], ensure float32
                    images.append(img_tensor)

                    # Load depth (.geometric.png for dynamicreplica)
                    depth_img = Image.open(depth_path)
                    depth_array = np.array(depth_img).astype(np.float32) / 1000.0  # mm to m
                    # Resize depth
                    depth_pil = Image.fromarray(depth_array)
                    depth_resized = depth_pil.resize((self.resolution, self.resolution), 0)  # 0 = NEAREST (positional arg)
                    depth_array = np.array(depth_resized)
                    # Convert to inverse depth (1/m) like CombinedDataset
                    inverse_depth = np.zeros_like(depth_array)
                    valid_mask = depth_array > 0
                    inverse_depth[valid_mask] = 1.0 / depth_array[valid_mask]
                    depth_tensor = torch.from_numpy(inverse_depth)  # [H, W]
                    depths.append(depth_tensor)

                # Stack
                images = torch.stack(images, dim=0)  # [T, 3, H, W]
                depths = torch.stack(depths, dim=0)  # [T, H, W]

                return images, depths, "single_sequence"

        return SingleSequenceDataset(sequence_path, resolution=self.config.eval.test_dataset_resolution)

    def _collate_fn(self, batch):
        """Custom collate function to filter out None values and convert tuple to dict"""
        # Filter out None values
        batch = [item for item in batch if item is not None]

        # If all items are None, skip this batch
        if len(batch) == 0:
            return None

        # CombinedDataset returns (images, depths, dataset_name) tuple for val/test splits
        # Convert to dict format for easier access
        if len(batch) > 0 and isinstance(batch[0], tuple):
            images, depths, names = zip(*batch)
            return {
                'image': torch.stack(images, dim=0),
                'depth': torch.stack(depths, dim=0),
                'dataset_name': names
            }

        # Segmentation datasets return dict with 'images' key, rename to 'image' for compatibility
        if len(batch) > 0 and isinstance(batch[0], dict) and 'images' in batch[0]:
            # Already batched by segmentation collate_fn
            result = batch[0]  # Batch size is 1
            if 'images' in result:
                result['image'] = result.pop('images')  # Rename 'images' -> 'image'
            return result

        # Use default collate for dict items (training split)
        return torch.utils.data.dataloader.default_collate(batch)

    @torch.no_grad()
    def test(self):
        """Main testing loop"""
        logger.info("Starting testing...")

        all_metrics = []
        all_object_wise_metrics = []  # Track object-wise metrics separately
        sequence_id = 0

        for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Testing")):
            try:
                metrics = self.test_sequence(batch, sequence_id)
                # Add sequence_id for tracking
                metrics['sequence_id'] = sequence_id
                all_metrics.append(metrics)

                # Extract and store object-wise metrics
                if self.object_wise_enabled and 'object_wise' in metrics:
                    all_object_wise_metrics.append(metrics['object_wise'])

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

            # Save overall results
            results_path = self.save_dir / "test_results.json"
            with open(results_path, 'w') as f:
                json.dump(avg_metrics, f, indent=2)
            logger.info(f"Results saved to {results_path}")

            # Save per-sequence results
            per_sequence_path = self.save_dir / "per_sequence_results.json"
            with open(per_sequence_path, 'w') as f:
                json.dump(all_metrics, f, indent=2)
            logger.info(f"Per-sequence results saved to {per_sequence_path}")

            # Find and save best sequence (lowest abs_rel)
            best_seq = min(all_metrics, key=lambda x: x['abs_rel'])
            best_seq_path = self.save_dir / "best_sequence.json"
            with open(best_seq_path, 'w') as f:
                json.dump(best_seq, f, indent=2)
            logger.info(f"\nBest sequence (lowest AbsRel): Sequence {best_seq['sequence_id']}")
            logger.info(f"  AbsRel: {best_seq['abs_rel']:.4f}")
            logger.info(f"  MAE: {best_seq['mae']:.4f}")
            logger.info(f"  RMSE: {best_seq['rmse']:.4f}")
            logger.info(f"  δ1: {best_seq['a1']:.4f}")
            logger.info(f"Best sequence saved to {best_seq_path}")

            # Aggregate and save object-wise metrics
            if self.object_wise_enabled and all_object_wise_metrics:
                logger.info("\n" + "="*80)
                logger.info("OBJECT-WISE EVALUATION RESULTS")
                logger.info("="*80)

                # Aggregate metrics across all sequences
                aggregated_class_metrics = self.object_wise_metrics.aggregate_metrics(all_object_wise_metrics)

                # Print summary
                self.object_wise_metrics.print_summary(aggregated_class_metrics)

                # Save to JSON
                object_wise_path = self.save_dir / "object_wise_results.json"
                self.object_wise_metrics.save_results(
                    aggregated_class_metrics,
                    object_wise_path
                )

        else:
            logger.warning("No metrics computed!")

    @torch.no_grad()
    def test_sequence(self, batch, sequence_id):
        """Test on a single sequence"""
        # Debug: Check batch type and structure
        if not isinstance(batch, dict):
            logger.error(f"Batch is not a dict! Type: {type(batch)}, Content: {batch if not isinstance(batch, torch.Tensor) else 'Tensor'}")
            raise TypeError(f"Expected dict, got {type(batch)}")

        # Preload all data to GPU memory (논문 방법: FPS 측정 시 데이터 전송 시간 제외)
        # "We preprocess all images and load them onto GPU memory before starting inference"
        images = batch['image'].to(self.device)  # [1, T, 3, H, W] - 전체 시퀀스를 GPU에 미리 로드
        gt_depth = batch['depth'].to(self.device)  # [1, T, H, W]

        # Add channel dimension if needed
        if gt_depth.ndim == 3:
            gt_depth = gt_depth.unsqueeze(2)  # [1, T, 1, H, W]
        elif gt_depth.ndim == 4 and gt_depth.shape[2] > 3:
            # Already [1, T, H, W], add channel dim
            gt_depth = gt_depth.unsqueeze(2)  # [1, T, 1, H, W]

        B, T = images.shape[:2]
        assert B == 1, "Batch size must be 1 for testing"

        # Dataloader gives inverse depth (1/m), scale to 100/m for training
        gt_depth_inverse_100 = gt_depth * 100.0  # [1, T, 1, H, W] in 100/m

        # Storage for predictions
        pred_depths = []
        importance_maps = []
        fg_features_list = []
        bg_features_list = []

        # Best frame tracking
        best_frame_idx = 0
        best_frame_abs_rel = float('inf')

        # Warmup run for FPS measurement
        logger.info(f"Warmup run for FPS measurement...")

        # Initialize Mamba sequence for warmup
        if hasattr(self.model, 'mamba'):
            self.model.mamba.start_new_sequence()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            img_warmup = images[0, 0].unsqueeze(0)  # Already on GPU
            encoder_features_warmup = self.model.pretrained.get_intermediate_layers(
                img_warmup, self.model.intermediate_layer_idx[self.model.encoder]
            )
            last_block = self.model.pretrained.blocks[-1]
            attention_weights_warmup = last_block.attn.attn_weights
            patch_tokens_warmup = encoder_features_warmup[-1]
            h_warmup, w_warmup = img_warmup.shape[2:]
            patch_h_warmup = h_warmup // self.model.patch_size
            patch_w_warmup = w_warmup // self.model.patch_size

            # Use forward_with_mamba for temporal processing
            B_warmup, T_warmup = 1, T  # Batch size 1 for warmup
            dpt_output_warmup = self.model.depth_head.forward_with_mamba(
                encoder_features_warmup, patch_h_warmup, patch_w_warmup,
                temporal_layer=self.model.mamba_in_dpt_layer,
                mamba_fn=self.model.dpt_features_to_mamba,
                shape_placeholder=(B_warmup, T_warmup, None, h_warmup, w_warmup)
            )

            # Wrap in list for Gear2 head compatibility
            path_1_warmup, _, _, _ = self.model.gear2_head(
                patch_tokens_warmup, attention_weights_warmup, [dpt_output_warmup], patch_h_warmup, patch_w_warmup
            )
            out_warmup = self.model.depth_head.scratch.output_conv1(path_1_warmup)
            out_warmup = F.interpolate(out_warmup, (h_warmup, w_warmup), mode="bilinear", align_corners=True)
            _ = self.model.depth_head.scratch.output_conv2(out_warmup)
        del encoder_features_warmup, attention_weights_warmup, patch_tokens_warmup, dpt_output_warmup
        del path_1_warmup, out_warmup
        torch.cuda.empty_cache()

        # FPS measurement (논문 방법: 데이터가 GPU에 미리 로드된 상태에서 순수 inference 시간만 측정)
        warmup_frames = min(5, T)  # Warmup frames to skip initial overhead
        start_time = None  # Will start timing after warmup

        # Initialize Mamba sequence for actual test (critical for temporal processing!)
        if hasattr(self.model, 'mamba'):
            self.model.mamba.start_new_sequence()

        # Process each frame
        for t in range(T):
            # Start timing after warmup frames (논문: "start timer after a single warmup iteration")
            if t == warmup_frames:
                torch.cuda.synchronize()
                import time
                start_time = time.time()

            # GPU에서 인덱싱만 (전송 없음, 데이터는 이미 GPU에 로드됨)
            img_t = images[0, t]  # [3, H, W] - GPU에서 인덱싱만
            gt_t_inverse = gt_depth_inverse_100[0, t]  # [1, H, W] - GPU에서 인덱싱만

            # Use BFloat16 for forward pass (same as train_gear2.py)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # Extract features from DINOv2
                encoder_features = self.model.pretrained.get_intermediate_layers(
                    img_t.unsqueeze(0), self.model.intermediate_layer_idx[self.model.encoder]
                )

                # Get attention weights from last block
                last_block = self.model.pretrained.blocks[-1]
                attention_weights = last_block.attn.attn_weights

                # Get patch tokens from last encoder layer
                patch_tokens = encoder_features[-1]

                # Get DPT features with Mamba temporal processing
                h, w = img_t.shape[1:]
                patch_h, patch_w = h // self.model.patch_size, w // self.model.patch_size
                B_test, T_test = 1, T  # Batch size 1 for test
                dpt_output = self.model.depth_head.forward_with_mamba(
                    encoder_features, patch_h, patch_w,
                    temporal_layer=self.model.mamba_in_dpt_layer,
                    mamba_fn=self.model.dpt_features_to_mamba,
                    shape_placeholder=(B_test, T_test, None, h, w)
                )  # Returns path_1 with Mamba applied

                # Apply Gear2 modulation (returns None for importance_map, fg_features, bg_features)
                # Wrap dpt_output in list for compatibility
                path_1_modulated, importance_map, fg_features, bg_features = self.model.gear2_head(
                    patch_tokens, attention_weights, [dpt_output], patch_h, patch_w
                )

                # Get depth prediction (output is inverse depth in 100/m scale)
                # path_1_modulated is already the modulated feature, no need to index
                out = self.model.depth_head.scratch.output_conv1(path_1_modulated)
                out = F.interpolate(out, (h, w), mode="bilinear", align_corners=True)
                out = self.model.depth_head.scratch.output_conv2(out)  # [1, 1, H, W]

                # Prediction is already positive (Softplus activation in output_conv2)
                pred_depth_inverse_100 = out  # [1, 1, H, W] in 100/m

                # Interpolate prediction to GT resolution (like train_gear2.py validation)
                gt_t_shape = gt_t_inverse.shape[-2:]  # GT original resolution
                if pred_depth_inverse_100.shape[-2:] != gt_t_shape:
                    pred_depth_inverse_100 = F.interpolate(
                        pred_depth_inverse_100, size=gt_t_shape, mode="bilinear", align_corners=True
                    )

                # Convert to metric depth: 100/m -> m
                pred_depth_metric = 100.0 / (pred_depth_inverse_100[0] + 1e-8)  # [1, H, W]

            # End timing for FPS measurement (after last frame, like original FlashDepth)
            if t == T - 1 and start_time is not None:
                torch.cuda.synchronize()
                end_time = time.time()

            # List append for visualization (outside FPS measurement)
            pred_depths.append(pred_depth_metric)
            # Gear2 returns None for importance_map, fg_features, bg_features
            if importance_map is not None:
                importance_maps.append(importance_map[0])
                fg_features_list.append(fg_features[0])
                bg_features_list.append(bg_features[0])

        # Calculate FPS (like original FlashDepth: exclude warmup frames)
        if start_time is not None:
            inference_time = end_time - start_time
            fps = (T - warmup_frames) / inference_time if inference_time > 0 else 0
            logger.info(f"Inference time: {inference_time:.4f}s for {T - warmup_frames} frames (warmup {warmup_frames} excluded)")
            logger.info(f"FPS: {fps:.2f} frames/second")
        else:
            # Too few frames for FPS measurement
            fps = 0
            logger.warning(f"Too few frames ({T}) for FPS measurement (need > {warmup_frames})")

        # Stack predictions
        pred_depths = torch.stack(pred_depths, dim=0)  # [T, 1, H, W] in meters
        # Gear2 doesn't produce importance maps (returns None)
        importance_maps = torch.stack(importance_maps, dim=0) if importance_maps else None
        fg_features_all = torch.stack(fg_features_list, dim=0) if fg_features_list else None
        bg_features_all = torch.stack(bg_features_list, dim=0) if bg_features_list else None

        # Convert GT to metric depth for evaluation: 100/m -> m
        gt_depth_metric = 100.0 / (gt_depth_inverse_100[0] + 1e-8)  # [T, 1, H, W] in meters

        # Compute metrics (both pred and GT are now in meters)
        # Move to CPU and compute per-frame metrics (like test_metric_head.py)
        pred_depths_cpu = pred_depths.cpu()
        gt_depth_metric_cpu = gt_depth_metric.cpu()

        frame_metrics = []
        for t in range(pred_depths.shape[0]):
            # Get individual frames (already on CPU)
            pred_frame = pred_depths_cpu[t, 0]  # [H, W]
            gt_frame = gt_depth_metric_cpu[t, 0]  # [H, W]

            # Create valid mask for this frame (like train_gear2 validation)
            # Use same MAX_DEPTH as Gear2Visualizer (70m)
            MAX_DEPTH = 70.0
            gt_valid_mask = (gt_frame > 0) & (gt_frame < MAX_DEPTH)  # GT valid pixels
            pred_valid_mask = (pred_frame > 0) & (pred_frame < MAX_DEPTH)  # Filter extreme values
            valid_mask = gt_valid_mask & pred_valid_mask  # [H, W] bool tensor

            # Debug logging for first frame of first sequence
            if t == 0 and sequence_id == 0:
                logger.info(f"DEBUG Metrics - Frame {t}")
                logger.info(f"  GT depth range: [{gt_frame.min():.2f}, {gt_frame.max():.2f}] meters")
                logger.info(f"  Pred depth range: [{pred_frame.min():.2f}, {pred_frame.max():.2f}] meters")
                logger.info(f"  Valid pixels: {valid_mask.sum()} / {valid_mask.numel()} ({100*valid_mask.sum()/valid_mask.numel():.1f}%)")
                if valid_mask.sum() > 0:
                    gt_valid_values = gt_frame[valid_mask]
                    pred_valid_values = pred_frame[valid_mask]
                    logger.info(f"  GT valid range: [{gt_valid_values.min():.2f}, {gt_valid_values.max():.2f}]")
                    logger.info(f"  Pred valid range: [{pred_valid_values.min():.2f}, {pred_valid_values.max():.2f}]")
                    mae = torch.abs(pred_valid_values - gt_valid_values).mean()
                    logger.info(f"  MAE: {mae:.4f} meters")

            if valid_mask.sum() > 0:
                frame_metric = self.metrics.compute_metric_depth_metrics(
                    pred_frame,  # [H, W]
                    gt_frame,   # [H, W]
                    valid_mask  # [H, W]
                )
                frame_metrics.append(frame_metric)

                # Track best frame (lowest abs_rel)
                if frame_metric['abs_rel'] < best_frame_abs_rel:
                    best_frame_abs_rel = frame_metric['abs_rel']
                    best_frame_idx = t

        # Average metrics across frames
        if len(frame_metrics) == 0:
            logger.warning(f"No valid frames for sequence {sequence_id}")
            return {k: 0.0 for k in ["mae", "rmse", "abs_rel", "a1", "tae", "fps"]}

        metrics = {}
        for key in frame_metrics[0].keys():
            values = [m[key] for m in frame_metrics]
            metrics[key] = np.mean(values)

        # Compute TAE (Temporal Alignment Error) - sequence-level metric
        # TAE measures frame-to-frame consistency
        if len(pred_depths) > 1:
            tae_errors = []
            for t in range(len(pred_depths) - 1):
                pred_t = pred_depths_cpu[t, 0]  # [H, W]
                pred_t_next = pred_depths_cpu[t + 1, 0]  # [H, W]
                gt_t = gt_depth_metric_cpu[t, 0]  # [H, W]
                gt_t_next = gt_depth_metric_cpu[t + 1, 0]  # [H, W]

                # Valid mask for both frames
                MAX_DEPTH = 70.0
                valid_t = (gt_t > 0) & (gt_t < MAX_DEPTH) & (pred_t > 0) & (pred_t < MAX_DEPTH)
                valid_t_next = (gt_t_next > 0) & (gt_t_next < MAX_DEPTH) & (pred_t_next > 0) & (pred_t_next < MAX_DEPTH)
                valid_both = valid_t & valid_t_next  # [H, W]

                if valid_both.sum() > 0:
                    # Compute depth change (temporal derivative)
                    pred_change = pred_t_next - pred_t  # [H, W]
                    gt_change = gt_t_next - gt_t  # [H, W]

                    # TAE: mean absolute error in temporal change
                    tae = torch.abs(pred_change[valid_both] - gt_change[valid_both]).mean()
                    tae_errors.append(tae.item())

            metrics['tae'] = np.mean(tae_errors) if len(tae_errors) > 0 else 0.0
        else:
            metrics['tae'] = 0.0

        # Add FPS to metrics
        metrics['fps'] = fps

        # Object-wise evaluation: compute per-class metrics
        if self.object_wise_enabled and 'segmentation' in batch:
            try:
                # Get segmentation mask for last frame
                seg_mask = batch['segmentation'][0]  # [H, W] - batch size is 1

                # Get last frame predictions and GT
                pred_last = pred_depths_cpu[-1, 0].numpy()  # [H, W] numpy array
                gt_last = gt_depth_metric_cpu[-1, 0].numpy()  # [H, W] numpy array
                seg_mask_np = seg_mask.cpu().numpy() if isinstance(seg_mask, torch.Tensor) else seg_mask

                # Resize segmentation to match pred/GT resolution if needed
                if seg_mask_np.shape != pred_last.shape:
                    import cv2
                    seg_mask_np = cv2.resize(
                        seg_mask_np.astype(np.int32),
                        (pred_last.shape[1], pred_last.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )

                # Compute per-class metrics
                class_metrics = self.object_wise_metrics.compute_metrics_per_class(
                    pred_depth=pred_last,
                    gt_depth=gt_last,
                    seg_mask=seg_mask_np,
                    min_pixels=100
                )

                metrics['object_wise'] = class_metrics
                logger.info(f"Computed object-wise metrics for {len(class_metrics)} classes")

                # Create object-wise visualization
                if class_metrics:
                    try:
                        # Get last frame image (convert to numpy HWC format)
                        img_last = images[0, -1].cpu().numpy()  # [3, H, W]
                        img_last = img_last.transpose(1, 2, 0)  # [H, W, 3]

                        # Normalize to [0, 1] if needed
                        if img_last.max() > 1.0:
                            img_last = img_last / 255.0

                        # Create visualization
                        objwise_vis_path = self.save_dir / f"objwise_seq{sequence_id:04d}.png"
                        create_object_wise_grid(
                            input_image=img_last,
                            gt_depth=gt_last,
                            pred_depth=pred_last,
                            seg_mask=seg_mask_np,
                            class_metrics=class_metrics,
                            class_names_dict=self.object_wise_metrics.classes,
                            output_path=str(objwise_vis_path)
                        )
                        logger.info(f"Saved object-wise visualization to {objwise_vis_path}")
                    except Exception as vis_e:
                        logger.error(f"Error creating object-wise visualization: {vis_e}")
                        import traceback
                        traceback.print_exc()

            except Exception as e:
                logger.error(f"Error computing object-wise metrics: {e}")
                import traceback
                traceback.print_exc()
                metrics['object_wise'] = {}

        # Recreate valid_mask on GPU for visualization
        valid_mask = (gt_depth_metric > 0)  # [T, 1, H, W] on GPU

        # Visualize
        if self.config.eval.get('save_grid', True):
            self._visualize_sequence(
                images[0], pred_depths, gt_depth_metric, importance_maps,
                valid_mask, sequence_id, metrics, fps
            )

        # Save video (GIF or MP4)
        if self.config.eval.get('out_video', True):
            # Resize images to match GT resolution for video creation
            gt_h, gt_w = gt_depth_metric.shape[-2:]
            images_resized = F.interpolate(
                images[0], size=(gt_h, gt_w), mode='bilinear', align_corners=True
            )
            self._save_video(
                images_resized, pred_depths, gt_depth_metric, valid_mask, sequence_id
            )

        # Save best frame visualizations
        if len(frame_metrics) > 0:
            logger.info(f"Best frame for sequence {sequence_id}: Frame {best_frame_idx} (AbsRel={best_frame_abs_rel:.4f})")
            # Extract features if available (None for Gear2)
            importance_map_frame = importance_maps[best_frame_idx, 0] if importance_maps is not None else None
            fg_features_frame = fg_features_all[best_frame_idx] if fg_features_all is not None else None
            bg_features_frame = bg_features_all[best_frame_idx] if bg_features_all is not None else None

            self._save_best_frame_visualizations(
                images[0, best_frame_idx],  # [3, H, W]
                pred_depths[best_frame_idx, 0],  # [H, W]
                gt_depth_metric[best_frame_idx, 0],  # [H, W]
                importance_map_frame,  # [patch_h, patch_w] or None
                fg_features_frame,  # [C, patch_h, patch_w] or None
                bg_features_frame,  # [C, patch_h, patch_w] or None
                sequence_id,
                best_frame_idx,
                best_frame_abs_rel,
                fps  # Add FPS
            )

        return metrics

    def _visualize_sequence(self, images, pred_depths, gt_depths, importance_maps,
                           valid_mask, sequence_id, metrics, fps=None):
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
            # Row 0: Image (denormalize ImageNet normalization)
            img = images[t].permute(1, 2, 0).cpu().numpy()
            # Denormalize ImageNet stats
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img * std + mean  # Reverse normalization
            img = np.clip(img, 0, 1)  # Clip to valid range
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

            # Row 3: Importance map (N/A for Gear2, which uses uniform modulation)
            if importance_maps is not None:
                importance_patch = importance_maps[t]  # [1, patch_h, patch_w]
                img_h, img_w = images[t].shape[1:]
                importance_upsampled = F.interpolate(
                    importance_patch.unsqueeze(0), size=(img_h, img_w),
                    mode='bilinear', align_corners=True
                ).squeeze().cpu().numpy()  # [H, W]
                axes[3, col].imshow(importance_upsampled, cmap='jet', vmin=0, vmax=1)
                axes[3, col].set_title(f'Importance')
            else:
                # Gear2 doesn't produce importance maps
                axes[3, col].text(0.5, 0.5, 'N/A\n(Uniform Modulation)',
                                ha='center', va='center', fontsize=14, color='gray')
                axes[3, col].set_title(f'Importance')
            axes[3, col].axis('off')

        # Add overall title with metrics
        title_str = (
            f"Sequence {sequence_id} | "
            f"TAE: {metrics.get('tae', 0):.4f} | "
            f"AbsRel: {metrics.get('abs_rel', 0):.4f} | "
            f"δ1: {metrics.get('a1', 0):.4f}"
        )
        if fps is not None:
            title_str += f" | FPS: {fps:.1f}"

        fig.suptitle(title_str, fontsize=14)

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

        # Convert depth to colorized numpy arrays (per-frame normalization like sequence.png)
        # Use same normalization as _visualize_sequence for consistency
        pred_frames = []
        gt_frames = []

        for t in range(T):
            # Process pred depth
            pred = pred_depths[t, 0].cpu().numpy()
            pred_valid = (pred > 0) & (pred < 1000)
            if pred_valid.sum() > 0:
                pred_vmin, pred_vmax = np.percentile(pred[pred_valid], [2, 98])
                pred_display = np.clip((pred - pred_vmin) / (pred_vmax - pred_vmin + 1e-8), 0, 1)
                pred_display[~pred_valid] = 0
            else:
                pred_display = np.zeros_like(pred)
            pred_colored = (plt.cm.plasma(pred_display)[:, :, :3] * 255).astype(np.uint8)
            pred_frames.append(pred_colored)

            # Process GT depth
            gt = gt_depths[t, 0].cpu().numpy()
            gt_valid = valid_mask[t, 0].cpu().numpy().astype(bool)
            if gt_valid.sum() > 0:
                gt_vmin, gt_vmax = np.percentile(gt[gt_valid], [2, 98])
                gt_display = np.clip((gt - gt_vmin) / (gt_vmax - gt_vmin + 1e-8), 0, 1)
                gt_display[~gt_valid] = 0
            else:
                gt_display = np.zeros_like(gt)
            gt_colored = (plt.cm.plasma(gt_display)[:, :, :3] * 255).astype(np.uint8)
            gt_frames.append(gt_colored)

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

    def _save_best_frame_visualizations(self, image, pred_depth, gt_depth, importance_map,
                                        fg_features, bg_features, sequence_id, frame_idx, abs_rel, fps=None):
        """
        Save best frame visualization matching train_gear3_upgrade layout

        Creates a comprehensive 4x3 grid visualization with format:
            best_frame_seq{N}_{frame_idx}_absrel_{abs_rel:.4f}.png

        Layout:
            Row 1: Input Image | GT Depth | Pred Depth
            Row 2: Importance Map (N/A) | FG Mask (N/A) | BG Mask (N/A)
            Row 3: Valid Mask | Error Map | Metrics
            Row 4: Depth Distribution (2 cols) | Importance Distribution (N/A)

        Args:
            image: [3, H, W] - RGB image
            pred_depth: [H, W] - Predicted metric depth
            gt_depth: [H, W] - Ground truth metric depth
            importance_map: None for Gear2 (uniform modulation)
            fg_features: Not used (Gear2 has no FG/BG separation)
            bg_features: Not used (Gear2 has no FG/BG separation)
            sequence_id: int - Sequence index
            frame_idx: int - Frame index within sequence
            abs_rel: float - AbsRel metric for this frame
            fps: float - Optional FPS measurement
        """
        # Convert tensors to numpy and move to CPU
        if isinstance(image, torch.Tensor):
            if image.shape[0] == 3:  # [3, H, W]
                image = image.permute(1, 2, 0)  # [H, W, 3]
            image = image.float().cpu().numpy()

        if isinstance(pred_depth, torch.Tensor):
            pred_depth = pred_depth.float().cpu().numpy()
        if isinstance(gt_depth, torch.Tensor):
            gt_depth = gt_depth.float().cpu().numpy()

        # Denormalize ImageNet normalization for image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = image * std + mean  # Reverse normalization
        image_np = np.clip(image_np, 0, 1)  # Clip to valid range

        # Get image size
        img_h, img_w = image_np.shape[:2]

        # Create valid mask
        MAX_DEPTH = 200.0
        gt_valid = (gt_depth > 0) & (gt_depth < MAX_DEPTH)
        pred_valid = (pred_depth > 0) & (pred_depth < 1000)
        valid_mask = gt_valid & pred_valid

        # Calculate error
        abs_error = np.abs(pred_depth - gt_depth)
        abs_error_masked = np.where(valid_mask, abs_error, np.nan)

        # Create figure with 4x3 grid layout matching train_gear3_upgrade
        fig = plt.figure(figsize=(15, 16))
        gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)

        # ==================== Row 1: Input, GT, Pred ====================

        # 1. Input Image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image_np)
        ax1.set_title('Input Image', fontsize=14, fontweight='bold')
        ax1.axis('off')

        # 2. Ground Truth Depth
        ax2 = fig.add_subplot(gs[0, 1])
        gt_display = np.where(valid_mask, gt_depth, np.nan)
        if valid_mask.sum() > 0:
            vmin, vmax = np.nanpercentile(gt_display, [2, 98])
        else:
            vmin, vmax = 0, 1
        im2 = ax2.imshow(gt_display, cmap='plasma', vmin=vmin, vmax=vmax)
        ax2.set_title('Ground Truth Depth (m)', fontsize=14, fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # 3. Predicted Metric Depth
        ax3 = fig.add_subplot(gs[0, 2])
        pred_display = np.where(valid_mask, pred_depth, np.nan)
        im3 = ax3.imshow(pred_display, cmap='plasma', vmin=vmin, vmax=vmax)
        ax3.set_title('Predicted Metric Depth (m)', fontsize=14, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        # ==================== Row 2: Importance, FG, BG (All N/A for Gear2) ====================

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
        ax5.text(0.5, 0.5, 'No FG/BG Separation\n\nGear2 Ablation',
                ha='center', va='center', transform=ax5.transAxes,
                fontsize=16, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax5.set_title('FG Mask (N/A)', fontsize=14, fontweight='bold')
        ax5.axis('off')

        # 6. BG Mask → N/A for Gear2
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.text(0.5, 0.5, 'No FG/BG Separation\n\nGear2 Ablation',
                ha='center', va='center', transform=ax6.transAxes,
                fontsize=16, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax6.set_title('BG Mask (N/A)', fontsize=14, fontweight='bold')
        ax6.axis('off')

        # ==================== Row 3: Valid Mask, Error, Metrics ====================

        # 7. Valid Mask
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.imshow(valid_mask.astype(np.uint8), cmap='gray_r', vmin=0, vmax=1)
        valid_ratio = valid_mask.sum() / valid_mask.size
        ax7.set_title(f'Valid Mask\n{valid_ratio*100:.1f}% ({valid_mask.sum():,} pixels)',
                     fontsize=14, fontweight='bold')
        ax7.axis('off')

        # 8. Absolute Error Map
        ax8 = fig.add_subplot(gs[2, 1])
        if valid_mask.sum() > 0:
            error_vmax = np.nanpercentile(abs_error_masked, 95)
        else:
            error_vmax = 1
        im8 = ax8.imshow(abs_error_masked, cmap='hot', vmin=0, vmax=error_vmax)
        ax8.set_title(f'Absolute Error (m)\nMean: {np.nanmean(abs_error_masked):.3f}',
                     fontsize=14, fontweight='bold')
        ax8.axis('off')
        plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)

        # 9. Depth Metrics
        ax9 = fig.add_subplot(gs[2, 2])
        y_pos = 0.95

        # Sequence info
        ax9.text(0.05, y_pos, f'Seq {sequence_id+1} Frame {frame_idx}', fontsize=11,
                transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'),
                fontweight='bold')
        y_pos -= 0.12

        # FPS if available
        if fps is not None:
            ax9.text(0.05, y_pos, f'FPS: {fps:.1f}', fontsize=10,
                    transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen'))
            y_pos -= 0.10

        # Depth metrics
        if valid_mask.sum() > 0:
            valid_gt = torch.from_numpy(gt_depth[valid_mask])
            valid_pred = torch.from_numpy(pred_depth[valid_mask])

            rmse = torch.sqrt(torch.mean((valid_pred - valid_gt) ** 2))
            mae = torch.mean(torch.abs(valid_pred - valid_gt))

            threshold = 1.25
            max_ratio = torch.max(valid_pred / valid_gt, valid_gt / valid_pred)
            delta_1 = (max_ratio < threshold).float().mean()
            delta_2 = (max_ratio < threshold ** 2).float().mean()
            delta_3 = (max_ratio < threshold ** 3).float().mean()

            ax9.text(0.05, y_pos, f'AbsRel: {abs_rel:.4f}', fontsize=10,
                    transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightcoral'))
            y_pos -= 0.08
            ax9.text(0.05, y_pos, f'Delta_1: {delta_1:.3f}', fontsize=10,
                    transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen'))
            y_pos -= 0.08
            ax9.text(0.05, y_pos, f'Delta_2: {delta_2:.3f}', fontsize=10,
                    transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen'))
            y_pos -= 0.08
            ax9.text(0.05, y_pos, f'Delta_3: {delta_3:.3f}', fontsize=10,
                    transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen'))
            y_pos -= 0.08
            ax9.text(0.05, y_pos, f'RMSE: {rmse:.3f}m', fontsize=9,
                    transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
            y_pos -= 0.08
            ax9.text(0.05, y_pos, f'MAE: {mae:.3f}m', fontsize=9,
                    transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightblue'))

        ax9.set_title('Depth Metrics', fontsize=14, fontweight='bold')
        ax9.axis('off')

        # ==================== Row 4: Depth Distribution, Importance Distribution ====================

        # 10. Depth Distribution Histogram
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
            ax10.set_title('Depth Distribution', fontsize=14, fontweight='bold')
            ax10.legend(fontsize=12)
            ax10.grid(True, alpha=0.3)

        # 11. Importance Distribution → N/A for Gear2
        ax11 = fig.add_subplot(gs[3, 2])
        ax11.text(0.5, 0.5, 'No Importance Map\n\nUniform Modulation\n(Same gamma/beta for all pixels)',
                 ha='center', va='center', transform=ax11.transAxes,
                 fontsize=14, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax11.set_title('Importance Distribution (N/A)', fontsize=14, fontweight='bold')
        ax11.axis('off')

        # Overall title
        plt.suptitle(f'Gear2 (Ablation): Sequence {sequence_id+1} Best Frame {frame_idx}',
                    fontsize=16, fontweight='bold')

        # Save with same naming convention
        save_path = self.save_dir / f"best_frame_seq{sequence_id+1}_{frame_idx}_absrel_{abs_rel:.4f}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        logger.info(f"Saved Gear2 best frame visualization: {save_path}")

    def _aggregate_metrics(self, all_metrics):
        """Aggregate metrics across sequences"""
        metric_keys = all_metrics[0].keys()
        aggregated = {}

        for key in metric_keys:
            values = [m[key] for m in all_metrics if key in m]
            if values:
                aggregated[key] = np.mean(values)

        return aggregated


@hydra.main(version_base=None, config_path="configs/gear2", config_name="config")
def main(config: DictConfig):
    """Main entry point"""
    import os

    # Override config for testing
    config.inference = True

    tester = Gear2Tester(config)
    tester.test()


if __name__ == "__main__":
    main()
