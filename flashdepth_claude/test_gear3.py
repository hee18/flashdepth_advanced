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
from dataloaders.waymo_segmentation_dataset import WaymoSegmentationDataset, collate_fn as waymo_collate_fn
from dataloaders.urbansyn_dataset import UrbanSynDepth
from dataloaders.urbansyn_segmentation_dataset import UrbanSynSegmentationDataset, urbansyn_collate_fn
from utils.metric_depth_metrics import MetricDepthMetrics, format_metrics
from utils.object_wise_evaluation import ObjectWiseMetrics
from utils.object_wise_visualization import create_object_wise_grid
from utils.helpers import save_gifs_as_grid, save_grid_to_mp4, depth_to_np_arr, torch_batch_to_np_arr
from utils.gear_common_helpers import depth_to_colored_frame
from utils.gear_video_utils import save_video as save_video_util



def get_canonical_focal_length(config):
    """
    Get canonical focal length (fixed at 1000.0 for all resolutions).

    Args:
        config: Configuration dict

    Returns:
        float: Canonical focal length (always 1000.0)
    """
    return 1000.0

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
            num_heads=num_heads
        )

        # Enable attention weights storage ONLY for last block (like train_gear3)
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
            resolution = self.config.get('resolution', self.config.eval.test_dataset_resolution)  # Allow resolution override
            data_root = self.config.dataset.data_root

            if self.object_wise_dataset == 'waymo':
                test_dataset = WaymoSegmentationDataset(
                    data_root=data_root,
                    split='val',
                    video_length=video_length,
                    resolution=resolution,
                    camera_name='FRONT',  # FRONT camera
                    objwise_mode=True  # Only use frames 0-19 with segmentation annotation
                )
                collate_fn = waymo_collate_fn
            elif self.object_wise_dataset == 'urbansyn':
                test_dataset = UrbanSynSegmentationDataset(
                    data_root=data_root,
                    split='test',
                    video_length=video_length,
                    resolution=resolution,
                    max_frames=1000
                )
                collate_fn = urbansyn_collate_fn
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
            # Priority: 1) eval.test_datasets (CLI override), 2) whole_seq_test flag
            whole_seq_test = self.config.get('whole_seq_test', False)

            # Check if eval.test_datasets is explicitly provided (CLI override)
            has_test_datasets_override = hasattr(self.config.eval, 'test_datasets') and len(self.config.eval.test_datasets) > 0

            if has_test_datasets_override:
                # CLI override: use provided test_datasets regardless of whole_seq_test
                test_datasets = self.config.eval.test_datasets
                logger.info(f"Using CLI-specified datasets: {test_datasets}")
            elif whole_seq_test:
                # Use all test datasets from config
                test_datasets = self.config.eval.test_datasets
                logger.info("Using ALL test datasets (whole_seq_test=True)")
            else:
                # Use validation datasets with waymo_seg (first 8 sequences)
                # Default to all available datasets if val_datasets not specified
                default_val_datasets = ['sintel', 'waymo_seg', 'eth3d', 'urbansyn', 'unreal4k', 'tartanair']
                test_datasets = self.config.dataset.get('val_datasets', default_val_datasets)
                # Replace 'waymo' with 'waymo_seg' if present
                test_datasets = ['waymo_seg' if d == 'waymo' else d for d in test_datasets]
                logger.info(f"Using VALIDATION datasets (whole_seq_test=False): {test_datasets}")

            video_length = int(self.config.get('vid_len', 50))  # Ensure integer (Hydra may pass string)  # Get from config override
            resolution = self.config.get('resolution', self.config.eval.test_dataset_resolution)  # Allow resolution override

            logger.info(f"Test datasets: {test_datasets}")
            logger.info(f"Video length: {video_length}")
            logger.info(f"Resolution: {resolution}")

            test_dataset = CombinedDataset(
                root_dir=self.config.dataset.data_root,
                enable_dataset_flags=test_datasets,
                resolution=resolution,
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
                
                # Infer dataset name from path for resolution mapping
                seq_path_str = str(seq_path).lower()
                dataset_name = None
                for ds in ['waymo_seg', 'waymo', 'eth3d', 'sintel', 'urbansyn', 'unreal4k', 'tartanair', 
                          'pointodyssey', 'dynamicreplica', 'spring', 'mvs-synth']:
                    if ds in seq_path_str:
                        dataset_name = ds
                        break
                
                # Resolution mapping based on combined_dataset.py logic
                # Test split uses non-square resolutions for different datasets
                if isinstance(resolution, str):
                    if resolution == 'base':
                        # Base resolution for test/val split (combined_dataset.py lines 86-99)
                        if dataset_name in ['eth3d', 'waymo', 'waymo_seg']:
                            self.resolution = (784, 518)  # (width, height)
                        elif dataset_name in ['sintel']:
                            self.resolution = (1022, 434)
                        elif dataset_name in ['urbansyn']:
                            self.resolution = (1036, 518)
                        elif dataset_name in ['unreal4k']:
                            self.resolution = (924, 518)
                        elif dataset_name in ['tartanair']:
                            self.resolution = (518, 518)
                        else:
                            # Default fallback for unknown datasets
                            logger.warning(f"Unknown dataset '{dataset_name}' in path, using default 518x518")
                            self.resolution = (518, 518)
                    
                    elif resolution == '2k':
                        # 2K resolution for test/val split (combined_dataset.py lines 108-118)
                        if dataset_name in ['eth3d', 'waymo', 'waymo_seg']:
                            self.resolution = (1918, 1274)  # (width, height)
                        elif dataset_name in ['sintel']:
                            self.resolution = (1022, 434)
                        elif dataset_name in ['urbansyn']:
                            self.resolution = (2044, 1022)
                        elif dataset_name in ['unreal4k']:
                            self.resolution = (2044, 1148)
                        else:
                            # Default fallback for unknown datasets
                            logger.warning(f"Unknown dataset '{dataset_name}' in path, using default 1918x1078")
                            self.resolution = (1918, 1078)
                    
                    else:
                        # Custom resolution string, try to parse as integer
                        try:
                            res_int = int(resolution)
                            self.resolution = (res_int, res_int)
                        except ValueError:
                            logger.error(f"Invalid resolution string: {resolution}, using 518x518")
                            self.resolution = (518, 518)
                else:
                    # Integer resolution, assume square
                    self.resolution = (int(resolution), int(resolution))

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
                logger.info(f"Dataset: {dataset_name}, Resolution: {self.resolution[0]}x{self.resolution[1]}")

            def __len__(self):
                return 1  # Single sequence

            def __getitem__(self, idx):
                # Load all frames
                images = []
                depths = []

                for img_path, depth_path in zip(self.image_files, self.depth_files):
                    # Load image
                    img = Image.open(img_path).convert('RGB')
                    # self.resolution is now (width, height) tuple from combined_dataset.py logic
                    # Use integer constants for PIL compatibility (2=BILINEAR, 0=NEAREST)
                    img = img.resize(self.resolution, 2)  # 2 = BILINEAR (positional arg)
                    img_array = np.array(img).astype(np.float32) / 255.0
                    # Normalize (ImageNet stats) - for model input only
                    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                    img_normalized = (img_array - mean) / std  # This is for model input, not visualization
                    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).float()  # [3, H, W], ensure float32
                    images.append(img_tensor)

                    # Load depth (.geometric.png for dynamicreplica)
                    depth_img = Image.open(depth_path)
                    depth_array = np.array(depth_img).astype(np.float32) / 1000.0  # mm to m
                    # Resize depth using same resolution as image
                    depth_pil = Image.fromarray(depth_array)
                    depth_resized = depth_pil.resize(self.resolution, 0)  # 0 = NEAREST (positional arg)
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

        # CombinedDataset returns (images, depths, focal_lengths, dataset_name) tuple for val/test splits
        # Convert to dict format for easier access
        if len(batch) > 0 and isinstance(batch[0], tuple):
            images, depths, focal_lengths, names = zip(*batch)
            return {
                'image': torch.stack(images, dim=0),
                'depth': torch.stack(depths, dim=0),
                'focal_lengths': torch.stack(focal_lengths, dim=0),
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
        images = batch['image'].to(self.device)  # [1, T, 3, H, W] - 전체 시퀀스를 GPU에 미리 로드
        gt_depth = batch['depth'].to(self.device)  # [1, T, H, W]
        focal_lengths = batch['focal_lengths'].to(self.device)  # [1, T]

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

        # Apply canonical space transformation to GT if enabled (for BOTH masks AND metrics calculation)
        # This ensures consistency with training/validation which use canonical space
        CANONICAL_FX = get_canonical_focal_length(self.config)
        use_canonical = self.config.get('use_canonical_space', False)
        if use_canonical:
            # Transform GT to canonical space for masks and metrics
            # inverse_canonical = inverse_actual * (fx_actual / CANONICAL_FX)
            fx_actual = focal_lengths.view(1, T, 1, 1, 1)  # [1, T, 1, 1, 1]
            gt_depth_inverse_100 = gt_depth_inverse_100 * (fx_actual / CANONICAL_FX)

        # Create canonical valid masks (70m threshold in canonical space)
        MIN_INVERSE_CANONICAL = 100.0 / 70.0
        canonical_gt_valid = (gt_depth_inverse_100 > MIN_INVERSE_CANONICAL)  # [1, T, 1, H, W]

        # Storage for predictions
        pred_depths = []
        importance_maps = []
        fg_features_list = []
        bg_features_list = []
        canonical_pred_valid_all = []  # Store canonical pred masks

        # Best frame tracking
        best_frame_idx = 0
        best_frame_abs_rel = float('inf')

        # Warmup run for FPS measurement
        logger.info(f"Warmup run for FPS measurement...")

        # Initialize Mamba sequence for warmup
        if hasattr(self.model, 'mamba'):
            self.model.mamba.start_new_sequence()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            img_warmup = images[0, 0].unsqueeze(0)
            encoder_features_warmup = self.model.pretrained.get_intermediate_layers(
                img_warmup, self.model.intermediate_layer_idx[self.model.encoder]
            )
            last_block = self.model.pretrained.blocks[-1]
            attention_weights_warmup = last_block.attn.attn_weights
            patch_tokens_warmup = encoder_features_warmup[-1]
            h_warmup, w_warmup = img_warmup.shape[2:]
            patch_h_warmup = h_warmup // self.model.patch_size
            patch_w_warmup = w_warmup // self.model.patch_size

            # Get DPT features first (without Mamba)
            dpt_features_warmup = self.model.depth_head.get_forward_features(
                encoder_features_warmup, patch_h_warmup, patch_w_warmup
            )
            path_1_warmup = dpt_features_warmup[-1]

            # Apply Gear3 modulation
            path_1_modulated_warmup, _, _, _ = self.model.gear3_head(
                patch_tokens_warmup, attention_weights_warmup, [path_1_warmup], patch_h_warmup, patch_w_warmup
            )

            # Apply Mamba temporal processing
            path_1_temporal_warmup = self.model.dpt_features_to_mamba(
                input_shape=(1, 3, h_warmup, w_warmup),
                dpt_features=path_1_modulated_warmup,
                in_dpt_layer=0
            )

            out_warmup = self.model.depth_head.scratch.output_conv1(path_1_temporal_warmup)
            out_warmup = F.interpolate(out_warmup, (h_warmup, w_warmup), mode="bilinear", align_corners=True)
            _ = self.model.depth_head.scratch.output_conv2(out_warmup)
        del encoder_features_warmup, attention_weights_warmup, patch_tokens_warmup, dpt_features_warmup
        del path_1_warmup, path_1_modulated_warmup, path_1_temporal_warmup, out_warmup
        torch.cuda.empty_cache()

        # FPS measurement (like original FlashDepth)
        warmup_frames = min(5, T)  # Warmup frames to skip initial overhead
        start_time = None  # Will start timing after warmup

        # Initialize Mamba sequence for actual test (critical for temporal processing!)
        if hasattr(self.model, 'mamba'):
            self.model.mamba.start_new_sequence()

        # Process each frame
        for t in range(T):
            # Start timing after warmup frames (like original FlashDepth)
            if t == warmup_frames:
                torch.cuda.synchronize()
                import time
                start_time = time.time()

            # GPU에서 인덱싱만 (전송 없음, 데이터는 이미 GPU에 로드됨)
            img_t = images[0, t]  # [3, H, W] - GPU에서 인덱싱만
            gt_t_inverse = gt_depth_inverse_100[0, t]  # [1, H, W] - GPU에서 인덱싱만

            # Use BFloat16 for forward pass (same as train_gear3.py)
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

                # Get DPT features first (without Mamba)
                h, w = img_t.shape[1:]
                patch_h, patch_w = h // self.model.patch_size, w // self.model.patch_size
                dpt_features = self.model.depth_head.get_forward_features(
                    encoder_features, patch_h, patch_w
                )
                path_1 = dpt_features[-1]

                # Apply Gear3 modulation
                path_1_modulated, importance_map, fg_features, bg_features = self.model.gear3_head(
                    patch_tokens, attention_weights, [path_1], patch_h, patch_w
                )

                # Apply Mamba temporal processing
                path_1_temporal = self.model.dpt_features_to_mamba(
                    input_shape=(1, 3, h, w),
                    dpt_features=path_1_modulated,
                    in_dpt_layer=0
                )

                # Get depth prediction (output is inverse depth in 100/m scale)
                out = self.model.depth_head.scratch.output_conv1(path_1_temporal)
                out = F.interpolate(out, (h, w), mode="bilinear", align_corners=True)
                out = self.model.depth_head.scratch.output_conv2(out)  # [1, 1, H, W]

                # Prediction is already positive (Softplus activation in output_conv2)
                pred_depth_inverse_100 = out  # [1, 1, H, W] in 100/m_canonical

                # Save canonical pred mask (before de-canonicalization!)
                canonical_pred_valid_t = (pred_depth_inverse_100 > MIN_INVERSE_CANONICAL)  # [1, 1, H, W]
                canonical_pred_valid_all.append(canonical_pred_valid_t.cpu())

                # De-canonicalization: convert from canonical space to actual metric space
                if use_canonical:
                    # pred_inverse_actual = pred_inverse_canonical * (CANONICAL_FX / fx_actual)
                    fx_t = focal_lengths[0, t]  # Focal length for this frame
                    pred_depth_inverse_100 = pred_depth_inverse_100 * (CANONICAL_FX / fx_t)

                # Interpolate prediction to GT resolution (like train_gear3.py validation)
                gt_t_shape = gt_t_inverse.shape[-2:]  # GT original resolution
                if pred_depth_inverse_100.shape[-2:] != gt_t_shape:
                    pred_depth_inverse_100 = F.interpolate(
                        pred_depth_inverse_100, size=gt_t_shape, mode="bilinear", align_corners=True
                    )

                # Convert to metric depth: 100/m -> m
                pred_depth_metric = 100.0 / (pred_depth_inverse_100[0] + 1e-8)  # [1, H, W]

                # Upsample importance_map to image resolution for smooth visualization
                # (like train_gear3.py does before passing to visualizer)
                h_full, w_full = img_t.shape[1:]  # Image resolution
                importance_map_resized = F.interpolate(
                    importance_map, size=(h_full, w_full), mode='bilinear', align_corners=True
                )  # [1, 1, H, W] at image resolution

            # End timing for FPS measurement (after last frame, like original FlashDepth)
            if t == T - 1 and start_time is not None:
                torch.cuda.synchronize()
                end_time = time.time()

            # List append for visualization (outside FPS measurement)
            pred_depths.append(pred_depth_metric)
            # Save upsampled importance_map (already smooth, no need to interpolate again in visualization)
            importance_maps.append(importance_map_resized[0])  # [1, H, W] at image resolution
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
        importance_maps = torch.stack(importance_maps, dim=0)  # [T, 1, patch_h, patch_w]
        fg_features_all = torch.stack(fg_features_list, dim=0)  # [T, C, patch_h, patch_w]
        bg_features_all = torch.stack(bg_features_list, dim=0)  # [T, C, patch_h, patch_w]

        # Stack canonical pred masks
        canonical_pred_valid = torch.cat(canonical_pred_valid_all, dim=1)  # [1, T, 1, H, W]

        # Convert GT to metric depth for evaluation: use canonical GT (consistent with training)
        # This ensures metrics are computed in canonical space, same as training/validation
        gt_depth_metric = 100.0 / (gt_depth_inverse_100[0] + 1e-8)  # [T, 1, H, W] in canonical meters

        # Compute metrics (both pred and GT are now in meters)
        # Move to CPU and compute per-frame metrics (like test_metric_head.py)
        pred_depths_cpu = pred_depths.cpu()
        gt_depth_metric_cpu = gt_depth_metric.cpu()

        frame_metrics = []
        for t in range(pred_depths.shape[0]):
            # Get individual frames (already on CPU)
            pred_frame = pred_depths_cpu[t, 0]  # [H, W]
            gt_frame = gt_depth_metric_cpu[t, 0]  # [H, W]

            # Use canonical space masks (70m threshold in canonical space)
            canonical_gt_valid_t = canonical_gt_valid[0, t, 0].cpu()  # [H, W]
            canonical_pred_valid_t = canonical_pred_valid[0, t, 0].cpu()  # [H, W]
            valid_mask = canonical_gt_valid_t & canonical_pred_valid_t  # [H, W] bool tensor

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
        # Object-wise evaluation: compute per-class metrics for all frames
        # Initialize variables
        seg_masks_np = None  # Will store per-frame segmentations
        per_frame_class_metrics = []  # Per-frame metrics

        if self.object_wise_enabled and 'segmentations' in batch:
            try:
                # Get per-frame segmentations
                seg_masks = batch['segmentations'][0]  # [T, H, W] - batch size is 1
                T_seg = seg_masks.shape[0]

                logger.info(f"Processing {T_seg} frames with segmentation")

                # Convert to numpy
                seg_masks_np = seg_masks.cpu().numpy() if isinstance(seg_masks, torch.Tensor) else seg_masks

                # Compute metrics for each frame
                for t in range(T_seg):
                    pred_frame = pred_depths_cpu[t, 0].numpy()  # [H, W]
                    gt_frame = gt_depth_metric_cpu[t, 0].numpy()  # [H, W]
                    seg_mask_frame = seg_masks_np[t]  # [H, W]

                    # Resize segmentation to match pred/GT if needed
                    if seg_mask_frame.shape != pred_frame.shape:
                        import cv2
                        seg_mask_frame = cv2.resize(
                            seg_mask_frame.astype(np.int32),
                            (pred_frame.shape[1], pred_frame.shape[0]),
                            interpolation=cv2.INTER_NEAREST
                        )
                        # Update in array
                        seg_masks_np[t] = seg_mask_frame

                    # Compute per-class metrics
                    frame_class_metrics = self.object_wise_metrics.compute_metrics_per_class(
                        pred_depth=pred_frame,
                        gt_depth=gt_frame,
                        seg_mask=seg_mask_frame,
                        min_pixels=100
                    )
                    per_frame_class_metrics.append(frame_class_metrics)

                # Aggregate across all frames
                class_metrics = self.object_wise_metrics.aggregate_metrics(per_frame_class_metrics)
                metrics['object_wise'] = class_metrics
                logger.info(f"Computed object-wise metrics for {len(class_metrics)} classes across {T_seg} frames")

                # Log per-class pixel counts
                if class_metrics:
                    logger.info("Per-class pixel counts (top 10):")
                    total_pixels = seg_mask_np.size
                    sorted_metrics = sorted(
                        class_metrics.items(),
                        key=lambda x: x[1].get('num_pixels', 0),
                        reverse=True
                    )[:10]
                    for class_name, metrics_dict in sorted_metrics:
                        num_pixels = metrics_dict.get('num_pixels', 0)
                        percent = 100.0 * num_pixels / total_pixels
                        logger.info(f"  {class_name:25s}: {num_pixels:8d} pixels ({percent:6.2f}%)")

                # Note: objwise_seq visualization disabled - using best_frame with object mask instead

            except Exception as e:
                logger.error(f"Error computing object-wise metrics: {e}")
                import traceback
                traceback.print_exc()
                metrics['object_wise'] = {}

        # Use canonical valid masks for visualization (combine GT and pred)
        # canonical_gt_valid: [1, T, 1, H, W], canonical_pred_valid: [1, T, 1, H, W]
        valid_mask = (canonical_gt_valid[0] & canonical_pred_valid[0])  # [T, 1, H, W]

        # Visualize
        if self.config.eval.get('save_grid', True):
            # Get actual frame numbers if available (for waymo_seg sparse segmentation)
            frame_numbers = batch.get('frame_indices', [None])[0] if 'frame_indices' in batch else None
            self._visualize_sequence(
                images[0], pred_depths, gt_depth_metric, importance_maps,
                valid_mask, sequence_id, metrics, fps, frame_numbers=frame_numbers, focal_lengths=focal_lengths[0]
            )

        # Save video (GIF or MP4)
        if self.config.eval.get('out_video', True):
            # Use original model resolution for images (following FlashDepth approach)
            # save_gifs_as_grid/save_grid_to_mp4 will handle downsampling to save_res
            save_video_util(
                images[0], pred_depths, gt_depth_metric, valid_mask, sequence_id,
                save_dir=self.save_dir,
                config=self.config
            )

        # Save best frame visualizations
        if len(frame_metrics) > 0:
            logger.info(f"Best frame for sequence {sequence_id}: Frame {best_frame_idx} (AbsRel={best_frame_abs_rel:.4f})")

            # Get segmentation and actual frame number for best frame
            if self.object_wise_enabled and seg_masks_np is not None:
                if best_frame_idx < len(seg_masks_np):
                    # Best frame has segmentation - use it
                    seg_mask_for_viz = seg_masks_np[best_frame_idx]  # [H, W]
                    class_metrics_for_viz = per_frame_class_metrics[best_frame_idx] if best_frame_idx < len(per_frame_class_metrics) else None

                    # Get actual frame number from frame_indices
                    actual_frame_number = batch['frame_indices'][0][best_frame_idx] if 'frame_indices' in batch else best_frame_idx
                    logger.info(f"Best frame batch_idx={best_frame_idx}, actual_frame={actual_frame_number} - showing object mask")
                else:
                    seg_mask_for_viz = None
                    class_metrics_for_viz = None
                    actual_frame_number = best_frame_idx
                    logger.info(f"Best frame {best_frame_idx} has no segmentation - showing valid mask instead")
            else:
                seg_mask_for_viz = None
                class_metrics_for_viz = None
                actual_frame_number = best_frame_idx

            self._save_best_frame_visualizations(
                images[0, best_frame_idx],  # [3, H, W]
                pred_depths[best_frame_idx, 0],  # [H, W]
                gt_depth_metric[best_frame_idx, 0],  # [H, W]
                importance_maps[best_frame_idx, 0],  # [patch_h, patch_w]
                fg_features_all[best_frame_idx],  # [C, patch_h, patch_w]
                bg_features_all[best_frame_idx],  # [C, patch_h, patch_w]
                sequence_id,
                actual_frame_number,  # Use actual frame number, not batch index
                best_frame_abs_rel,
                fps,  # Add FPS
                seg_mask_for_viz,  # Add segmentation mask
                class_metrics_for_viz,  # Add class metrics
                frame_metrics[best_frame_idx] if best_frame_idx < len(frame_metrics) else None  # Add frame metrics (includes boundary_f1)
            )

        return metrics

    def _visualize_sequence(self, images, pred_depths, gt_depths, importance_maps,
                           valid_mask, sequence_id, metrics, fps=None, frame_numbers=None, focal_lengths=None):
        """
        Create visualization grid for a sequence.

        Rows: Image, Metric Depth (Prediction), Metric Depth (GT), Importance Map

        Args:
            frame_numbers: Optional list of actual frame numbers (from batch['frame_indices'])
            focal_lengths: Optional tensor of focal lengths [T]
        """
        T = images.shape[0]
        # For waymo_seg objwise: interval=2, max 10 frames (regardless of actual frame numbers)
        interval = 2
        frames_to_show = min(10, (T + interval - 1) // interval)  # Ceiling division
        frame_indices = list(range(0, T, interval))[:frames_to_show]

        # Create figure with actual number of frames (not frames_to_show)
        actual_frames = len(frame_indices)
        fig, axes = plt.subplots(4, actual_frames, figsize=(actual_frames * 3, 12))
        if actual_frames == 1:
            axes = axes.reshape(-1, 1)

        for col, t in enumerate(frame_indices):
            # Determine frame label (use actual frame number if available)
            if frame_numbers is not None and t < len(frame_numbers):
                frame_label = f'Frame {frame_numbers[t]}'
            else:
                frame_label = f'Frame {t}'

            # Row 0: Image (denormalize ImageNet normalization)
            img = images[t].permute(1, 2, 0).cpu().numpy()
            # Min-Max normalization (FlashDepth original method)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = np.clip(img, 0, 1)
            img = (img * 255).astype(np.uint8)
            axes[0, col].imshow(img)
            axes[0, col].set_title(frame_label)
            axes[0, col].axis('off')

            # Row 1: Predicted metric depth (invalid pixels = black)
            MAX_DEPTH = 70.0
            pred = pred_depths[t, 0].cpu().numpy()
            gt = gt_depths[t, 0].cpu().numpy()

            # Check if dataset is sparse (< 50% valid GT pixels)
            gt_exists = (gt > 0)
            gt_density = gt_exists.sum() / gt_exists.size
            is_sparse = gt_density < 0.5

            # GT valid mask (canonical 70m)
            gt_valid = (gt > 0) & (gt < MAX_DEPTH)

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
                    height_mask = np.zeros_like(gt, dtype=bool)
                    height_mask[min_valid_row:max_valid_row+1, :] = True
                else:
                    height_mask = np.ones_like(gt, dtype=bool)

                # 2. GT missing mask (sparse LiDAR gaps)
                gt_missing = ~gt_exists

                # 3. Pred valid mask (canonical 70m)
                pred_valid_depth = (pred > 0) & (pred < MAX_DEPTH)

                # 4. Final: Within height AND (GT valid OR (GT missing AND Pred valid))
                pred_show_mask = height_mask & (gt_valid | (gt_missing & pred_valid_depth))
            else:
                # Dense dataset (sintel): Just use GT valid mask
                pred_show_mask = gt_valid

            pred_display = np.where(pred_show_mask, pred, np.nan)  # Invalid = NaN (will be black)
            if pred_show_mask.sum() > 0:
                pred_vmin = np.nanpercentile(pred_display, 2)
                pred_vmax = np.nanpercentile(pred_display, 98)
            else:
                pred_vmin, pred_vmax = 0, 1
            cmap_pred = plt.cm.plasma_r.copy()
            cmap_pred.set_bad(color='black')  # NaN pixels = black
            axes[1, col].imshow(pred_display, cmap=cmap_pred, vmin=pred_vmin, vmax=pred_vmax)  # plasma_r: near=bright, far=dark
            axes[1, col].set_title(f'Pred (m)')
            axes[1, col].axis('off')

            # Row 2: GT metric depth (invalid pixels = black)
            gt = gt_depths[t, 0].cpu().numpy()
            gt_valid = (gt > 0) & (gt < MAX_DEPTH)  # Only <70m
            gt_display = np.where(gt_valid, gt, np.nan)  # Invalid = NaN
            if gt_valid.sum() > 0:
                gt_vmin = np.nanpercentile(gt_display, 2)
                gt_vmax = np.nanpercentile(gt_display, 98)
            else:
                gt_vmin, gt_vmax = 0, 1
            cmap_gt = plt.cm.plasma_r.copy()
            cmap_gt.set_bad(color='black')  # NaN pixels = black
            axes[2, col].imshow(gt_display, cmap=cmap_gt, vmin=gt_vmin, vmax=gt_vmax)  # plasma_r: near=bright, far=dark
            axes[2, col].set_title(f'GT (m)')
            axes[2, col].axis('off')

            # Row 3: Importance map (already upsampled to image resolution in test_sequence)
            importance_resized = importance_maps[t]  # [1, H, W] already at image resolution
            importance_display = importance_resized.squeeze().cpu().numpy()  # [H, W]
            axes[3, col].imshow(importance_display, cmap='jet', vmin=0, vmax=1)
            axes[3, col].set_title(f'Importance')
            axes[3, col].axis('off')

        # Add overall title with metrics
        title_str = (
            f"Sequence {sequence_id} | "
            f"TAE: {metrics.get('tae', 0):.4f} | "
            f"AbsRel: {metrics.get('abs_rel', 0):.4f} | "
            f"δ1: {metrics.get('a1', 0):.4f} | "
            f"F1: {metrics.get('boundary_f1', 0):.3f}"
        )
        if fps is not None:
            title_str += f" | FPS: {fps:.1f}"

        # Add focal length and max depth info
        if focal_lengths is not None:
            # Use first frame's focal length (resized)
            fx_value = focal_lengths[0].item()
            # Calculate max GT depth from valid region
            gt_max = 0.0
            if valid_mask.sum() > 0:
                # gt_depths: [T, 1, H, W], valid_mask: [T, 1, H, W]
                # Remove channel dimension from both and apply mask
                valid_mask_no_channel = valid_mask[:, 0] if valid_mask.ndim == 4 else valid_mask
                gt_depths_no_channel = gt_depths[:, 0] if gt_depths.ndim == 4 else gt_depths
                gt_valid_depths = gt_depths_no_channel[valid_mask_no_channel]  # Extract valid depths
                if len(gt_valid_depths) > 0:
                    gt_max = gt_valid_depths.max().item()

            # Show valid GT range (canonical 70m in actual space)
            use_canonical = self.config.get('use_canonical_space', False)
            if use_canonical:
                CANONICAL_FX = get_canonical_focal_length(self.config)
                # depth_canonical = depth_actual * (CANONICAL_FX / fx_actual) = 70
                # Therefore: depth_actual = 70 * (fx_actual / CANONICAL_FX)
                valid_gt_max = 70.0 * (fx_value / CANONICAL_FX)
                title_str += f"\nresized_fx: {fx_value:.1f}, valid_gt_max: {valid_gt_max:.3f}m"
            else:
                title_str += f"\nresized_fx: {fx_value:.1f}, valid_gt_max: {gt_max:.3f}m"

        fig.suptitle(title_str, fontsize=14)

        plt.tight_layout()
        save_path = self.save_dir / f"sequence_{sequence_id:04d}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Saved visualization: {save_path}")

    def _save_best_frame_visualizations(self, image, pred_depth, gt_depth, importance_map,
                                        fg_features, bg_features, sequence_id, frame_idx, abs_rel, fps=None,
                                        seg_mask=None, class_metrics=None, frame_metrics=None):
        """
        Save best frame visualization matching train_gear3_upgrade layout

        Creates a comprehensive 4x3 grid visualization with format:
            best_frame_seq{N}_{frame_idx}_absrel_{abs_rel:.4f}.png

        Layout:
            Row 1: Input Image | GT Depth | Pred Depth
            Row 2: Importance Map | FG Mask | BG Mask
            Row 3: Valid Mask | Error Map | Metrics
            Row 4: Depth Distribution (2 cols) | Importance Distribution

        Args:
            image: [3, H, W] - RGB image
            pred_depth: [H, W] - Predicted metric depth
            gt_depth: [H, W] - Ground truth metric depth
            importance_map: [patch_h, patch_w] - Importance map (0-1 normalized)
            fg_features: [C, patch_h, patch_w] - Foreground features (used to extract FG mask)
            bg_features: [C, patch_h, patch_w] - Background features (used to extract BG mask)
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
        if isinstance(importance_map, torch.Tensor):
            importance_map = importance_map.float().cpu().numpy()

        # Min-Max normalization (FlashDepth original method)
        image_np = (image * 1.0 - image.min()) / (image.max() - image.min() + 1e-8)
        image_np = np.clip(image_np, 0, 1)

        # Get image size
        img_h, img_w = image_np.shape[:2]

        # Create separate valid masks for GT and Pred
        MAX_DEPTH = 70.0  # Same as training (100/70 = 1.43 inverse depth threshold)

        # Check if dataset is sparse (< 50% valid GT pixels)
        gt_exists = (gt_depth > 0)
        gt_density = gt_exists.sum() / gt_exists.size
        is_sparse = gt_density < 0.5

        # GT valid mask (canonical 70m)
        gt_valid_mask = (gt_depth > 0) & (gt_depth < MAX_DEPTH)  # GT valid pixels

        if is_sparse:
            # Sparse dataset (waymo_seg): Apply height mask + fill sparse gaps
            # 1. Find valid scan height range from GT (to exclude sky/non-scanned regions)
            valid_pixels_per_row = gt_exists.sum(axis=1)  # [H]
            min_valid_pixels_threshold = 10  # At least 10 GT pixels per row
            valid_rows = valid_pixels_per_row >= min_valid_pixels_threshold
            valid_row_indices = np.where(valid_rows)[0]

            if len(valid_row_indices) > 0:
                min_valid_row = valid_row_indices.min()
                max_valid_row = valid_row_indices.max()
                height_mask = np.zeros_like(gt_depth, dtype=bool)
                height_mask[min_valid_row:max_valid_row+1, :] = True
            else:
                height_mask = np.ones_like(gt_depth, dtype=bool)

            # 2. GT missing mask (sparse LiDAR gaps)
            gt_missing = ~gt_exists

            # 3. Pred valid mask (canonical 70m)
            pred_valid_depth = (pred_depth > 0) & (pred_depth < MAX_DEPTH)

            # 4. Final: Within height AND (GT valid OR (GT missing AND Pred valid))
            pred_show_mask = height_mask & (gt_valid_mask | (gt_missing & pred_valid_depth))

            # 5. Error mask (both GT and Pred valid)
            error_valid_mask = gt_valid_mask & pred_valid_depth
        else:
            # Dense dataset (sintel): Just use GT valid mask
            pred_show_mask = gt_valid_mask
            pred_valid_depth = (pred_depth > 0) & (pred_depth < MAX_DEPTH)
            error_valid_mask = gt_valid_mask & pred_valid_depth

        # Calculate error (only where both GT and Pred are valid)
        abs_error = np.abs(pred_depth - gt_depth)
        abs_error_masked = np.where(error_valid_mask, abs_error, np.nan)

        # Calculate importance statistics
        imp_mean = importance_map.mean()
        imp_std = importance_map.std()

        # Extract FG/BG masks from importance map (binary thresholding)
        # Use mean threshold to separate foreground from background
        fg_mask = (importance_map >= imp_mean).astype(np.float32)
        bg_mask = (importance_map < imp_mean).astype(np.float32)

        # Upsample FG/BG masks to match input image resolution (bilinear for smoother visualization)
        fg_mask_upsampled = F.interpolate(
            torch.from_numpy(fg_mask).unsqueeze(0).unsqueeze(0),
            size=(img_h, img_w),
            mode='bilinear',
            align_corners=True
        ).squeeze().numpy()

        bg_mask_upsampled = F.interpolate(
            torch.from_numpy(bg_mask).unsqueeze(0).unsqueeze(0),
            size=(img_h, img_w),
            mode='bilinear',
            align_corners=True
        ).squeeze().numpy()

        # Create figure with 4x3 grid layout matching train_gear3_upgrade
        fig = plt.figure(figsize=(15, 16))
        gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)

        # ==================== Row 1: Input, GT, Pred ====================

        # 1. Input Image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image_np)
        ax1.set_title('Input Image', fontsize=14, fontweight='bold')
        ax1.axis('off')

        # 2. Ground Truth Depth (invalid pixels = black)
        ax2 = fig.add_subplot(gs[0, 1])
        gt_display = np.where(gt_valid_mask, gt_depth, np.nan)  # Invalid = NaN
        if gt_valid_mask.sum() > 0:
            vmin = np.nanpercentile(gt_display, 2)
            vmax = np.nanpercentile(gt_display, 98)
        else:
            vmin, vmax = 0, 1
        cmap_gt = plt.cm.plasma_r.copy()
        cmap_gt.set_bad(color='black')  # NaN pixels = black
        im2 = ax2.imshow(gt_display, cmap=cmap_gt, vmin=vmin, vmax=vmax)  # plasma_r: near=bright, far=dark
        ax2.set_title('Ground Truth Depth (m)', fontsize=14, fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # 3. Predicted Metric Depth (invalid pixels = black)
        ax3 = fig.add_subplot(gs[0, 2])
        pred_display = np.where(pred_show_mask, pred_depth, np.nan)  # Invalid = NaN (will be black)
        cmap_pred = plt.cm.plasma_r.copy()
        cmap_pred.set_bad(color='black')  # NaN pixels = black
        im3 = ax3.imshow(pred_display, cmap=cmap_pred, vmin=vmin, vmax=vmax)  # plasma_r: near=bright, far=dark
        ax3.set_title('Predicted Metric Depth (m)', fontsize=14, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        # ==================== Row 2: Importance, FG, BG ====================

        # 4. Importance Map (upsampled with bilinear for smooth visualization)
        ax4 = fig.add_subplot(gs[1, 0])
        importance_upsampled = F.interpolate(
            torch.from_numpy(importance_map).unsqueeze(0).unsqueeze(0),
            size=(img_h, img_w),
            mode='bilinear',
            align_corners=True
        ).squeeze().numpy()
        im4 = ax4.imshow(importance_upsampled, cmap='jet', vmin=0, vmax=1)
        ax4.set_title(f'Importance Map\nmean={imp_mean:.3f}, std={imp_std:.3f}',
                     fontsize=14, fontweight='bold')
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

        # 5. FG Mask (Red overlay)
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.imshow(image_np)
        # Create FG overlay (Red channel only)
        fg_overlay = np.zeros((*fg_mask_upsampled.shape, 3))
        fg_overlay[..., 0] = fg_mask_upsampled  # Red channel
        ax5.imshow(fg_overlay, alpha=0.5)
        fg_ratio = fg_mask.mean() * 100
        ax5.set_title(f'FG Mask (Red)\n{fg_ratio:.1f}%', fontsize=14, fontweight='bold')
        ax5.axis('off')

        # 6. BG Mask (Blue overlay)
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.imshow(image_np)
        # Create BG overlay (Blue channel only)
        bg_overlay = np.zeros((*bg_mask_upsampled.shape, 3))
        bg_overlay[..., 2] = bg_mask_upsampled  # Blue channel
        ax6.imshow(bg_overlay, alpha=0.5)
        bg_ratio = bg_mask.mean() * 100
        ax6.set_title(f'BG Mask (Blue)\n{bg_ratio:.1f}%', fontsize=14, fontweight='bold')
        ax6.axis('off')

        # ==================== Row 3: Object Mask (objwise) or Valid Mask (general), Error, Metrics ====================

        # 7. Object Mask (objwise mode) or Valid Depth Mask (general mode)
        ax7 = fig.add_subplot(gs[2, 0])
        if self.object_wise_enabled:
            # Object-wise mode: MUST show object segmentation mask
            if seg_mask is None:
                raise ValueError("Segmentation mask is required for best_frame visualization in object-wise mode. "
                               "Please ensure object_wise.enabled=true and dataset provides segmentation.")

            # Show object mask: binary mask where any object class > 0 is white
            object_mask = (seg_mask > 0).astype(np.uint8)
            ax7.imshow(object_mask, cmap='gray', vmin=0, vmax=1)
            object_ratio = object_mask.sum() / object_mask.size
            num_classes = len(np.unique(seg_mask[seg_mask > 0])) if (seg_mask > 0).any() else 0
            ax7.set_title(f'Object Mask\n{object_ratio*100:.1f}% ({object_mask.sum():,} pixels)\n{num_classes} classes',
                         fontsize=14, fontweight='bold')
        else:
            # General mode: show GT Valid Mask (valid=white, invalid=black)
            gt_valid_ratio = gt_valid_mask.sum() / gt_valid_mask.size
            ax7.imshow(gt_valid_mask.astype(np.uint8), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            ax7.set_title(f'GT Valid Mask ({gt_valid_ratio*100:.1f}%)\ninvalid: black',
                         fontsize=12, fontweight='bold')
        ax7.axis('off')

        # 8. Absolute Error Map
        ax8 = fig.add_subplot(gs[2, 1])
        if error_valid_mask.sum() > 0:
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

        # FG:BG ratio
        ax9.text(0.05, y_pos, f'FG:BG = {fg_ratio:.1f}:{bg_ratio:.1f}', fontsize=10,
                transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightcyan'))
        y_pos -= 0.10

        # FPS if available
        if fps is not None:
            ax9.text(0.05, y_pos, f'FPS: {fps:.1f}', fontsize=10,
                    transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen'))
            y_pos -= 0.10

        # Depth metrics (computed on pixels where both GT and Pred are valid)
        if error_valid_mask.sum() > 0:
            valid_gt = torch.from_numpy(gt_depth[error_valid_mask])
            valid_pred = torch.from_numpy(pred_depth[error_valid_mask])

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
            y_pos -= 0.08

            # Add boundary F1 score if available
            if frame_metrics is not None and 'boundary_f1' in frame_metrics:
                boundary_f1 = frame_metrics['boundary_f1']
                ax9.text(0.05, y_pos, f'F1: {boundary_f1:.3f}', fontsize=9,
                        transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lavender'))
                y_pos -= 0.08

        # Per-class pixel counts (if available)
        if class_metrics is not None and len(class_metrics) > 0:
            # Add separator
            y_pos -= 0.02
            ax9.text(0.05, y_pos, '─' * 30, fontsize=8, transform=ax9.transAxes)
            y_pos -= 0.06

            # Add header
            ax9.text(0.05, y_pos, 'Top 5 Classes (Pixels):', fontsize=9,
                    transform=ax9.transAxes, fontweight='bold',
                    bbox=dict(boxstyle="round", facecolor='lightyellow'))
            y_pos -= 0.08

            # Sort by pixel count and show top 5
            sorted_classes = sorted(
                class_metrics.items(),
                key=lambda x: x[1].get('num_pixels', 0),
                reverse=True
            )[:5]

            total_pixels = seg_mask.size if seg_mask is not None else 1

            for class_name, metrics_dict in sorted_classes:
                num_pixels = metrics_dict.get('num_pixels', 0)
                percent = 100.0 * num_pixels / total_pixels
                # Shorten class name if too long
                display_name = class_name[:12] + '...' if len(class_name) > 15 else class_name
                ax9.text(0.05, y_pos, f'{display_name}: {percent:.1f}%',
                        fontsize=8, transform=ax9.transAxes,
                        bbox=dict(boxstyle="round", facecolor='lavender', alpha=0.7))
                y_pos -= 0.06

        ax9.set_title('Depth Metrics', fontsize=14, fontweight='bold')
        ax9.axis('off')

        # ==================== Row 4: Depth Distribution, Importance Distribution ====================

        # 10. Depth Distribution Histogram
        ax10 = fig.add_subplot(gs[3, :2])
        if error_valid_mask.sum() > 0:
            gt_valid = gt_depth[error_valid_mask]
            pred_valid = pred_depth[error_valid_mask]

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
        importance_flat = importance_map.flatten()

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
        ax11.set_title('Importance Distribution', fontsize=14, fontweight='bold')
        ax11.legend(fontsize=10)
        ax11.grid(True, alpha=0.3)

        # Overall title
        plt.suptitle(f'Gear3: Sequence {sequence_id} Best Frame {frame_idx}',
                    fontsize=16, fontweight='bold')

        # Save with same naming convention
        save_path = self.save_dir / f"best_frame_seq{sequence_id}_{frame_idx}_absrel_{abs_rel:.4f}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        logger.info(f"Saved Gear3 best frame visualization: {save_path}")

    def _aggregate_metrics(self, all_metrics):
        """Aggregate metrics across sequences"""
        metric_keys = all_metrics[0].keys()
        aggregated = {}

        for key in metric_keys:
            # Skip nested dictionaries (like object_wise metrics)
            if key == 'object_wise':
                continue

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
