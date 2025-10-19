"""
Gear3 Training Script: Feature-level Metric Depth Learning

Two-phase training:
    Phase 1: Train on MVS-Synth, PointOdyssey, Spring, TartanAir, DynamicReplica
    Phase 2: Fine-tune on nuScenes only

Key differences from baseline:
    - No scale/shift operation on depth map
    - Feature-level modulation using FiLM
    - Canonical space normalization (focal_length=1000)
    - Loss on inverse depth: loss(100/pred, 100/gt)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from tqdm import tqdm
import numpy as np
from pathlib import Path
from einops import rearrange
import math
import time
from datetime import timedelta

from flashdepth.model import FlashDepth
from utils.gear3_visualization import Gear3Visualizer
from flashdepth.gear3_modules import Gear3MetricHead
from dataloaders.combined_dataset import CombinedDataset
from utils.helpers import *
from utils.metric_depth_metrics import MetricDepthMetrics


def init_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # torchrun/distributed launch
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        # Single GPU fallback
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            timeout=timedelta(seconds=3600),
            rank=rank,
            world_size=world_size
        )
        dist.barrier()

    return rank, world_size, local_rank


class CanonicalSpaceNormalizer:
    """
    Canonical space normalization using fixed focal length.

    Converts metric depth to canonical space:
        depth_canonical = depth * (focal_canonical / focal_actual)

    And de-canonicalizes predictions:
        depth_actual = depth_canonical * (focal_actual / focal_canonical)
    """
    def __init__(self, focal_canonical=1000.0, enable=True):
        self.focal_canonical = focal_canonical
        self.enable = enable
        logging.info(f"Canonical space normalization: {'enabled' if enable else 'disabled'} (f={focal_canonical})")

    def canonicalize(self, depth, focal_length):
        """Convert metric depth to canonical space"""
        if not self.enable:
            return depth

        if isinstance(focal_length, (int, float)):
            focal_length = torch.tensor(focal_length, device=depth.device, dtype=depth.dtype)

        # depth_canonical = depth * (focal_canonical / focal_actual)
        scale_factor = self.focal_canonical / focal_length
        return depth * scale_factor.view(-1, 1, 1, 1)

    def canonicalize_inverse(self, inverse_depth, focal_length):
        """Convert inverse depth to canonical space

        For inverse depth: inverse_canonical = inverse_depth / (focal_canonical / focal_actual)
        Because: inverse = 1/depth, depth_canonical = depth * scale
        Therefore: inverse_canonical = 1/depth_canonical = 1/(depth*scale) = inverse/scale
        """
        if not self.enable:
            return inverse_depth

        if isinstance(focal_length, (int, float)):
            focal_length = torch.tensor(focal_length, device=inverse_depth.device, dtype=inverse_depth.dtype)

        # inverse_canonical = inverse_depth / (focal_canonical / focal_actual)
        scale_factor = self.focal_canonical / focal_length
        return inverse_depth / scale_factor.view(-1, 1, 1, 1)

    def decanonicalize(self, depth_canonical, focal_length):
        """Convert canonical space depth back to metric depth"""
        if not self.enable:
            return depth_canonical

        if isinstance(focal_length, (int, float)):
            focal_length = torch.tensor(focal_length, device=depth_canonical.device, dtype=depth_canonical.dtype)

        # depth_actual = depth_canonical * (focal_actual / focal_canonical)
        scale_factor = focal_length / self.focal_canonical
        return depth_canonical * scale_factor.view(-1, 1, 1, 1)

    def decanonicalize_inverse(self, inverse_depth_canonical, focal_length):
        """Convert canonical inverse depth back to actual inverse depth

        For inverse depth: inverse_actual = inverse_canonical * (focal_canonical / focal_actual)
        Because: inverse_canonical = inverse_actual / scale
        Therefore: inverse_actual = inverse_canonical * scale
        """
        if not self.enable:
            return inverse_depth_canonical

        if isinstance(focal_length, (int, float)):
            focal_length = torch.tensor(focal_length, device=inverse_depth_canonical.device, dtype=inverse_depth_canonical.dtype)

        # inverse_actual = inverse_canonical * (focal_canonical / focal_actual)
        scale_factor = self.focal_canonical / focal_length
        return inverse_depth_canonical * scale_factor.view(-1, 1, 1, 1)


class LogL1Loss(nn.Module):
    """
    Log L1 loss for inverse depth learning.

    Loss = L1(log(pred_inverse), log(gt_inverse))
    Works directly with inverse depth values (100/m)
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred_inverse, gt_inverse, valid_mask=None):
        """
        Args:
            pred_inverse: [B, 1, H, W] predicted inverse depth (100/m)
            gt_inverse: [B, 1, H, W] ground truth inverse depth (100/m)
            valid_mask: [B, 1, H, W] valid pixels (optional)

        Returns:
            loss: scalar
        """
        # Apply valid mask BEFORE log to avoid log(negative values)
        if valid_mask is not None:
            # Only compute loss on valid pixels
            pred_valid = pred_inverse[valid_mask.bool()]
            gt_valid = gt_inverse[valid_mask.bool()]

            if len(pred_valid) == 0:
                return torch.tensor(0.0, device=pred_inverse.device)

            # Log L1 loss on valid pixels only
            loss = F.l1_loss(
                torch.log(pred_valid + 1e-8),
                torch.log(gt_valid + 1e-8),
                reduction='mean'
            )
        else:
            # Fallback: compute on all pixels
            loss = F.l1_loss(
                torch.log(pred_inverse + 1e-8),
                torch.log(gt_inverse + 1e-8),
                reduction='mean'
            )

        return loss


class DepthVariancePseudoLabelLoss(nn.Module):
    """
    Depth Variance Pseudo-Label Loss for importance maps.

    Uses local depth variance as pseudo-label (supervision) for importance map.

    High variance regions (complex geometry) → High importance
    Low variance regions (flat surfaces) → Low importance

    This encourages importance map to have spatial diversity (high std).

    **CRITICAL**: GT depth variance is computed with torch.no_grad() to prevent gradient flow.
    """
    def __init__(self, kernel_size=15, sigma=3.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

        # Create Gaussian kernel for weighted variance computation
        self.register_buffer('gaussian_kernel', self._create_gaussian_kernel(kernel_size, sigma))

    def _create_gaussian_kernel(self, kernel_size, sigma):
        """Create 2D Gaussian kernel for weighted variance"""
        # Create 1D Gaussian
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))

        # Normalize to sum to 1
        kernel = kernel / kernel.sum()

        # Reshape for conv2d: [1, 1, kernel_size, kernel_size]
        return kernel.view(1, 1, kernel_size, kernel_size)

    def compute_local_variance(self, depth_map):
        """
        Compute local variance using Gaussian-weighted window.

        Variance = E[x²] - E[x]²

        Args:
            depth_map: [B, 1, H, W] depth values (inverse depth in 100/m scale)

        Returns:
            variance: [B, 1, H, W] local variance map
        """
        # Match dtype and device
        kernel = self.gaussian_kernel.to(dtype=depth_map.dtype, device=depth_map.device)
        padding = self.kernel_size // 2

        # E[x] (local mean)
        local_mean = F.conv2d(depth_map, kernel, padding=padding)

        # E[x²] (local mean of squares)
        local_mean_sq = F.conv2d(depth_map**2, kernel, padding=padding)

        # Variance = E[x²] - E[x]²
        variance = local_mean_sq - local_mean**2

        # Clamp to avoid negative values due to numerical errors
        return variance.clamp(min=0)

    def forward(self, importance_map, depth_map, valid_mask=None):
        """
        Args:
            importance_map: [B, 1, H, W] predicted importance in range [0, 1]
            depth_map: [B, 1, H, W] GT depth (inverse depth in 100/m scale)
            valid_mask: [B, 1, H, W] valid pixels (optional)

        Returns:
            loss: scalar L1 distance between importance and normalized variance
        """
        # CRITICAL: Compute variance WITHOUT gradient to GT depth
        with torch.no_grad():
            # Compute local variance from GT depth
            variance = self.compute_local_variance(depth_map)  # [B, 1, H, W]

            # Normalize to [0, 1] range (min-max normalization)
            if valid_mask is not None:
                # Only consider valid pixels for normalization
                variance_valid = variance[valid_mask.bool()]
                if len(variance_valid) > 0:
                    var_min = variance_valid.min()
                    var_max = variance_valid.max()
                else:
                    var_min = variance.min()
                    var_max = variance.max()
            else:
                var_min = variance.min()
                var_max = variance.max()

            # Avoid division by zero
            variance_range = var_max - var_min + 1e-8
            variance_norm = (variance - var_min) / variance_range  # [0, 1]

        # L1 loss: importance_map (trainable) vs variance_norm (pseudo-label)
        if valid_mask is not None:
            loss = F.l1_loss(
                importance_map[valid_mask.bool()],
                variance_norm[valid_mask.bool()]
            )
        else:
            loss = F.l1_loss(importance_map, variance_norm)

        return loss


class EdgeAwareLoss(nn.Module):
    """
    Edge-aware loss for importance maps.

    Aligns importance map edges with depth edges, ensuring that:
    - FG/BG boundaries coincide with depth discontinuities
    - Interior regions remain smooth
    - Prevents noisy importance maps

    Uses Sobel filter to compute gradients.

    Reference: "Edge-Guided Depth Estimation" (CVPR 2024)
    """
    def __init__(self):
        super().__init__()

        # Sobel kernels for edge detection (fixed, non-trainable)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def compute_edges(self, tensor):
        """
        Compute edge magnitude using Sobel filter.

        Args:
            tensor: [B, 1, H, W]

        Returns:
            edges: [B, 1, H, W] edge magnitude
        """
        # Match dtype and device of input tensor (handles BFloat16)
        sobel_x = self.sobel_x.to(dtype=tensor.dtype, device=tensor.device)
        sobel_y = self.sobel_y.to(dtype=tensor.dtype, device=tensor.device)

        grad_x = F.conv2d(tensor, sobel_x, padding=1)
        grad_y = F.conv2d(tensor, sobel_y, padding=1)

        # Edge magnitude
        edges = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        return edges

    def forward(self, importance_map, depth_map):
        """
        Args:
            importance_map: [B, 1, H, W] importance values in range [0, 1]
            depth_map: [B, 1, H, W] depth values (inverse depth in 100/m scale)

        Returns:
            loss: scalar (L1 distance between edges)
        """
        # Compute edges
        importance_edges = self.compute_edges(importance_map)
        depth_edges = self.compute_edges(depth_map)

        # Normalize edges to [0, 1] for fair comparison
        importance_edges = importance_edges / (importance_edges.max() + 1e-8)
        depth_edges = depth_edges / (depth_edges.max() + 1e-8)

        # L1 loss between edge maps
        return F.l1_loss(importance_edges, depth_edges)


class ContrastiveFGBGLoss(nn.Module):
    """
    Contrastive loss for FG/BG features.

    Encourages FG and BG features to be different in embedding space.
    Based on InfoNCE loss: maximize distance between FG and BG features.

    This ensures that modulation parameters (γ_fg, β_fg, γ_bg, β_bg) are distinct,
    leading to effective spatial modulation.

    Reference: "Foreground-Aware Feature Contrast (FAC++)" (CVPR 2024)
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, fg_features, bg_features):
        """
        Args:
            fg_features: [B, feature_dim] foreground features
            bg_features: [B, feature_dim] background features

        Returns:
            loss: scalar (negative cosine similarity, to maximize distance)
        """
        B = fg_features.shape[0]

        # Normalize features to unit sphere
        fg_norm = F.normalize(fg_features, dim=1)  # [B, feature_dim]
        bg_norm = F.normalize(bg_features, dim=1)  # [B, feature_dim]

        # Compute cosine similarity for same batch indices
        # We want FG[i] and BG[i] to be DIFFERENT (low similarity)
        similarity = (fg_norm * bg_norm).sum(dim=1)  # [B]

        # Average over batch
        avg_similarity = similarity.mean()

        # Maximize distance = minimize similarity
        # Add temperature scaling for numerical stability
        return avg_similarity / self.temperature


class Gear3Trainer:
    """
    Trainer for Gear3 metric depth learning.

    Frozen: DINOv2, DPT
    Fine-tuned: Mamba (LR: 1e-5)
    Trained: Gear3 modules (LR: 1e-4)
    """
    def __init__(self, config, rank, world_size, local_rank):
        self.config = config
        self.phase = config.get('phase', 1)  # Training phase (1 or 2)
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank

        # Setup device
        self.device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)

        if rank == 0:
            logging.info(f"Training Phase {self.phase} on {world_size} GPU(s)")

        # Setup results directory (only rank 0)
        phase_suffix = f"_phase{self.phase}"
        self.results_dir = Path(config.get('results_dir', f'./train_results/gear3{phase_suffix}'))
        if rank == 0:
            self.results_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging (only rank 0)
        if rank == 0:
            # Create file handler with immediate flushing
            file_handler = logging.FileHandler(self.results_dir / 'training.log')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

            # Force immediate flush after each log
            class FlushFileHandler(logging.FileHandler):
                def emit(self, record):
                    super().emit(record)
                    self.flush()  # Flush immediately

            file_handler = FlushFileHandler(self.results_dir / 'training.log')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.INFO)
            stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[file_handler, stream_handler]
            )
        else:
            logging.basicConfig(level=logging.ERROR)  # Other ranks only show errors

        self.logger = logging.getLogger(__name__)

        if rank == 0:
            self.logger.info(f"Results directory: {self.results_dir}")
            self.logger.info(f"Training phase: {self.phase}")

        # Setup canonical space normalizer
        self.canonical_normalizer = CanonicalSpaceNormalizer(
            focal_canonical=config.get('canonical_focal_length', 1000.0),
            enable=config.get('use_canonical_space', True)
        )

        # Initialize model
        self.model = self._setup_model()

        # Set proper training mode (trainable parts in train mode, frozen parts in eval mode)
        self._set_train_mode()

        # Setup data loaders
        self.train_loader, self.val_loader = self._setup_data_loaders()

        # Setup optimizer and loss
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.loss_fn = LogL1Loss()

        # Regularization losses - ALL DISABLED
        # Importance map now uses raw attention directly (no trainable parameters)
        # Therefore no regularization is needed
        self.use_depth_variance_loss = False
        self.use_edge_aware_loss = False
        self.use_contrastive_fgbg_loss = False

        if rank == 0:
            self.logger.info("=== Regularization Losses ===")
            self.logger.info("✗ ALL regularization losses disabled (importance map uses raw attention)")

        # FPS measurement config
        self.measure_fps = config.training.get('measure_fps', False)
        self.fps_log_freq = config.training.get('fps_log_freq', 100)
        self.fps_buffer = []  # Store recent FPS measurements

        if rank == 0 and self.measure_fps:
            self.logger.info(f"FPS measurement enabled (log every {self.fps_log_freq} steps)")

        # Setup wandb
        if config.training.get('wandb', False):
            wandb.init(
                project="flashdepth-gear3",
                name=f"gear3_phase{self.phase}_{config.training.get('wandb_name', 'experiment')}",
                config=dict(config)
            )

        # Setup visualizer with separate folders
        self.train_visualizer = Gear3Visualizer(save_dir=self.results_dir / "visualizations" / "train")
        self.val_visualizer = Gear3Visualizer(save_dir=self.results_dir / "visualizations" / "valid")

        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_step = 0  # Track which step achieved best validation loss
        self.current_val_loss = None  # Track current validation loss for checkpoint

        # Validation visualization config: track which sequences to visualize
        self.val_vis_config = {
            'sintel': {'count': 3, 'interval': 5, 'saved': []},
            'waymo': {'count': 8, 'interval': 5, 'saved': []}
        }

    def _setup_model(self):
        """Initialize FlashDepth with Gear3 metric head"""
        # Create base FlashDepth model
        model_config = dict(self.config.model)
        model_config['batch_size'] = self.config.training.batch_size
        model_config['use_metric_head'] = False  # Don't use GSP head

        model = FlashDepth(**model_config)

        # Load pre-trained checkpoint (DINOv2 + DPT only, excluding Mamba)
        checkpoint_path = self.config.get('load')
        if checkpoint_path and checkpoint_path != 'true':
            if os.path.exists(checkpoint_path):
                self.logger.info(f"Loading checkpoint from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location='cpu')

                # Extract state dict
                if isinstance(checkpoint, dict) and 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

                # Remove module. prefix if present
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

                # Load only DINOv2 and DPT refinement layers
                # Exclude: Mamba (modulated input), output_conv1/2 (modulated features)
                loaded_dict = {}
                excluded_keys = []
                for k, v in state_dict.items():
                    # Exclude modules that receive modulated features
                    if any(x in k for x in ['mamba', 'hybrid_fusion', 'teacher_model',
                                            'output_conv1', 'output_conv2']):
                        excluded_keys.append(k)
                    else:
                        loaded_dict[k] = v

                # Load state dict (strict=False to allow missing modules)
                model.load_state_dict(loaded_dict, strict=False)
                self.logger.info(f"Loaded {len(loaded_dict)} parameters from checkpoint")
                self.logger.info(f"  - DINOv2 encoder: ✓")
                self.logger.info(f"  - DPT projects/resize/refinenet: ✓")
                self.logger.info(f"Excluded {len(excluded_keys)} parameters for training:")
                self.logger.info(f"  - Mamba (modulated input)")
                self.logger.info(f"  - output_conv1/2 (modulated features)")
                self.logger.info(f"  - hybrid_fusion, teacher_model (not used)")
            else:
                self.logger.warning(f"Checkpoint {checkpoint_path} not found")

        # Add Gear3 metric head
        embed_dim = 1024 if model.encoder == 'vitl' else 384
        dpt_dim = 256 if model.encoder == 'vitl' else 64
        num_heads = 16 if model.encoder == 'vitl' else 6

        model.gear3_head = Gear3MetricHead(
            embed_dim=embed_dim,
            dpt_dim=dpt_dim,
            num_heads=num_heads
        )

        # Enable attention weights storage ONLY for last block (saves ~11GB memory)
        # Gear3 only uses last block's attention for importance prediction
        for i, block in enumerate(model.pretrained.blocks):
            if i == len(model.pretrained.blocks) - 1:
                block.attn.store_attn_weights = True
                self.logger.info(f"Enabled attention weights storage for block {i} (last block)")
            else:
                block.attn.store_attn_weights = False

        model = model.to(self.device)

        # Freeze and configure parameters
        self._configure_parameters(model)

        # Wrap with DDP if multi-GPU
        if self.world_size > 1:
            model = DDP(
                model,
                device_ids=[self.local_rank],
                find_unused_parameters=True  # Important for frozen parameters
            )
            if self.rank == 0:
                self.logger.info(f"Model wrapped with DDP on {self.world_size} GPUs")

        return model

    def _configure_parameters(self, model):
        """
        Freeze: DINOv2, DPT refinement layers
        Train from scratch: Mamba, output_conv1/2, Gear3 modules
        """
        frozen_params = 0
        mamba_params = 0
        output_conv_params = 0
        gear3_params = 0

        for name, param in model.named_parameters():
            if 'gear3_head' in name:
                # Gear3 modules: trainable
                param.requires_grad = True
                gear3_params += param.numel()
                self.logger.info(f"Trainable (Gear3): {name} - {param.shape}")

            elif 'mamba' in name:
                # Mamba: train from scratch (receives modulated input)
                param.requires_grad = True
                mamba_params += param.numel()

            elif 'output_conv' in name:
                # DPT output head: train from scratch (receives modulated features)
                param.requires_grad = True
                output_conv_params += param.numel()
                self.logger.info(f"Trainable (output_conv): {name} - {param.shape}")

            else:
                # Everything else (DINOv2, DPT refinement): frozen
                param.requires_grad = False
                frozen_params += param.numel()

        self.logger.info(f"Frozen (DINOv2 + DPT refinement): {frozen_params:,}")
        self.logger.info(f"Trainable (Mamba, from scratch): {mamba_params:,}")
        self.logger.info(f"Trainable (output_conv, from scratch): {output_conv_params:,}")
        self.logger.info(f"Trainable (Gear3 modules): {gear3_params:,}")

        total_trainable = mamba_params + output_conv_params + gear3_params
        self.logger.info(f"Total trainable: {total_trainable:,}")

    def _set_train_mode(self):
        """
        Set model to training mode, but keep frozen parts in eval mode.
        This prevents BatchNorm/Dropout in frozen parts from updating.
        """
        self.model.train()

        # Keep frozen parts in eval mode (like train_metric_head)
        for name, module in self.model.named_modules():
            # Skip empty name (root module)
            if name == '':
                continue

            # Keep trainable parts in train mode
            if any(keyword in name for keyword in ['gear3_head', 'mamba', 'output_conv']):
                continue

            # Set frozen parts to eval mode
            module.eval()

    def _setup_data_loaders(self):
        """Setup phase-specific data loaders"""
        if self.phase == 1:
            # Phase 1: Use datasets from config (allow config override)
            train_datasets = self.config.dataset.get('train_datasets', ['mvs-synth', 'tartanair', 'pointodyssey', 'spring'])
            val_datasets = self.config.dataset.get('val_datasets', ['sintel', 'waymo'])
        else:
            # Phase 2: nuScenes only
            train_datasets = ['nuscenes']
            val_datasets = ['nuscenes']

        if self.rank == 0:
            self.logger.info(f"Phase {self.phase} - Train datasets: {train_datasets}")
            self.logger.info(f"Phase {self.phase} - Val datasets: {val_datasets}")

        # Training dataset
        train_dataset = CombinedDataset(
            root_dir=self.config.dataset.data_root,
            enable_dataset_flags=train_datasets,
            resolution=self.config.dataset.resolution,
            split='train',
            video_length=self.config.dataset.video_length,
            color_aug=False  # No augmentation for metric training
        )

        # Validation dataset: use full sequence like original FlashDepth
        val_dataset = CombinedDataset(
            root_dir=self.config.dataset.data_root,
            enable_dataset_flags=val_datasets,
            resolution=self.config.dataset.resolution,
            split='val',
            video_length=self.config.dataset.video_length  # Full sequence for temporal modeling
        )

        # Setup samplers for DDP
        if self.world_size > 1:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=True
            )
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                drop_last=True
            )
        else:
            train_sampler = None
            val_sampler = None

        # Data loaders with custom collate_fn
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),  # Only shuffle if not using sampler
            num_workers=self.config.training.workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.collate_fn
        )

        # Validation: batch_size=1 like original FlashDepth to avoid memory issues
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,  # Process one video sequence at a time
            sampler=val_sampler,
            shuffle=False,
            num_workers=self.config.training.workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self.collate_fn
        )

        if self.rank == 0:
            self.logger.info(f"Train dataset size: {len(train_dataset)}")
            self.logger.info(f"Val dataset size: {len(val_dataset)}")
            self.logger.info(f"Train batch size: {self.config.training.batch_size}, video_length: {self.config.dataset.video_length}")
            self.logger.info(f"Val batch size: 1, video_length: {self.config.dataset.video_length} (like original FlashDepth)")

        return train_loader, val_loader

    def collate_fn(self, batch):
        """Custom collate function to filter out None values"""
        # Filter out None values
        batch = [item for item in batch if item is not None]

        # If all items are None, skip this batch
        if len(batch) == 0:
            return None

        # Use default collate for valid items
        return torch.utils.data.dataloader.default_collate(batch)

    def _setup_optimizer(self):
        """Setup optimizer - same LR for all trainable modules (all train from scratch)"""
        base_lr = self.config.training.get('gear3_lr', 1e-4)  # Use same LR for all

        # Separate parameter groups (for monitoring and potential future tuning)
        mamba_params = []
        gear3_params = []
        output_conv_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'gear3_head' in name:
                    gear3_params.append(param)
                elif 'mamba' in name:
                    mamba_params.append(param)
                elif 'output_conv' in name:
                    output_conv_params.append(param)
                else:
                    # Fallback: should not happen, but log for debugging
                    self.logger.warning(f"Trainable parameter not in any group: {name}")

        param_groups = [
            {'params': gear3_params, 'lr': base_lr, 'name': 'gear3'},
            {'params': mamba_params, 'lr': base_lr, 'name': 'mamba'},
            {'params': output_conv_params, 'lr': base_lr, 'name': 'output_conv'}
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            betas=[0.9, 0.95],  # Same as original FlashDepth
            weight_decay=self.config.training.get('weight_decay', 1e-6)
        )

        self.logger.info(f"Optimizer setup:")
        self.logger.info(f"  Gear3: {len(gear3_params)} params, LR={base_lr}")
        self.logger.info(f"  Mamba: {len(mamba_params)} params, LR={base_lr}")
        self.logger.info(f"  Output_conv: {len(output_conv_params)} params, LR={base_lr}")

        return optimizer

    def _setup_scheduler(self):
        """Setup cosine annealing scheduler with warmup"""
        total_steps = self.config.training.iterations
        warmup_steps = 1000  # Same as original FlashDepth
        decay_start = int(total_steps * 0.3)  # Start decay at 30%

        def lr_lambda(step):
            if step < warmup_steps:
                # Warmup: 0.1x -> 1x
                return 0.1 + 0.9 * (step / warmup_steps)
            elif step < decay_start:
                # Stable phase
                return 1.0
            else:
                # Cosine decay: 1.0 -> 0.01
                progress = (step - decay_start) / (total_steps - decay_start)
                return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        return scheduler

    def train(self):
        """Main training loop"""
        if self.rank == 0:
            self.logger.info("Starting training...")

        train_iterator = iter(self.train_loader)

        # Use tqdm for progress bar (only rank 0)
        pbar = tqdm(
            range(self.config.training.iterations),
            desc="Training",
            disable=(self.rank != 0)
        )
        
        for step in pbar:
            self.global_step = step

            # Get batch
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(self.train_loader)
                batch = next(train_iterator)

            # Skip None batches (all items were invalid)
            if batch is None:
                continue

            # Training step
            loss_dict = self.train_step(batch)
            
            # Get learning rates
            lr_gear3 = self.optimizer.param_groups[0]['lr']
            lr_mamba = self.optimizer.param_groups[1]['lr']

            # Update progress bar (every step)
            postfix_dict = {
                'loss': f'{loss_dict["loss"]:.4f}',
                'depth': f'{loss_dict["depth_loss"]:.4f}',
                'lr_g3': f'{lr_gear3:.2e}',
                'lr_mb': f'{lr_mamba:.2e}'
            }

            # Add FPS to progress bar if available
            if self.measure_fps and len(self.fps_buffer) > 0:
                avg_fps = sum(self.fps_buffer[-20:]) / min(len(self.fps_buffer), 20)  # Last 20 samples
                postfix_dict['fps'] = f'{avg_fps:.1f}'

            pbar.set_postfix(postfix_dict)

            # FPS logging (every fps_log_freq steps)
            if self.measure_fps and step > 0 and step % self.fps_log_freq == 0 and len(self.fps_buffer) > 0:
                avg_fps = sum(self.fps_buffer[-self.fps_log_freq:]) / min(len(self.fps_buffer), self.fps_log_freq)
                if self.rank == 0:
                    self.logger.info(f"Step {step}: Average FPS (forward pass only): {avg_fps:.2f}")

            # WandB logging (every step)
            wandb_dict = {**loss_dict, 'lr_gear3': lr_gear3, 'lr_mamba': lr_mamba}
            if self.measure_fps and len(self.fps_buffer) > 0:
                wandb_dict['fps'] = sum(self.fps_buffer[-20:]) / min(len(self.fps_buffer), 20)
            if self.config.training.get('wandb', False):
                wandb.log(wandb_dict, step=step)

            # Training visualization (steps 0, 10, 50, 100, then every 250) - rank 0 only
            vis_steps = [0, 10, 50, 100]
            if (step in vis_steps or step % 250 == 0) and self.train_visualizer and self.rank == 0:
                try:
                    # Use current training batch for visualization
                    self.model.eval()

                    # Unwrap DDP for visualization
                    model = self.model.module if isinstance(self.model, DDP) else self.model

                    # Unpack and move to device (batch is still on CPU from dataloader)
                    images, gt_depth, dataset_idx = batch
                    images = images.to(self.device)
                    gt_depth = gt_depth.to(self.device)

                    if gt_depth.ndim == 3:
                        gt_depth = gt_depth.unsqueeze(1)
                    elif gt_depth.ndim == 4 and gt_depth.shape[1] != 1:
                        gt_depth = gt_depth.unsqueeze(2)

                    focal_length = 1000.0
                    gt_depth_inverse_canonical = self.canonical_normalizer.canonicalize_inverse(gt_depth, focal_length)
                    gt_depth_inverse = gt_depth_inverse_canonical * 100.0  # Training uses 100/m

                    img_t = images[:, 0]
                    gt_t_inverse = gt_depth_inverse[:, 0]

                    with torch.no_grad():
                        encoder_features = model.pretrained.get_intermediate_layers(
                            img_t, model.intermediate_layer_idx[model.encoder]
                        )
                        last_block = model.pretrained.blocks[-1]
                        attention_weights = last_block.attn.attn_weights
                        patch_tokens = encoder_features[-1]

                        h, w = img_t.shape[2:]
                        patch_h, patch_w = h // model.patch_size, w // model.patch_size
                        dpt_features = model.depth_head.get_forward_features(encoder_features, patch_h, patch_w)

                        path_1_modulated, importance_map, fg_features, bg_features = model.gear3_head(
                            patch_tokens, attention_weights, dpt_features, patch_h, patch_w
                        )

                        out = model.depth_head.scratch.output_conv1(path_1_modulated)
                        out = F.interpolate(out, (h, w), mode="bilinear", align_corners=True)
                        out = model.depth_head.scratch.output_conv2(out)

                        # Convert prediction to metric depth for visualization: 100/m -> m
                        # Already positive (Softplus activation in output_conv2)
                        pred_depth_metric = 100.0 / (out + 1e-8)
                        gt_depth_metric = 100.0 / (gt_t_inverse + 1e-8)

                        importance_map_resized = F.interpolate(
                            importance_map, size=(h, w), mode='bilinear', align_corners=True
                        )

                        # Move tensors to CPU for visualization (only first batch, first frame)
                        sample_batch = (
                            images[:1, :1].cpu(),  # [1, 1, 3, H, W]
                            gt_depth_metric[:1].cpu(),  # [1, 1, H, W] (already has channel dim)
                            dataset_idx
                        )
                        model_outputs_cpu = {
                            'pred_depth': pred_depth_metric[:1].cpu(),  # [1, 1, H, W]
                            'importance_map': importance_map_resized[:1].cpu()  # [1, 1, H, W]
                        }

                        # Get FPS for visualization
                        current_fps = None
                        if self.measure_fps and len(self.fps_buffer) > 0:
                            current_fps = sum(self.fps_buffer[-20:]) / min(len(self.fps_buffer), 20)

                        # Pass loss_dict for visualization
                        self.train_visualizer.create_validation_summary(
                            sample_batch, model_outputs_cpu, step, prefix="training", fps=current_fps, loss_dict=loss_dict
                        )

                    self._set_train_mode()

                except Exception as e:
                    self.logger.warning(f"Failed to save training visualization: {e}")
                    self._set_train_mode()

            # Validation (run at step 0 and every val_freq steps)
            if step % self.config.training.get('val_freq', 1000) == 0:
                val_metrics = self.validate()
                self.logger.info(f"Validation at step {step}: {val_metrics}")

                # Update current validation loss for checkpoint
                self.current_val_loss = val_metrics['loss']

                if self.config.training.get('wandb', False):
                    wandb.log({f'val/{k}': v for k, v in val_metrics.items()}, step=step)

                # Save best model
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.best_step = step  # Track best step
                    self.save_checkpoint(f'best.pth')
                    self.logger.info(f"New best model at step {step}: val_loss={val_metrics['loss']:.4f}")

                self._set_train_mode()

            # Save checkpoint
            if step % self.config.training.get('save_freq', 5000) == 0 and step > 0:
                self.save_checkpoint(f'checkpoint_step{step}_phase{self.phase}.pth')

        # Final save
        self.save_checkpoint(f'final_step{step}_phase{self.phase}.pth')
        self.logger.info("Training completed!")

    def train_step(self, batch):
        """Single training step with BFloat16 autocast"""
        # Unpack batch
        images, gt_depth, dataset_idx = batch
        images = images.to(self.device)
        gt_depth = gt_depth.to(self.device)

        # Add channel dimension if needed
        if gt_depth.ndim == 3:
            gt_depth = gt_depth.unsqueeze(1)
        elif gt_depth.ndim == 4 and gt_depth.shape[1] != 1:
            gt_depth = gt_depth.unsqueeze(2)

        focal_length = 1000.0
        B, T = images.shape[:2]

        # Get the actual model (unwrap DDP if needed)
        model = self.model.module if isinstance(self.model, DDP) else self.model

        # DEBUG: Check raw GT depth values
        if self.global_step < 5:  # Only log first few steps
            self.logger.info(f"DEBUG - Raw GT depth from dataloader: min={gt_depth.min():.4f}, max={gt_depth.max():.4f}, shape={gt_depth.shape}")
            self.logger.info(f"DEBUG - GT depth has {(gt_depth > 0).sum()} valid pixels out of {gt_depth.numel()} total")

        # Convert GT to canonical inverse depth and scale by 100
        gt_depth_inverse_canonical = self.canonical_normalizer.canonicalize_inverse(gt_depth, focal_length)
        gt_depth_inverse = gt_depth_inverse_canonical * 100.0  # 100/meters for training

        # DEBUG: Check after canonicalization
        if self.global_step < 5:
            self.logger.info(f"DEBUG - After canonicalization & scaling: min={gt_depth_inverse.min():.4f}, max={gt_depth_inverse.max():.4f}")
            self.logger.info(f"DEBUG - Has {(gt_depth_inverse > 0).sum()} valid pixels")

        # Forward pass with BFloat16 autocast (like original FlashDepth)
        total_loss = 0
        total_depth_loss = 0
        valid_frames = 0

        # FPS measurement (forward pass only)
        if self.measure_fps:
            torch.cuda.synchronize()  # Wait for GPU operations to finish
            fps_start = time.perf_counter()

        for t in range(T):
            img_t = images[:, t]
            gt_t = gt_depth_inverse[:, t]

            # Use BFloat16 for forward pass
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # Extract features from DINOv2 (frozen, no grad)
                with torch.no_grad():
                    encoder_features = model.pretrained.get_intermediate_layers(
                        img_t, model.intermediate_layer_idx[model.encoder]
                    )
                    
                    # Get attention weights (already computed and stored)
                    last_block = model.pretrained.blocks[-1]
                    attention_weights = last_block.attn.attn_weights
                    
                    # Get patch tokens from last encoder layer
                    patch_tokens = encoder_features[-1]

                # Get DPT features (frozen, no grad)
                h, w = img_t.shape[2:]
                patch_h, patch_w = h // model.patch_size, w // model.patch_size

                with torch.no_grad():
                    dpt_features = model.depth_head.get_forward_features(
                        encoder_features, patch_h, patch_w
                    )

                # Apply Gear3 modulation (trainable) - only path_1 is modulated
                path_1_modulated, importance_map, fg_features, bg_features = model.gear3_head(
                    patch_tokens, attention_weights, dpt_features, patch_h, patch_w
                )

                # Pass through DPT output head (trainable)
                out = model.depth_head.scratch.output_conv1(path_1_modulated)
                out = F.interpolate(out, (h, w), mode="bilinear", align_corners=True)
                out = model.depth_head.scratch.output_conv2(out)

                # Prediction is already positive (Softplus activation in output_conv2)
                pred_depth_inverse = out

            # End forward pass timing HERE (before loss computation)
            if self.measure_fps and t == T - 1:  # Only measure after last frame
                torch.cuda.synchronize()  # Ensure all GPU operations complete
                fps_end = time.perf_counter()
                elapsed = fps_end - fps_start
                fps = (B * T) / elapsed  # Total frames (batch_size * video_length) per second
                self.fps_buffer.append(fps)

            # Continue with loss computation (outside autocast for stability)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # DEBUG: Check prediction values in first few steps
                if self.global_step < 5:
                    self.logger.info(f"DEBUG - Pred inverse depth: min={pred_depth_inverse.min():.4f}, max={pred_depth_inverse.max():.4f}, mean={pred_depth_inverse.mean():.4f}")

                # Clamp prediction to reasonable range to prevent NaN
                pred_depth_inverse = torch.clamp(pred_depth_inverse, min=1e-3, max=1e4)

                # Compute loss with valid mask (GT only, like original FlashDepth)
                # Filter out invalid inverse depths: should be in reasonable range
                # Max 70m depth = 100/70 = 1.43 in (100/m) inverse depth
                # So inverse depth should be > 1.43 (i.e., depth < 70m)
                MIN_INVERSE_DEPTH = 100.0 / 70.0  # 100/70m = 1.43 in 100/m scale (max 70m depth)
                gt_valid_mask = (gt_t > MIN_INVERSE_DEPTH)  # Filter out >70m depths and invalid values
                valid_mask = gt_valid_mask

                if valid_mask.sum() == 0:
                    self.logger.warning(f"Skipping frame {t} - no valid GT pixels")
                    continue

                # Loss computation in BFloat16
                depth_loss_t = self.loss_fn(pred_depth_inverse, gt_t, valid_mask.float())

                # No regularization losses - importance map uses raw attention
                loss_t = depth_loss_t

            # Safety check: skip if loss is NaN or too large
            if torch.isnan(loss_t) or torch.isinf(loss_t):
                self.logger.warning(f"Skipping frame {t} due to NaN/Inf loss")
                continue

            if loss_t > 1e6:
                self.logger.warning(f"Skipping frame {t} due to abnormal loss: {loss_t.item():.2f}")
                continue

            total_loss += loss_t
            total_depth_loss += depth_loss_t
            valid_frames += 1

        # Average loss over valid frames
        if valid_frames == 0:
            self.logger.error("No valid frames in batch!")
            return {'loss': 0.0, 'depth_loss': 0.0}

        loss = total_loss / valid_frames
        avg_depth_loss = total_depth_loss / valid_frames

        # Backward pass (outside autocast for numerical stability)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        return {
            'loss': loss.item(),
            'depth_loss': avg_depth_loss.item()
        }

    @torch.no_grad()
    def validate(self):
        """Validation loop with visualization - like original FlashDepth"""
        # Clear cache before validation to free memory from training
        torch.cuda.empty_cache()

        self.model.eval()

        # Unwrap DDP if needed
        model = self.model.module if isinstance(self.model, DDP) else self.model

        total_loss = 0
        num_batches = 0

        # Reset visualization tracking for this validation run
        # This ensures we save visualizations at EVERY validation step
        for dataset_name in self.val_vis_config:
            self.val_vis_config[dataset_name]['saved'] = []

        # Track dataset-specific sequence counters for visualization
        dataset_sequence_counters = {'sintel': 0, 'waymo': 0}

        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation", disable=(self.rank != 0))):
            # Skip None batches (all items were invalid)
            if batch is None:
                continue

            # Unpack batch
            images, gt_depth, dataset_idx = batch
            
            # Use BFloat16 autocast like original FlashDepth
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                images = images.to(self.device)
                gt_depth = gt_depth.to(self.device)

                # Add channel dimension if needed
                if gt_depth.ndim == 3:
                    gt_depth = gt_depth.unsqueeze(1)
                elif gt_depth.ndim == 4 and gt_depth.shape[1] != 1:
                    gt_depth = gt_depth.unsqueeze(2)

                focal_length = 1000.0
                B, T = images.shape[:2]

                # Canonicalize GT inverse depth and scale to 100/m
                gt_depth_inverse_canonical = self.canonical_normalizer.canonicalize_inverse(gt_depth, focal_length)
                gt_depth_inverse = gt_depth_inverse_canonical * 100.0  # Training uses 100/m

                # Process all frames in sequence (like original FlashDepth)
                frame_losses = []
                for t in range(T):
                    img_t = images[:, t]
                    gt_t_inverse = gt_depth_inverse[:, t]

                    # Extract features from DINOv2
                    encoder_features = model.pretrained.get_intermediate_layers(
                        img_t, model.intermediate_layer_idx[model.encoder]
                    )

                    # Get attention weights from last block
                    last_block = model.pretrained.blocks[-1]
                    attention_weights = last_block.attn.attn_weights
                    patch_tokens = encoder_features[-1]

                    # Get DPT features
                    h, w = img_t.shape[2:]
                    patch_h, patch_w = h // model.patch_size, w // model.patch_size
                    dpt_features = model.depth_head.get_forward_features(encoder_features, patch_h, patch_w)

                    # Apply modulation and get importance map (only path_1 is modulated)
                    path_1_modulated, importance_map, fg_features, bg_features = model.gear3_head(
                        patch_tokens, attention_weights, dpt_features, patch_h, patch_w
                    )

                    # Get depth (at model resolution)
                    out = model.depth_head.scratch.output_conv1(path_1_modulated)
                    out = F.interpolate(out, (h, w), mode="bilinear", align_corners=True)
                    out = model.depth_head.scratch.output_conv2(out)

                    pred_depth_inverse = out  # [B, 1, h, w] at model resolution

                    # Interpolate prediction to GT resolution (like original FlashDepth)
                    gt_t_shape = gt_t_inverse.shape[-2:]  # GT original resolution
                    if pred_depth_inverse.shape[-2:] != gt_t_shape:
                        pred_depth_inverse = F.interpolate(
                            pred_depth_inverse, size=gt_t_shape, mode="bilinear", align_corners=True
                        )

                    # Compute loss in inverse depth space (100/m)
                    # Filter out invalid inverse depths (valid_mask: >= 0 for original FlashDepth)
                    gt_valid_mask = (gt_t_inverse >= 0)  # -1 means invalid
                    valid_mask = gt_valid_mask.float()
                    
                    if valid_mask.sum() > 0:
                        loss_t = self.loss_fn(pred_depth_inverse, gt_t_inverse, valid_mask)
                        frame_losses.append(loss_t.float())  # Convert to Float32 for accumulation

                    # Get dataset name for this batch
                    if isinstance(dataset_idx, str):
                        current_dataset = dataset_idx
                    elif isinstance(dataset_idx, (list, tuple)):
                        current_dataset = str(dataset_idx[0])
                    elif torch.is_tensor(dataset_idx):
                        current_dataset = str(dataset_idx[0].item() if dataset_idx.dim() > 0 else dataset_idx.item())
                    else:
                        current_dataset = str(dataset_idx)

                    # Validation visualization: save multiple sequences per dataset (rank 0 only)
                    # Sintel: 3 samples (sequence 0, 5, 10)
                    # Waymo: 8 samples (sequence 0, 5, 10, 15, 20, 25, 30, 35)
                    if t == 0 and self.val_visualizer and self.rank == 0:
                        # Check if this dataset is in our visualization config
                        if current_dataset in self.val_vis_config:
                            config = self.val_vis_config[current_dataset]
                            seq_num = dataset_sequence_counters[current_dataset]

                            # Check if we should save this sequence
                            # Save at intervals: 0, interval, 2*interval, ... until count is reached
                            # seq_num % interval == 0 gives us sequences 0, 5, 10, 15, 20, ...
                            # We only save if this seq_num hasn't been saved yet in THIS validation run
                            should_save = (
                                seq_num % config['interval'] == 0 and
                                seq_num not in config['saved'] and
                                len(config['saved']) < config['count']
                            )

                            if should_save:
                                try:
                                    # Convert to metric depth for visualization: 100/m -> m
                                    # Convert to Float32 for CPU operations
                                    pred_depth_metric = (100.0 / (pred_depth_inverse.float() + 1e-8)).cpu()
                                    gt_depth_metric = (100.0 / (gt_t_inverse.float() + 1e-8)).cpu()

                                    # Get GT resolution for visualization
                                    gt_h, gt_w = gt_t_inverse.shape[-2:]

                                    # Resize importance map to GT resolution
                                    importance_map_resized = F.interpolate(
                                        importance_map, size=(gt_h, gt_w), mode='bilinear', align_corners=True
                                    )

                                    # Resize images to GT resolution (for visualization consistency)
                                    img_t_resized = F.interpolate(
                                        img_t, size=(gt_h, gt_w), mode='bilinear', align_corners=True
                                    )

                                    model_outputs = {
                                        'pred_depth': pred_depth_metric,  # [B, 1, gt_h, gt_w] at GT resolution
                                        'importance_map': importance_map_resized.float().cpu()  # [B, 1, gt_h, gt_w]
                                    }

                                    # For visualization, we need [B, T, ...] format like training
                                    # But we only have one frame (t=0), so unsqueeze T dimension
                                    sample_batch = (
                                        img_t_resized.unsqueeze(1).float().cpu(),  # [B, 1, C, gt_h, gt_w] at GT resolution
                                        gt_depth_metric.unsqueeze(1),  # [B, 1, gt_h, gt_w]
                                        dataset_idx
                                    )

                                    # Get FPS for visualization
                                    current_fps = None
                                    if self.measure_fps and len(self.fps_buffer) > 0:
                                        current_fps = sum(self.fps_buffer[-20:]) / min(len(self.fps_buffer), 20)

                                    # Create loss_dict with current frame loss (for visualization)
                                    val_loss_dict = {
                                        'val_loss': loss_t.item() if len(frame_losses) > 0 else 0.0
                                    }

                                    # Save with dataset and sequence-specific name
                                    save_name = f"validation_{current_dataset}_seq{seq_num:03d}_step_{self.global_step:06d}"
                                    self.val_visualizer.create_validation_summary(
                                        sample_batch, model_outputs, self.global_step,
                                        save_name=save_name, fps=current_fps, loss_dict=val_loss_dict
                                    )
                                    config['saved'].append(seq_num)
                                    self.logger.info(f"Saved validation visualization: {current_dataset} sequence {seq_num} ({len(config['saved'])}/{config['count']})")
                                except Exception as e:
                                    if self.rank == 0:
                                        self.logger.warning(f"Failed to save validation visualization for {current_dataset} seq {seq_num}: {e}")

                    # Clear intermediate tensors to free memory after each frame
                    del encoder_features, attention_weights, patch_tokens, dpt_features
                    del path_1_modulated, importance_map, fg_features, bg_features, pred_depth_inverse
                    if t > 0:  # Don't delete on first frame if we need it for visualization
                        torch.cuda.empty_cache()

                # Average loss over all frames in sequence
                if len(frame_losses) > 0:
                    avg_loss = sum(frame_losses) / len(frame_losses)
                    total_loss += avg_loss.item()
                    num_batches += 1

            # Increment sequence counter for this dataset
            if current_dataset in dataset_sequence_counters:
                dataset_sequence_counters[current_dataset] += 1

            # Clear batch memory
            del images, gt_depth, gt_depth_inverse

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Clear cache after validation
        torch.cuda.empty_cache()

        # Back to training mode
        self.model.train()

        return {'loss': avg_loss}

    def save_checkpoint(self, filename):
        """Save model checkpoint (only rank 0)"""
        if self.rank != 0:
            return

        checkpoint_path = self.results_dir / filename

        # Unwrap DDP for saving
        model = self.model.module if isinstance(self.model, DDP) else self.model

        checkpoint = {
            'global_step': self.global_step,
            'model': model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_step': self.best_step,  # Save which step was best
            'current_val_loss': self.current_val_loss,  # Save current validation loss
            'config': OmegaConf.to_container(self.config, resolve=True),
            'phase': self.phase
        }

        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")


@hydra.main(version_base=None, config_path="configs/gear3", config_name="config")
def main(config: DictConfig):
    """Main entry point"""
    # Initialize distributed training
    rank, world_size, local_rank = init_distributed()

    # Create trainer
    trainer = Gear3Trainer(config, rank, world_size, local_rank)
    trainer.train()

    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
