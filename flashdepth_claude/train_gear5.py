"""
Gear5 Training Script: Unified Single-Stage Temporal Scale Prediction

Single-stage training approach:
    - Input: 2-layer CLS tokens [11, 23] for ViT-L or [5, 11] for ViT-S
    - Processing: GRU-based temporal modeling for scale/shift
    - Output: Frame-wise scale and shift parameters
    - Applied to: Final relative depth output (after output_conv2)
    - Formula: D_metric = Scale × D_relative + Shift

Freezing Strategy:
    - Frozen: ViT encoder, DPT decoder, Mamba modules, output_conv
    - Trainable: Only Gear5MetricHead (~132K parameters)

Loss:
    - Importance-weighted Log L1 loss
    - Importance map from 2-layer attention weights
    - Weighted by foreground ratio (alpha)

Key features:
    - Temporal consistency via GRU
    - Attention-based importance weighting
    - Canonical space normalization (focal_length=500)
    - Resolution: 518×518 (Phase 1) or 2K (Phase 2)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
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
from utils.gear5_visualization import Gear5Visualizer
from flashdepth.gear5_modules import Gear5MetricHead
from dataloaders.combined_dataset import CombinedDataset
from utils.helpers import *
from utils.metric_depth_metrics import MetricDepthMetrics
from utils.gear_losses import LogL1Loss


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





class Gear5Trainer:
    """
    Trainer for Gear5 unified single-stage metric depth learning.

    Unified Stage:
        Frozen: ViT + DPT + Mamba + output_conv
        Trainable: Gear5MetricHead only (~132K parameters)
    """
    def __init__(self, config, rank, world_size, local_rank):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank

        # Detect phase based on config_variant
        # Phase 1: 518×518, config_l or config_s
        # Phase 2: 2K, config_hybrid
        config_variant = config.get('config_variant', 'l')
        self.phase = 2 if config_variant == 'hybrid' else 1

        # Setup device
        self.device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)

        if rank == 0:
            logging.info(f"Training Phase {self.phase} on {world_size} GPU(s)")
            if self.phase == 2:
                logging.info(f"  Phase 2 (Hybrid): 2K resolution, Gear5-S weights + FlashDepth-hybrid")

        # Setup results directory (only rank 0)
        phase_suffix = f"_phase{self.phase}" if self.phase > 1 else ""
        self.results_dir = Path(config.get('results_dir', f'./train_results/gear5{phase_suffix}'))
        if rank == 0:
            self.results_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging (only rank 0)
        # Force immediate flush after each log
        class FlushFileHandler(logging.FileHandler):
            def emit(self, record):
                super().emit(record)
                self.flush()  # Flush immediately

        if rank == 0:
            # Get root logger and clear any existing handlers
            root_logger = logging.getLogger()
            root_logger.setLevel(logging.INFO)
            root_logger.handlers.clear()  # Clear existing handlers

            # Create file handler with immediate flushing
            file_handler = FlushFileHandler(self.results_dir / 'training.log', mode='a')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

            # Create console handler
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.INFO)
            stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

            # Add handlers to root logger
            root_logger.addHandler(file_handler)
            root_logger.addHandler(stream_handler)
        else:
            # Other ranks: only show errors
            logging.basicConfig(level=logging.ERROR)

        self.logger = logging.getLogger(__name__)

        if rank == 0:
            self.logger.info(f"Results directory: {self.results_dir}")
            self.logger.info(f"Training phase: {self.phase}")

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

        # FPS measurement removed (only in test scripts with batch=1 for fair comparison)

        # Setup wandb
        if config.training.get('wandb', False):
            wandb.init(
                project="flashdepth-gear5",
                name=f"gear5_phase{self.phase}_{config.training.get('wandb_name', 'experiment')}",
                config=dict(config)
            )

        # Setup visualizer with separate folders
        self.train_visualizer = Gear5Visualizer(save_dir=self.results_dir / "visualizations" / "train")
        self.val_visualizer = Gear5Visualizer(save_dir=self.results_dir / "visualizations" / "valid")

        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_step = 0  # Track which step achieved best validation loss
        self.current_val_loss = None  # Track current validation loss for checkpoint
        self.dataset_losses = None  # Track per-dataset validation losses
        self.num_sequences = None  # Track number of sequences per dataset

        # Validation visualization config: track which sequences to visualize
        # Phase 2: Save 1 sample per dataset at every validation step (for step-by-step comparison)
        # Sintel is smaller (1022×434), Waymo_seg is 2K (1918×1274)
        if self.phase >= 2:
            self.val_vis_config = {
                'sintel': {'sequences': [0], 'saved': []},  # seq 0 only
                'waymo_seg': {'sequences': [0], 'saved': []}    # seq 0 only
            }
        else:
            # Phase 1: Can afford more samples (518x518, no batch limit)
            self.val_vis_config = {
                'sintel': {'sequences': [0, 4, 7], 'saved': []},  # seq 0, 4, 7 (3 samples)
                'waymo_seg': {'sequences': [0, 1, 2, 3, 4, 5, 6, 7], 'saved': []}  # all 8 sequences
            }

    def _get_canonical_focal_length(self):
        """
        Get canonical focal length (fixed at 500.0 for 518×518 resolution).

        Returns:
            float: Canonical focal length (always 500.0)
        """
        return 500.0

    def _setup_model(self):
        """Initialize FlashDepth with Gear5 metric head"""
        # Create base FlashDepth model
        model_config = dict(self.config.model)
        model_config['batch_size'] = self.config.training.batch_size
        model_config['use_metric_head'] = False  # Don't use original GSP head

        model = FlashDepth(**model_config)

        # Get model architecture info
        embed_dim = 1024 if model.encoder == 'vitl' else 384
        dpt_dim = 256 if model.encoder == 'vitl' else 64

        # Load pre-trained checkpoint
        # Phase 1: Load FlashDepth-L checkpoint (ViT, DPT, Mamba, output_conv)
        # Phase 2: Load Phase 1 Gear5 checkpoint, then overwrite ViT+DPT with FlashDepth-S hybrid
        if self.phase == 1:
            checkpoint_path = self.config.get('load')
        else:
            # Phase 2: Load Gear5 Phase 1 checkpoint
            checkpoint_path = self.config.get('gear_checkpoint')
            if not checkpoint_path:
                raise ValueError(
                    "Phase 2 (Hybrid) requires 'gear_checkpoint' in config! "
                    "Set gear_checkpoint to your Gear5 Phase 1 checkpoint path."
                )
            if self.rank == 0:
                self.logger.info(f"Phase {self.phase}: Will load Gear5 Phase 1 checkpoint from {checkpoint_path}")

        # Phase 1: Load FlashDepth checkpoint (ViT, DPT, Mamba, output_conv)
        if self.phase == 1 and checkpoint_path and checkpoint_path != 'true':
            if os.path.exists(checkpoint_path):
                self.logger.info(f"Phase 1: Loading FlashDepth checkpoint from {checkpoint_path}")
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

                # Load ALL parameters from FlashDepth (exclude only gear5_metric_head)
                loaded_dict = {}
                excluded_keys = []
                for k, v in state_dict.items():
                    if 'gear5_metric_head' in k:
                        excluded_keys.append(k)
                    else:
                        loaded_dict[k] = v

                # Load state dict (strict=False to allow missing gear5_metric_head)
                model.load_state_dict(loaded_dict, strict=False)
                self.logger.info(f"Phase 1: Loaded {len(loaded_dict)} parameters from FlashDepth checkpoint")
                self.logger.info(f"  - DINOv2 encoder: ✓ (will be frozen)")
                self.logger.info(f"  - DPT decoder: ✓ (will be frozen)")
                self.logger.info(f"  - Mamba modules: ✓ (will be frozen)")
                self.logger.info(f"  - output_conv1/2: ✓ (will be frozen)")
                if excluded_keys:
                    self.logger.info(f"Excluded {len(excluded_keys)} parameters (gear5_metric_head will be created)")
            else:
                self.logger.warning(f"Checkpoint {checkpoint_path} not found")

        # Add Gear5 metric head (unified single-stage)
        # Uses 2-layer CLS tokens and GRU for temporal modeling
        model.gear5_metric_head = Gear5MetricHead(
            embed_dim=embed_dim,
            feature_dim=256,
            hidden_dim=128
        )

        # Phase 2: Load Phase 1 checkpoint, then overwrite ViT+DPT with FlashDepth-S hybrid
        if self.phase == 2 and checkpoint_path and checkpoint_path != 'true' and os.path.exists(checkpoint_path):
            # Step 1: Load Phase 1 Gear5 checkpoint (all components including gear5_metric_head)
            self.logger.info(f"Phase 2: Loading Gear5 Phase 1 checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            # Load all parameters from Phase 1
            model.load_state_dict(state_dict, strict=False)
            self.logger.info(f"Phase 2: Loaded Gear5 Phase 1 checkpoint")
            self.logger.info(f"  - ViT (ViT-L): ✓ (will be overwritten with hybrid)")
            self.logger.info(f"  - DPT: ✓ (will be overwritten with hybrid)")
            self.logger.info(f"  - Mamba: ✓ (kept from Phase 1, will be frozen)")
            self.logger.info(f"  - output_conv: ✓ (kept from Phase 1, will be frozen)")
            self.logger.info(f"  - Gear5MetricHead: ✓ (kept from Phase 1, trainable)")

            # Step 2: Overwrite ViT + DPT with FlashDepth-S hybrid weights
            hybrid_path = self.config.get('load', 'configs/flashdepth/iter_43002.pth')
            if os.path.exists(hybrid_path):
                self.logger.info(f"Phase 2: Loading FlashDepth-S hybrid weights from {hybrid_path}")
                hybrid_checkpoint = torch.load(hybrid_path, map_location='cpu')

                # Extract state dict
                if isinstance(hybrid_checkpoint, dict) and 'model' in hybrid_checkpoint:
                    hybrid_state_dict = hybrid_checkpoint['model']
                elif isinstance(hybrid_checkpoint, dict) and 'state_dict' in hybrid_checkpoint:
                    hybrid_state_dict = hybrid_checkpoint['state_dict']
                else:
                    hybrid_state_dict = hybrid_checkpoint

                # Remove module. prefix if present
                hybrid_state_dict = {k.replace('module.', ''): v for k, v in hybrid_state_dict.items()}

                # Load ONLY ViT and DPT parameters (exclude Mamba, output_conv, gear5_metric_head)
                loaded_hybrid = {}
                for k, v in hybrid_state_dict.items():
                    # Include only encoder and DPT (exclude Mamba, output_conv, gear5_metric_head)
                    if not any(x in k for x in ['mamba', 'hybrid_fusion', 'teacher_model',
                                                 'output_conv1', 'output_conv2', 'gear5_metric_head']):
                        loaded_hybrid[k] = v

                # Overwrite ViT + DPT parameters with hybrid weights
                model.load_state_dict(loaded_hybrid, strict=False)
                self.logger.info(f"Phase 2: Overwritten {len(loaded_hybrid)} ViT + DPT parameters with hybrid weights")
                self.logger.info(f"  - ViT (ViT-L + ViT-S + Cross Attn): ✓")
                self.logger.info(f"  - DPT: ✓")
                self.logger.info(f"  - Kept from Phase 1: Mamba (frozen), output_conv (frozen), Gear5MetricHead (trainable)")
            else:
                self.logger.warning(f"FlashDepth-S hybrid checkpoint {hybrid_path} not found! Using Phase 1 ViT + DPT.")

        # Enable attention weights storage
        # Unified single-stage: use 2 layers for CLS token extraction and importance mapping
        # Layers [11, 23] for ViT-L or [5, 11] for ViT-S
        target_blocks = {
            'vitl': [11, 23],
            'vits': [5, 11]
        }[model.encoder]

        for i, block in enumerate(model.pretrained.blocks):
            if i in target_blocks:
                block.attn.store_attn_weights = True
                self.logger.info(f"Enabled attention weights storage for block {i}")
            else:
                block.attn.store_attn_weights = False

        self.logger.info(f"2-layer attention storage: blocks {target_blocks}")

        # Store target blocks and compute encoder_features indices
        # encoder_features from get_intermediate_layers returns features at intermediate_layer_idx
        # For vitl: intermediate_layer_idx = [4, 11, 17, 23], target_blocks = [11, 23]
        #           encoder_indices = [1, 3] (indices for layers 11 and 23)
        # For vits: intermediate_layer_idx = [2, 5, 8, 11], target_blocks = [5, 11]
        #           encoder_indices = [1, 3] (indices for layers 5 and 11)
        intermediate_idx = model.intermediate_layer_idx[model.encoder]
        encoder_indices = [intermediate_idx.index(block) for block in target_blocks]
        self.encoder_indices = encoder_indices
        self.logger.info(f"Encoder features indices: {encoder_indices} (for CLS token extraction)")

        # Store target blocks for use in training/validation steps
        self.target_blocks = target_blocks

        model = model.to(self.device)

        # Freeze and configure parameters
        self._configure_parameters(model)

        # Apply gradient checkpointing before DDP wrapping (critical for 2K resolution)
        if self.config.training.get('gradient_checkpointing', False):
            if self.rank == 0:
                self.logger.info("Applying gradient checkpointing to ViT and DPT (saves ~50% memory)")
            apply_activation_checkpointing(
                getattr(model, 'pretrained'),  # DINOv2 ViT
                checkpoint_wrapper_fn=checkpoint_wrapper,
                check_fn=lambda _: True
            )
            apply_activation_checkpointing(
                getattr(model, 'depth_head'),  # DPT
                checkpoint_wrapper_fn=checkpoint_wrapper,
                check_fn=lambda _: True
            )
            # Note: Mamba is not compatible with gradient checkpointing (recurrent architecture)

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
        Unified single-stage freezing strategy:
            Frozen: ViT encoder, DPT decoder, Mamba modules, output_conv
            Trainable: Only Gear5MetricHead (~132K parameters)
        """
        frozen_vit_dpt = 0
        frozen_mamba = 0
        frozen_output_conv = 0
        trainable_gear5 = 0

        for name, param in model.named_parameters():
            # Gear5 metric head: always trainable
            if 'gear5_metric_head' in name:
                param.requires_grad = True
                trainable_gear5 += param.numel()
                self.logger.info(f"Trainable (Gear5): {name} - {param.shape}")

            # Mamba: frozen
            elif 'mamba' in name:
                param.requires_grad = False
                frozen_mamba += param.numel()

            # DPT output head: frozen
            elif 'output_conv' in name:
                param.requires_grad = False
                frozen_output_conv += param.numel()

            # Everything else (ViT encoder, DPT decoder): frozen
            else:
                param.requires_grad = False
                frozen_vit_dpt += param.numel()

        # Log summary
        self.logger.info(f"=== Parameter Configuration (Phase {self.phase}) ===")
        self.logger.info(f"Frozen:")
        self.logger.info(f"  - ViT + DPT: {frozen_vit_dpt:,}")
        self.logger.info(f"  - Mamba: {frozen_mamba:,}")
        self.logger.info(f"  - output_conv: {frozen_output_conv:,}")

        self.logger.info(f"Trainable:")
        self.logger.info(f"  - Gear5MetricHead: {trainable_gear5:,}")

        total_frozen = frozen_vit_dpt + frozen_mamba + frozen_output_conv
        self.logger.info(f"Total frozen: {total_frozen:,}")
        self.logger.info(f"Total trainable: {trainable_gear5:,}")

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
            if any(keyword in name for keyword in ['gear5_metric_head', 'mamba', 'output_conv']):
                continue

            # Set frozen parts to eval mode
            module.eval()

    def _setup_data_loaders(self):
        """Setup phase-specific data loaders"""
        if self.phase == 1:
            # Phase 1: 5 datasets, 518×518
            train_datasets = self.config.dataset.get('train_datasets',
                ['mvs-synth', 'dynamicreplica', 'tartanair', 'pointodyssey', 'spring'])
            val_datasets = self.config.dataset.get('val_datasets', ['sintel', 'waymo'])
            resolution = 'base'  # 518×518
        else:
            # Phase 2: mvs-synth, spring only, 2K resolution (Hybrid)
            train_datasets = self.config.dataset.get('train_datasets', ['mvs-synth', 'spring'])
            val_datasets = self.config.dataset.get('val_datasets', ['sintel', 'waymo_seg'])
            resolution = '2k'

        if self.rank == 0:
            self.logger.info(f"Phase {self.phase} - Train datasets: {train_datasets}")
            self.logger.info(f"Phase {self.phase} - Val datasets: {val_datasets}")
            self.logger.info(f"Phase {self.phase} - Resolution: {resolution}")

        # Training dataset
        train_dataset = CombinedDataset(
            root_dir=self.config.dataset.data_root,
            enable_dataset_flags=train_datasets,
            resolution=resolution,  # Use phase-specific resolution
            split='train',
            video_length=self.config.dataset.video_length,
            color_aug=False  # No augmentation for metric training
        )

        # Validation dataset: use shorter sequence for Phase 2 to save memory
        # Phase 2 uses 2K resolution which requires much more memory
        val_video_length = 1 if self.phase >= 2 else self.config.dataset.video_length
        val_dataset = CombinedDataset(
            root_dir=self.config.dataset.data_root,
            enable_dataset_flags=val_datasets,
            resolution=resolution,  # Use phase-specific resolution
            split='val',
            video_length=val_video_length  # Single frame for Phase 2/3 (2K resolution)
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
            # Validation: No DistributedSampler - all ranks see the same data
            # This ensures rank 0 can save visualizations for all sequences
            val_sampler = None
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
            self.logger.info(f"Val batch size: 1, video_length: {val_video_length} {'(reduced for 2K)' if self.phase >= 2 else '(like original FlashDepth)'}")

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
                if 'gear5_metric_head' in name:
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
                'lr_g3': f'{lr_gear3:.2e}',
                'lr_mb': f'{lr_mamba:.2e}'
            }

            pbar.set_postfix(postfix_dict)

            # WandB logging (every step)
            wandb_dict = {**loss_dict, 'lr_gear3': lr_gear3, 'lr_mamba': lr_mamba}
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
                    images, gt_depth, focal_lengths_canonical, focal_lengths_actual, actual_valid_masks, dataset_idx = batch
                    images = images.to(self.device)
                    gt_depth = gt_depth.to(self.device)
                    focal_lengths_canonical = focal_lengths_canonical.to(self.device)
                    focal_lengths_actual = focal_lengths_actual.to(self.device)
                    actual_valid_masks = actual_valid_masks.to(self.device)

                    if gt_depth.ndim == 3:
                        gt_depth = gt_depth.unsqueeze(1)
                    elif gt_depth.ndim == 4 and gt_depth.shape[1] != 1:
                        gt_depth = gt_depth.unsqueeze(2)

                    # GT depth from dataloader is inverse depth (1/m), scale to 100/m for training
                    gt_depth_inverse_100 = gt_depth * 100.0  # (1/m) * 100 = 100/m

                    # Get batch size for Mamba initialization
                    B_orig, T_orig, C, H, W = images.shape

                    # NOTE: GT depth is already in canonical space (fx=500 at 518×518)
                    # Canonical transformation is now handled in the dataloader

                    # Initialize Mamba sequence for temporal processing
                    if hasattr(model, 'mamba'):
                        model.mamba.start_new_sequence()

                    # Process entire sequence at once (full sequence processing)
                    images_flat = rearrange(images, 'b t c h w -> (b t) c h w')
                    patch_h, patch_w = H // model.patch_size, W // model.patch_size

                    with torch.no_grad():
                        # Extract features from all frames at once
                        encoder_features = model.pretrained.get_intermediate_layers(
                            images_flat, model.intermediate_layer_idx[model.encoder]
                        )
                        last_block = model.pretrained.blocks[-1]
                        attention_weights = last_block.attn.attn_weights
                        patch_tokens = encoder_features[-1]

                        # Get DPT features WITHOUT Mamba
                        dpt_features = model.depth_head.get_forward_features(
                            encoder_features, patch_h, patch_w
                        )
                        path_1 = dpt_features[-1]  # Extract path_1

                        # Prepare inputs based on separation_method
                        attention_weights_multi_layer = None
                        cls_token = None

                        # Gear5: Collect CLS tokens and attention weights from multiple layers
                        # Collect 2-layer CLS tokens and average them
                        cls_tokens_list = [
                            encoder_features[i][:, 0]  # CLS token from each layer [B*T, embed_dim]
                            for i in self.encoder_indices
                        ]
                        # Average across layers and reshape to [B, T, embed_dim]
                        cls_tokens_avg = torch.stack(cls_tokens_list, dim=0).mean(dim=0)  # [B*T, embed_dim]
                        cls_tokens = rearrange(cls_tokens_avg, '(b t) d -> b t d', b=B_orig, t=T_orig)  # [B, T, embed_dim]

                        # Collect attention weights from 2 target layers
                        attention_weights_list = [
                            model.pretrained.blocks[block_idx].attn.attn_weights  # [B*T, num_heads, N+1, N+1]
                            for block_idx in self.target_blocks
                        ]

                        # Apply Gear5 metric head (no feature modulation, just scale/shift/importance prediction)
                        gear5_outputs = model.gear5_metric_head(
                            cls_tokens=cls_tokens,
                            attention_weights_list=attention_weights_list,
                            patch_h=patch_h,
                            patch_w=patch_w
                        )

                        scale = gear5_outputs['scale']  # [B, T]
                        shift = gear5_outputs['shift']  # [B, T]
                        importance_map = gear5_outputs['importance_map']  # [B, T, patch_h, patch_w]

                        # Apply Mamba temporal modeling to path_1 (no modulation in Gear5)
                        path_1_temporal = model.dpt_features_to_mamba(
                            input_shape=(B_orig, T_orig, None, H, W),
                            dpt_features=path_1,
                            in_dpt_layer=0
                        )

                        out = model.depth_head.scratch.output_conv1(path_1_temporal)
                        out = F.interpolate(out, (H, W), mode="bilinear", align_corners=True)
                        relative_depth = model.depth_head.scratch.output_conv2(out)  # [B*T, 1, H, W]

                        # Apply scale/shift to relative depth (inverse depth space: 100/m)
                        scale_flat = scale.view(B_orig * T_orig, 1, 1, 1)  # [B*T, 1, 1, 1]
                        shift_flat = shift.view(B_orig * T_orig, 1, 1, 1)  # [B*T, 1, 1, 1]
                        pred_depth_inverse_100 = scale_flat * relative_depth + shift_flat  # [B*T, 1, H, W] in 100/m

                        # Save prediction inverse depth for mask calculation
                        pred_depth_inverse = pred_depth_inverse_100  # [B*T, 1, H, W] inverse depth (100/m)

                        # Convert prediction to metric depth: 100/m -> m
                        # Output shape: (B*T, 1, H, W)
                        pred_depth_metric = 100.0 / (pred_depth_inverse_100 + 1e-8)

                        # Reshape prediction to (B, T, ...) and select first frame
                        pred_depth_inverse_seq = rearrange(pred_depth_inverse, '(b t) 1 h w -> b t 1 h w', b=B_orig, t=T_orig)
                        pred_depth_metric_seq = rearrange(pred_depth_metric, '(b t) 1 h w -> b t 1 h w', b=B_orig, t=T_orig)
                        pred_depth_inverse_vis = pred_depth_inverse_seq[:, 0]  # [B, 1, H, W]
                        pred_depth_metric_vis = pred_depth_metric_seq[:, 0]  # [B, 1, H, W]

                        # GT depth for first frame
                        gt_depth_inverse_vis = gt_depth_inverse_100[:, 0]  # [B, 1, H, W]
                        gt_depth_metric = 100.0 / (gt_depth_inverse_vis + 1e-8)

                        # Compute canonical masks for visualization (70m threshold in canonical space)
                        # Use same logic as training loss calculation (but no warmup)
                        MIN_INVERSE_DEPTH_VIS = 100.0 / 70.0  # Canonical space 70m
                        canonical_gt_valid_vis = (gt_depth_inverse_vis > MIN_INVERSE_DEPTH_VIS)  # [B, 1, H, W]

                        # Training: Pred outlier filtering (200m, like training loss)
                        MAX_DEPTH_OUTLIER_VIS = 200.0
                        MIN_INVERSE_OUTLIER_VIS = 100.0 / MAX_DEPTH_OUTLIER_VIS
                        canonical_pred_valid_vis = (pred_depth_inverse_vis > MIN_INVERSE_OUTLIER_VIS)  # [B, 1, H, W]

                        # Move tensors to CPU for visualization (only first batch, first frame)
                        sample_batch = (
                            images[:1, :1].float().cpu(),  # [1, 1, 3, H, W] - convert BFloat16 to Float32 first
                            gt_depth_metric[:1].float().cpu(),  # [1, 1, H, W]
                            dataset_idx,
                            focal_lengths_actual[:1, :1].float().cpu()  # [1, 1] - original focal length
                        )

                        # importance_map is already [B, T, patch_h, patch_w]
                        # Select first frame for visualization
                        importance_map_vis = importance_map[:, 0]  # [B, patch_h, patch_w]
                        importance_map_vis = importance_map_vis.unsqueeze(1)  # [B, 1, patch_h, patch_w]

                        # Resize importance map to match image resolution
                        importance_map_resized = F.interpolate(
                            importance_map_vis, size=(H, W), mode='bilinear', align_corners=True
                        )

                        # Include outputs with importance map + canonical masks + scale/shift
                        model_outputs_cpu = {
                            'pred_depth': pred_depth_metric_vis[:1].cpu(),  # [1, 1, H, W]
                            'importance_map': importance_map_resized[:1].cpu(),  # [1, 1, H, W]
                            'canonical_gt_valid': canonical_gt_valid_vis[:1].cpu(),  # [1, 1, H, W] - canonical space mask
                            'canonical_pred_valid': canonical_pred_valid_vis[:1].cpu(),  # [1, 1, H, W] - canonical space mask
                            'scale': scale[:1, :1].cpu(),  # [1, 1] - first batch, first frame
                            'shift': shift[:1, :1].cpu()   # [1, 1] - first batch, first frame
                        }

                        # FPS removed from training (only measured in test_gear3.py)
                        current_fps = None

                        # No layer_weights in unified Gear5 (GRU-based, not multi-layer fusion)
                        # Pass loss_dict for visualization
                        self.train_visualizer.create_validation_summary(
                            sample_batch, model_outputs_cpu, step, prefix="training", fps=current_fps, loss_dict=loss_dict, config=self.config
                        )

                    self._set_train_mode()

                except Exception as e:
                    self.logger.warning(f"Failed to save training visualization: {e}")
                    self._set_train_mode()

            # Validation (run at step 0 and every val_freq steps)
            # Only validate on main process (rank 0) to avoid duplicate validation
            if step % self.config.training.get('val_freq', 1000) == 0:
                if self.rank == 0:
                    val_metrics = self.validate()
                    self.logger.info(f"Validation at step {step}: {val_metrics}")

                    # Update current validation loss for checkpoint
                    self.current_val_loss = val_metrics['loss']
                    self.dataset_losses = val_metrics.get('dataset_losses', None)
                    self.num_sequences = val_metrics.get('num_sequences', None)

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
                self.save_checkpoint(f'checkpoint_step{step}.pth')

        # Final save
        self.save_checkpoint('last.pth')
        self.logger.info("Training completed!")

    def train_step(self, batch):
        """Single training step with BFloat16 autocast"""
        # Unpack batch (now includes fx_canonical, fx_actual, and actual_valid_mask)
        images, gt_depth, focal_lengths_canonical, focal_lengths_actual, actual_valid_mask, dataset_idx = batch
        images = images.to(self.device)
        gt_depth = gt_depth.to(self.device)
        focal_lengths_canonical = focal_lengths_canonical.to(self.device)  # Shape: (B, T), all 500.0 (canonical)
        focal_lengths_actual = focal_lengths_actual.to(self.device)  # Shape: (B, T), original focal lengths
        actual_valid_mask = actual_valid_mask.to(self.device)  # Shape: (B, T, H, W), actual space <70m

        # Add channel dimension if needed
        if gt_depth.ndim == 3:
            gt_depth = gt_depth.unsqueeze(1)
        elif gt_depth.ndim == 4 and gt_depth.shape[1] != 1:
            gt_depth = gt_depth.unsqueeze(2)

        B, T = images.shape[:2]

        # Get the actual model (unwrap DDP if needed)
        model = self.model.module if isinstance(self.model, DDP) else self.model

        # DEBUG: Check raw GT depth values
        if self.global_step < 5:  # Only log first few steps
            self.logger.info(f"DEBUG - Raw GT depth from dataloader: min={gt_depth.min():.4f}, max={gt_depth.max():.4f}, shape={gt_depth.shape}")
            self.logger.info(f"DEBUG - GT depth has {(gt_depth > 0).sum()} valid pixels out of {gt_depth.numel()} total")

        # GT depth from dataloader is inverse depth (1/m), scale to 100/m for training
        # This matches FlashDepth's relative depth scale (≈ 100/metric_depth)
        gt_depth_inverse_100 = gt_depth * 100.0  # (1/m) * 100 = 100/m

        # DEBUG: Check after scaling to 100/m
        if self.global_step < 5:
            self.logger.info(f"DEBUG - After scaling to 100/m: min={gt_depth_inverse_100.min():.4f}, max={gt_depth_inverse_100.max():.4f}")
            self.logger.info(f"DEBUG - Has {(gt_depth_inverse_100 > 0).sum()} valid pixels")

        # NOTE: GT depth is already in canonical space (fx=500 at 518×518)
        # Canonical transformation is now handled in the dataloader
        # focal_lengths_canonical tensor contains all 500.0 values (canonical focal length)
        # focal_lengths_actual contains original focal lengths from datasets
        CANONICAL_FX = self._get_canonical_focal_length()  # 500.0

        if self.global_step < 5:
            self.logger.info(f"DEBUG - Canonical space: CANONICAL_FX={CANONICAL_FX}, fx_canonical from dataloader: {focal_lengths_canonical[0,0]:.1f}")
            self.logger.info(f"DEBUG - Original fx_actual from dataloader: {focal_lengths_actual[0,0]:.1f}")
            self.logger.info(f"DEBUG - GT depth already in canonical space (transformed in dataloader)")

        # Forward pass following original FlashDepth pattern (whole sequence at once)
        # Initialize Mamba sequence (critical for temporal processing!)
        if hasattr(model, 'mamba'):
            model.mamba.start_new_sequence()

        # Reshape video from (B, T, C, H, W) to (B*T, C, H, W) for encoder
        B_orig, T_orig, C, H, W = images.shape
        images_flat = rearrange(images, 'b t c h w -> (b t) c h w')

        patch_h, patch_w = H // model.patch_size, W // model.patch_size

        # Use BFloat16 for forward pass
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # Extract features from DINOv2 (frozen, no grad) - all frames at once
            with torch.no_grad():
                encoder_features = model.pretrained.get_intermediate_layers(
                    images_flat, model.intermediate_layer_idx[model.encoder]
                )

                # Get attention weights from last block (B*T, num_heads, N, N)
                last_block = model.pretrained.blocks[-1]
                attention_weights = last_block.attn.attn_weights

                # Get patch tokens from last encoder layer (B*T, N, C)
                patch_tokens = encoder_features[-1]

            # Get DPT features WITHOUT Mamba (DPT frozen, no grad)
            with torch.no_grad():
                dpt_features = model.depth_head.get_forward_features(
                    encoder_features, patch_h, patch_w
                )  # Returns [path_4, path_3, path_2, path_1], each (B*T, dpt_dim, h, w)
                path_1 = dpt_features[-1]  # Extract path_1: (B*T, dpt_dim, h, w)

            # Gear5: Extract 2-layer CLS tokens and attention weights
            # For ViT-L: layers [11, 23], encoder_indices = [1, 3]
            # For ViT-S: layers [5, 11], encoder_indices = [1, 3]
            with torch.no_grad():
                cls_tokens_list = [
                    encoder_features[i][:, 0]  # CLS token: [B*T, embed_dim]
                    for i in self.encoder_indices
                ]
                attention_weights_list = [
                    model.pretrained.blocks[block_idx].attn.attn_weights  # [B*T, num_heads, N+1, N+1]
                    for block_idx in self.target_blocks
                ]

                # Average 2-layer CLS tokens and reshape to [B, T, embed_dim]
                cls_tokens_avg = torch.stack(cls_tokens_list, dim=0).mean(dim=0)  # [B*T, embed_dim]
                cls_tokens = rearrange(cls_tokens_avg, '(b t) d -> b t d', b=B_orig, t=T_orig)  # [B, T, embed_dim]

            # Get DPT features (frozen, no grad)
            with torch.no_grad():
                dpt_features = model.depth_head.get_forward_features(
                    encoder_features, patch_h, patch_w
                )
                path_1 = dpt_features[-1]  # [B*T, dpt_dim, h, w]

            # Apply Mamba temporal modeling (frozen)
            with torch.no_grad():
                path_1_temporal = model.dpt_features_to_mamba(
                    input_shape=(B_orig, T_orig, None, H, W),
                    dpt_features=path_1,
                    in_dpt_layer=0
                )  # [B*T, dpt_dim, h, w]

            # Get relative depth from frozen output_conv
            with torch.no_grad():
                out = model.depth_head.scratch.output_conv1(path_1_temporal)
                out = F.interpolate(out, (H, W), mode="bilinear", align_corners=True)
                relative_depth = model.depth_head.scratch.output_conv2(out)  # [B*T, 1, H, W]

            # Get scale/shift/importance_map from Gear5MetricHead (trainable)
            gear5_outputs = model.gear5_metric_head(
                cls_tokens=cls_tokens,  # [B, T, 1024]
                attention_weights_list=attention_weights_list,  # List of 2 attention weights
                patch_h=patch_h,
                patch_w=patch_w
            )

            scale = gear5_outputs['scale']  # [B, T]
            shift = gear5_outputs['shift']  # [B, T]
            importance_map = gear5_outputs['importance_map']  # [B, T, patch_h, patch_w]

            # DEBUG: Check scale/shift values at validation steps
            if self.rank == 0 and (self.global_step % 1000 == 0 or self.global_step < 10):
                scale_mean = scale.mean().item()
                scale_min = scale.min().item()
                scale_max = scale.max().item()
                shift_mean = shift.mean().item()
                shift_min = shift.min().item()
                shift_max = shift.max().item()
                self.logger.info(f"VALIDATION Step {self.global_step} - Scale: mean={scale_mean:.4f}, range=[{scale_min:.4f}, {scale_max:.4f}]")
                self.logger.info(f"VALIDATION Step {self.global_step} - Shift: mean={shift_mean:.4f}, range=[{shift_min:.4f}, {shift_max:.4f}]")
                self.logger.info(f"VALIDATION Step {self.global_step} - Relative depth: mean={relative_depth.mean():.4f}, range=[{relative_depth.min():.4f}, {relative_depth.max():.4f}]")

            # Apply shift lower bound constraint (Option 1)
            # Ensure pred stays positive even for far depths (e.g., 300m → inverse = 0.333)
            GLOBAL_MIN_INVERSE = 100.0 / 300.0  # 0.333 (300m max depth assumption)
            shift_lower_bound = -scale.abs() * GLOBAL_MIN_INVERSE  # [B, T]
            shift_clamped = torch.clamp(shift, min=shift_lower_bound)  # [B, T]

            # Apply scale/shift to relative depth with clamped shift
            scale_flat = scale.view(B_orig * T_orig, 1, 1, 1)  # [B*T, 1, 1, 1]
            shift_flat = shift_clamped.view(B_orig * T_orig, 1, 1, 1)  # [B*T, 1, 1, 1]
            pred_depth_inverse_100 = scale_flat * relative_depth + shift_flat  # [B*T, 1, H, W] in 100/m

            # DEBUG: Check pred_depth_inverse_100 after scale/shift
            if self.rank == 0 and (self.global_step % 1000 == 0 or self.global_step < 10):
                self.logger.info(f"TRAINING Step {self.global_step} - Shift clamped: mean={shift_clamped.mean():.4f}, range=[{shift_clamped.min():.4f}, {shift_clamped.max():.4f}]")
                self.logger.info(f"TRAINING Step {self.global_step} - Pred inverse 100: mean={pred_depth_inverse_100.mean():.4f}, range=[{pred_depth_inverse_100.min():.4f}, {pred_depth_inverse_100.max():.4f}]")

        # Reshape GT from (B, T, 1, H, W) to (B*T, 1, H, W)
        gt_depth_inverse_flat = rearrange(gt_depth_inverse_100, 'b t 1 h w -> (b t) 1 h w')

        # Flatten for loss computation
        pred_depth_flat = pred_depth_inverse_100.flatten()  # [B*T*H*W] inverse depth (100/m)
        gt_depth_flat = gt_depth_inverse_flat.flatten()  # [B*T*H*W] inverse depth (100/m)

        # Valid mask: GT valid + Pred positive (no 70m restriction in training, like original FlashDepth)
        valid_mask = (gt_depth_flat >= 0) & (pred_depth_flat > 0)

        if valid_mask.sum() == 0:
            self.logger.error("No valid GT pixels in batch!")
            return {'loss': 0.0}

        # Resize importance map to depth map size and flatten
        importance_map_resized = F.interpolate(
            importance_map.view(B_orig * T_orig, 1, patch_h, patch_w),
            size=(H, W),
            mode='bilinear',
            align_corners=True
        )  # [B*T, 1, H, W]
        importance_flat = importance_map_resized.flatten()  # [B*T*H*W]

        # Compute fg_ratio (alpha) - fraction of high-importance pixels
        importance_threshold = importance_flat.mean()
        fg_mask = (importance_flat > importance_threshold)
        fg_ratio = fg_mask.float().mean()

        # Importance-weighted Log L1 Loss (in inverse depth space)
        with torch.amp.autocast('cuda', enabled=False):
            epsilon = 1e-3
            # pred_depth_flat and gt_depth_flat are already inverse depth (100/m)
            # Compute log L1 loss directly in inverse depth space
            loss = torch.abs(
                torch.log(pred_depth_flat.float() + epsilon) -
                torch.log(gt_depth_flat.float() + epsilon)
            )
            weighted_loss = loss * (1.0 + fg_ratio * importance_flat.float())
            final_loss = weighted_loss[valid_mask].mean()

        # Backward pass (outside autocast for numerical stability)
        self.optimizer.zero_grad()
        final_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        return {
            'loss': final_loss.item()
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

        # Track per-dataset losses for detailed analysis
        dataset_losses = {}  # {dataset_name: [loss1, loss2, ...]}
        dataset_sequence_counts = {}  # {dataset_name: num_sequences_evaluated}

        # Reset visualization tracking for this validation run
        # This ensures we save the SAME sequences at EVERY validation step (for step-by-step comparison)
        for dataset_name in self.val_vis_config:
            self.val_vis_config[dataset_name]['saved'] = []

        # Track dataset-specific sequence counters for visualization
        dataset_sequence_counters = {'sintel': 0, 'waymo_seg': 0}

        # Phase 2: Limit validation batches to save memory (2K resolution)
        # Dataset-specific limits: sintel 4개, waymo_seg 2개 (총 6 batches)
        if self.phase >= 2:
            max_val_batches = 6  # Total max batches
            dataset_max_sequences = {'sintel': 4, 'waymo_seg': 2}
        else:
            max_val_batches = None
            dataset_max_sequences = {}

        total_processed = 0  # Track total processed sequences across all datasets

        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation", disable=(self.rank != 0))):
            # Skip None batches (all items were invalid)
            if batch is None:
                continue

            # Unpack batch (now includes fx_canonical, fx_actual, and actual_valid_mask)
            images, gt_depth, focal_lengths_canonical, focal_lengths_actual, actual_valid_mask, dataset_idx = batch

            # Get dataset name for this batch (before frame loop)
            if isinstance(dataset_idx, str):
                current_dataset = dataset_idx
            elif isinstance(dataset_idx, (list, tuple)):
                current_dataset = str(dataset_idx[0])
            elif torch.is_tensor(dataset_idx):
                current_dataset = str(dataset_idx[0].item() if dataset_idx.dim() > 0 else dataset_idx.item())
            else:
                current_dataset = str(dataset_idx)

            # Check dataset-specific limit (Phase 2/3 only)
            if dataset_max_sequences and current_dataset in dataset_max_sequences:
                if dataset_sequence_counters.get(current_dataset, 0) >= dataset_max_sequences[current_dataset]:
                    if self.rank == 0 and dataset_sequence_counters.get(current_dataset, 0) == dataset_max_sequences[current_dataset]:
                        self.logger.info(f"  [{current_dataset}] Reached max {dataset_max_sequences[current_dataset]} sequences, skipping further...")
                    continue

            # Check total batch limit
            if max_val_batches is not None and total_processed >= max_val_batches:
                break

            # Use BFloat16 autocast like original FlashDepth
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                images = images.to(self.device)
                gt_depth = gt_depth.to(self.device)
                focal_lengths_canonical = focal_lengths_canonical.to(self.device)  # Shape: (B, T), all 500.0
                focal_lengths_actual = focal_lengths_actual.to(self.device)  # Shape: (B, T), original focal lengths
                actual_valid_mask = actual_valid_mask.to(self.device)  # Shape: (B, T, H, W)

                # Add channel dimension if needed
                if gt_depth.ndim == 3:
                    gt_depth = gt_depth.unsqueeze(1)
                elif gt_depth.ndim == 4 and gt_depth.shape[1] != 1:
                    gt_depth = gt_depth.unsqueeze(2)

                B, T = images.shape[:2]

                # GT depth from dataloader is inverse depth (1/m), scale to 100/m for validation
                gt_depth_inverse_100 = gt_depth * 100.0  # (1/m) * 100 = 100/m

                # NOTE: GT depth is already in canonical space (transformed in dataloader)
                # No need to apply canonical transformation here

                # Initialize Mamba sequence for temporal processing
                if hasattr(model, 'mamba'):
                    model.mamba.start_new_sequence()

                # Process entire sequence at once (like training)
                B_orig, T_orig, C, H, W = images.shape
                images_flat = rearrange(images, 'b t c h w -> (b t) c h w')
                patch_h, patch_w = H // model.patch_size, W // model.patch_size

                # Extract features from DINOv2 - all frames at once
                encoder_features = model.pretrained.get_intermediate_layers(
                    images_flat, model.intermediate_layer_idx[model.encoder]
                )

                # Get attention weights from last block (B*T, num_heads, N, N)
                last_block = model.pretrained.blocks[-1]
                attention_weights = last_block.attn.attn_weights
                patch_tokens = encoder_features[-1]

                # Get DPT features WITHOUT Mamba
                dpt_features = model.depth_head.get_forward_features(
                    encoder_features, patch_h, patch_w
                )  # Returns [path_4, path_3, path_2, path_1]
                path_1 = dpt_features[-1]  # Extract path_1: (B*T, dpt_dim, h, w)

                # Gear5: Extract 2-layer CLS tokens and attention weights
                # For ViT-L: layers [11, 23], encoder_indices = [1, 3]
                # For ViT-S: layers [5, 11], encoder_indices = [1, 3]
                cls_tokens_list = [
                    encoder_features[i][:, 0]  # CLS token: [B*T, embed_dim]
                    for i in self.encoder_indices
                ]
                attention_weights_list = [
                    model.pretrained.blocks[block_idx].attn.attn_weights  # [B*T, num_heads, N+1, N+1]
                    for block_idx in self.target_blocks
                ]

                # Average 2-layer CLS tokens and reshape to [B, T, embed_dim]
                cls_tokens_avg = torch.stack(cls_tokens_list, dim=0).mean(dim=0)  # [B*T, embed_dim]
                cls_tokens = rearrange(cls_tokens_avg, '(b t) d -> b t d', b=B_orig, t=T_orig)  # [B, T, embed_dim]

                # Apply Mamba temporal modeling (frozen)
                path_1_temporal = model.dpt_features_to_mamba(
                    input_shape=(B_orig, T_orig, None, H, W),
                    dpt_features=path_1,
                    in_dpt_layer=0
                )  # [B*T, dpt_dim, h, w]

                # Get relative depth from frozen output_conv
                out = model.depth_head.scratch.output_conv1(path_1_temporal)
                out = F.interpolate(out, (H, W), mode="bilinear", align_corners=True)
                relative_depth = model.depth_head.scratch.output_conv2(out)  # [B*T, 1, H, W]

                # Get scale/shift/importance_map from Gear5MetricHead
                gear5_outputs = model.gear5_metric_head(
                    cls_tokens=cls_tokens,  # [B, T, 1024]
                    attention_weights_list=attention_weights_list,  # List of 2 attention weights
                    patch_h=patch_h,
                    patch_w=patch_w
                )

                scale = gear5_outputs['scale']  # [B, T]
                shift = gear5_outputs['shift']  # [B, T]
                importance_map = gear5_outputs['importance_map']  # [B, T, patch_h, patch_w]

                # Apply shift lower bound constraint (Option 1)
                GLOBAL_MIN_INVERSE = 100.0 / 300.0  # 0.333 (300m max depth assumption)
                shift_lower_bound = -scale.abs() * GLOBAL_MIN_INVERSE  # [B, T]
                shift_clamped = torch.clamp(shift, min=shift_lower_bound)  # [B, T]

                # Apply scale/shift to relative depth with clamped shift
                scale_flat = scale.view(B_orig * T_orig, 1, 1, 1)  # [B*T, 1, 1, 1]
                shift_flat = shift_clamped.view(B_orig * T_orig, 1, 1, 1)  # [B*T, 1, 1, 1]
                pred_depth_inverse_100 = scale_flat * relative_depth + shift_flat  # [B*T, 1, H, W] in 100/m

                pred_depth_inverse = pred_depth_inverse_100  # [B*T, 1, H, W] inverse depth (100/m)

                # Reshape GT from (B, T, 1, H, W) to (B*T, 1, H, W)
                gt_depth_inverse_flat = rearrange(gt_depth_inverse_100, 'b t 1 h w -> (b t) 1 h w')

                # DEBUG: Check prediction range at step 0
                if self.global_step == 0 and batch_idx == 0 and self.rank == 0:
                    self.logger.info(f"DEBUG VALIDATION - Pred before interpolate: min={pred_depth_inverse.min():.4f}, max={pred_depth_inverse.max():.4f}, mean={pred_depth_inverse.mean():.4f}")
                    self.logger.info(f"DEBUG VALIDATION - GT inverse_100: min={gt_depth_inverse_flat.min():.4f}, max={gt_depth_inverse_flat.max():.4f}")

                # Interpolate prediction to GT resolution (like original FlashDepth validation)
                gt_shape = gt_depth_inverse_flat.shape[-2:]
                if pred_depth_inverse.shape[-2:] != gt_shape:
                    pred_depth_inverse = F.interpolate(
                        pred_depth_inverse, size=gt_shape, mode="bilinear", align_corners=True
                    )

                # DEBUG: Check after interpolate
                if self.global_step == 0 and batch_idx == 0 and self.rank == 0:
                    self.logger.info(f"DEBUG VALIDATION - Pred after interpolate: min={pred_depth_inverse.min():.4f}, max={pred_depth_inverse.max():.4f}")

                # Get GT resolution for loss computation and visualization
                H_gt, W_gt = gt_depth_inverse_flat.shape[-2:]

                # Compute importance-weighted validation loss
                # Flatten for loss computation
                pred_depth_flat = pred_depth_inverse.flatten()
                gt_depth_flat = gt_depth_inverse_flat.flatten()

                # Valid mask: GT valid + Pred positive (no 70m restriction in validation, like original FlashDepth)
                valid_mask = (gt_depth_flat >= 0) & (pred_depth_flat > 0)

                # Resize importance map to depth map size and flatten
                importance_map_resized = F.interpolate(
                    importance_map.view(B_orig * T_orig, 1, patch_h, patch_w),
                    size=(H_gt, W_gt),
                    mode='bilinear',
                    align_corners=True
                )  # [B*T, 1, H_gt, W_gt]
                importance_flat = importance_map_resized.flatten()

                # Compute fg_ratio (alpha)
                importance_threshold = importance_flat.mean()
                fg_mask_flat = (importance_flat > importance_threshold)
                fg_ratio = fg_mask_flat.float().mean()

                # Save masks for visualization
                canonical_gt_valid = (gt_depth_inverse_flat > 0).cpu()  # [B*T, 1, H_gt, W_gt] - GT valid pixels
                canonical_pred_valid = (pred_depth_inverse > 0).cpu()  # [B*T, 1, H_gt, W_gt] - Pred valid pixels

                if valid_mask.sum() > 0:
                    # DEBUG: Check for negative or zero values before log
                    if self.rank == 0 and (self.global_step % 1000 == 0 or self.global_step < 10):
                        pred_min = pred_depth_flat[valid_mask].min().item()
                        pred_max = pred_depth_flat[valid_mask].max().item()
                        gt_min = gt_depth_flat[valid_mask].min().item()
                        gt_max = gt_depth_flat[valid_mask].max().item()
                        self.logger.info(f"VALIDATION Step {self.global_step} - Pred range: [{pred_min:.4f}, {pred_max:.4f}]")
                        self.logger.info(f"VALIDATION Step {self.global_step} - GT range: [{gt_min:.4f}, {gt_max:.4f}]")

                        # Check for negative values
                        pred_negative = (pred_depth_flat[valid_mask] < 0).sum().item()
                        gt_negative = (gt_depth_flat[valid_mask] < 0).sum().item()
                        if pred_negative > 0 or gt_negative > 0:
                            self.logger.warning(f"VALIDATION - Negative values detected! Pred: {pred_negative}, GT: {gt_negative}")

                    # Importance-weighted Log L1 Loss (in inverse depth space)
                    epsilon = 1e-3
                    # pred_depth_flat and gt_depth_flat are already inverse depth (100/m)
                    # Compute log L1 loss directly in inverse depth space
                    # IMPORTANT: Only compute loss on valid pixels to avoid NaN from negative values
                    pred_valid = pred_depth_flat[valid_mask].float()
                    gt_valid = gt_depth_flat[valid_mask].float()
                    importance_valid = importance_flat[valid_mask].float()

                    # Additional safety: filter out non-positive values
                    positive_mask = (pred_valid > 0) & (gt_valid > 0)
                    if positive_mask.sum() == 0:
                        self.logger.warning(f"VALIDATION Step {self.global_step} - No positive depth values!")
                        frame_losses = []
                    else:
                        pred_positive = pred_valid[positive_mask]
                        gt_positive = gt_valid[positive_mask]
                        importance_positive = importance_valid[positive_mask]

                        loss = torch.abs(
                            torch.log(pred_positive + epsilon) -
                            torch.log(gt_positive + epsilon)
                        )

                        # Compute fg_ratio for this positive subset
                        fg_ratio_positive = (importance_positive > importance_threshold).float().mean()

                        weighted_loss = loss * (1.0 + fg_ratio_positive * importance_positive)
                        loss_batch = weighted_loss.mean()

                        # Check for NaN
                        if torch.isnan(loss_batch):
                            self.logger.error(f"VALIDATION Step {self.global_step} - NaN loss detected!")
                            self.logger.error(f"  pred_positive range: [{pred_positive.min():.4f}, {pred_positive.max():.4f}]")
                            self.logger.error(f"  gt_positive range: [{gt_positive.min():.4f}, {gt_positive.max():.4f}]")
                            frame_losses = []
                        else:
                            frame_losses = [loss_batch.float()]
                else:
                    frame_losses = []

                # Validation visualization: save multiple sequences per dataset (rank 0 only)
                # Reshape predictions back to (B, T, 1, H, W) for visualization
                pred_depth_inverse_seq = rearrange(pred_depth_inverse, '(b t) 1 h w -> b t 1 h w', b=B_orig, t=T_orig)

                # Use first frame for visualization
                if self.val_visualizer and self.rank == 0:
                    # Check if this dataset is in our visualization config
                    if current_dataset in self.val_vis_config:
                        config = self.val_vis_config[current_dataset]
                        seq_num = dataset_sequence_counters[current_dataset]

                        # Check if we should save this sequence
                        should_save = (
                            seq_num in config['sequences'] and
                            seq_num not in config['saved']
                        )

                        if should_save:
                            try:
                                # Use first frame (t=0) for visualization
                                pred_depth_inverse_vis = pred_depth_inverse_seq[:, 0]  # [B, 1, H, W]
                                gt_depth_inverse_vis = gt_depth_inverse_100[:, 0]  # [B, 1, H, W]

                                # Convert to metric depth for visualization: 100/m -> m
                                # Convert to Float32 for CPU operations
                                pred_depth_metric = (100.0 / (pred_depth_inverse_vis.float() + 1e-8)).cpu()
                                gt_depth_metric = (100.0 / (gt_depth_inverse_vis.float() + 1e-8)).cpu()

                                # Get GT resolution for visualization
                                gt_h, gt_w = gt_depth_inverse_vis.shape[-2:]

                                # Resize images to GT resolution (for visualization consistency)
                                img_vis = images[:, 0]  # [B, C, H, W] - first frame
                                img_t_resized = F.interpolate(
                                    img_vis, size=(gt_h, gt_w), mode='bilinear', align_corners=True
                                )

                                # Reshape canonical masks to (B, T, 1, H, W) and select first frame
                                canonical_gt_valid_seq = rearrange(canonical_gt_valid, '(b t) 1 h w -> b t 1 h w', b=B_orig, t=T_orig)
                                canonical_pred_valid_seq = rearrange(canonical_pred_valid, '(b t) 1 h w -> b t 1 h w', b=B_orig, t=T_orig)
                                canonical_gt_valid_vis = canonical_gt_valid_seq[:, 0]  # [B, 1, H, W]
                                canonical_pred_valid_vis = canonical_pred_valid_seq[:, 0]

                                # Unified: Include importance map
                                # importance_map from gear5_outputs is [B, T, patch_h, patch_w]
                                # Select first frame: [B, patch_h, patch_w]
                                importance_map_vis = importance_map[:, 0]  # [B, patch_h, patch_w]

                                # Resize importance map to GT resolution
                                importance_map_resized_vis = F.interpolate(
                                    importance_map_vis.unsqueeze(1), size=(gt_h, gt_w), mode='bilinear', align_corners=True
                                )  # [B, 1, gt_h, gt_w]

                                # Unified: Include all outputs + scale/shift
                                model_outputs = {
                                    'pred_depth': pred_depth_metric,  # [B, 1, gt_h, gt_w]
                                    'importance_map': importance_map_resized_vis.float().cpu(),  # [B, 1, gt_h, gt_w]
                                    'canonical_gt_valid': canonical_gt_valid_vis,  # [B, 1, H, W]
                                    'canonical_pred_valid': canonical_pred_valid_vis,  # [B, 1, H, W]
                                    'scale': scale[:, 0:1].cpu(),  # [B, 1] - first frame
                                    'shift': shift[:, 0:1].cpu()   # [B, 1] - first frame
                                }

                                # For visualization, we need [B, T, ...] format like training
                                # But we only have one frame (t=0), so unsqueeze T dimension
                                sample_batch = (
                                    img_t_resized.unsqueeze(1).float().cpu(),  # [B, 1, C, gt_h, gt_w] at GT resolution
                                    gt_depth_metric.unsqueeze(1),  # [B, 1, gt_h, gt_w]
                                    dataset_idx,
                                    focal_lengths_actual[:, 0:1].float().cpu()  # [B, 1] - original focal length for first frame
                                )

                                # FPS removed from training (only measured in test_gear3.py)
                                current_fps = None

                                # Create loss_dict with current frame loss (for visualization)
                                val_loss_dict = {
                                    'val_loss': loss_batch.item() if len(frame_losses) > 0 else 0.0
                                }

                                # No layer_weights in unified Gear5 (GRU-based, not multi-layer fusion)
                                # Save with dataset and sequence-specific name
                                save_name = f"validation_{current_dataset}_seq{seq_num:03d}_step_{self.global_step:06d}"
                                self.val_visualizer.create_validation_summary(
                                    sample_batch, model_outputs, self.global_step,
                                    save_name=save_name, fps=current_fps, loss_dict=val_loss_dict, dataset_name=current_dataset, config=self.config
                                )
                                config['saved'].append(seq_num)
                                self.logger.info(f"Saved validation visualization: {current_dataset} sequence {seq_num} ({len(config['saved'])}/{len(config['sequences'])})")
                            except Exception as e:
                                if self.rank == 0:
                                    self.logger.warning(f"Failed to save validation visualization for {current_dataset} seq {seq_num}: {e}")

                    # Clear intermediate tensors to free memory after each sequence
                    del encoder_features, attention_weights, patch_tokens, dpt_features, path_1, path_1_temporal
                    del relative_depth, pred_depth_inverse, importance_map
                    # Always clear cache after each sequence to prevent OOM during validation
                    torch.cuda.empty_cache()                # Track sequence count (regardless of valid pixels)
                if current_dataset not in dataset_sequence_counts:
                    dataset_sequence_counts[current_dataset] = 0
                dataset_sequence_counts[current_dataset] += 1

                # Track loss only if valid pixels exist
                if len(frame_losses) > 0:
                    avg_loss = sum(frame_losses) / len(frame_losses)
                    total_loss += avg_loss.item()
                    num_batches += 1

                    # Track per-dataset loss
                    if current_dataset not in dataset_losses:
                        dataset_losses[current_dataset] = []
                    dataset_losses[current_dataset].append(avg_loss.item())

            # Increment sequence counter for this dataset
            if current_dataset in dataset_sequence_counters:
                dataset_sequence_counters[current_dataset] += 1

            # Increment total processed count (for batch limit)
            total_processed += 1

            # Clear batch memory and GPU cache to prevent OOM
            del images, gt_depth, gt_depth_inverse_100
            torch.cuda.empty_cache()

        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')

        # Log detailed per-dataset statistics (rank 0 only)
        if self.rank == 0:
            self.logger.info("=" * 80)
            self.logger.info(f"VALIDATION SUMMARY (Step {self.global_step})")
            self.logger.info("=" * 80)
            self.logger.info(f"Total batches with valid pixels: {num_batches}")

            # Show all datasets (including those with no valid pixels)
            for dataset_name in dataset_sequence_counts.keys():
                total_seqs = dataset_sequence_counts[dataset_name]
                if dataset_name in dataset_losses:
                    losses = dataset_losses[dataset_name]
                    dataset_avg = np.mean(losses)
                    dataset_std = np.std(losses)
                    valid_seqs = len(losses)
                    self.logger.info(
                        f"  [{dataset_name}] {valid_seqs}/{total_seqs} sequences with valid pixels | "
                        f"Loss: {dataset_avg:.4f} ± {dataset_std:.4f}"
                    )
                else:
                    self.logger.info(
                        f"  [{dataset_name}] 0/{total_seqs} sequences with valid pixels | "
                        f"Loss: N/A (no valid pixels)"
                    )

            self.logger.info(f"Overall Average Loss: {avg_loss:.4f} (from {num_batches} sequences)")

            # WARNING: Check if validation set is too small (Phase 2)
            if self.phase >= 2 and num_batches < 20:
                self.logger.warning(
                    f"⚠️  SMALL VALIDATION SET: Only {num_batches} batches evaluated! "
                    f"Consider increasing max_val_batches for more reliable validation."
                )

            self.logger.info("=" * 80)

        # Clear cache after validation
        torch.cuda.empty_cache()

        # Back to training mode
        self.model.train()

        return {
            'loss': avg_loss,
            'dataset_losses': {k: float(np.mean(v)) for k, v in dataset_losses.items()},
            'num_sequences': {k: v for k, v in dataset_sequence_counts.items()}
        }

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
            'dataset_losses': self.dataset_losses,  # Save per-dataset validation losses
            'num_sequences': self.num_sequences,  # Save number of sequences per dataset
            'config': OmegaConf.to_container(self.config, resolve=True),
            'phase': self.phase
        }

        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")


@hydra.main(version_base=None, config_path="configs/gear5", config_name="config")
def main(config: DictConfig):
    """Main entry point"""
    # Initialize distributed training
    rank, world_size, local_rank = init_distributed()

    # Create trainer
    trainer = Gear5Trainer(config, rank, world_size, local_rank)
    trainer.train()

    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
