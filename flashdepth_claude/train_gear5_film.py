"""
Gear5 FiLM Training Script: Temporal FiLM-style Feature Modulation

Single-stage training approach:
    - Input: 2-layer CLS tokens [11, 23] for ViT-L or [5, 11] for ViT-S
    - Processing: FiLM-style modulation of DPT features before Mamba
    - Modulation: Channel-wise gamma and beta parameters from CLS tokens
    - Formula: modulated_feature = gamma ⊙ feature + beta (per channel)
    - Output: Inverse depth from modulated features after Mamba + output_conv1/2

Freezing Strategy:
    - Frozen: ViT encoder, DPT decoder
    - Trainable: Gear5FilmHead, Mamba modules, output_conv1/2

Loss:
    - Configurable loss type: 'log_l1' (default) or 'importance' (weighted)
    - Applied directly on inverse depth predictions
    - Supports importance map weighting for loss (FiLM generates importance map)

Key features:
    - FiLM-style modulation instead of GRU-based scale/shift
    - Modulates intermediate features, not final depth output
    - Temporal consistency via Mamba (after modulation)
    - Canonical space normalization (focal_length=500, configurable)
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
from utils.gear5_film_visualization import Gear5FilmVisualizer
from flashdepth.gear5_film_modules import Gear5FilmHead
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




class Gear5FilmTrainer:
    """
    Trainer for Gear5 FiLM - temporal FiLM-style feature modulation.

    Trainable parts:
        - Gear5FilmHead: FiLM modulation network (~262K parameters)
        - Mamba: Temporal processing modules
        - output_conv1/2: Final depth prediction layers (both trainable)

    Frozen parts:
        - ViT encoder (DINOv2)
        - DPT decoder
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
                logging.info(f"  Phase 2 (Hybrid): 2K resolution, Gear5-FiLM weights + FlashDepth-hybrid")

        # Setup results directory (only rank 0)
        phase_suffix = f"_phase{self.phase}" if self.phase > 1 else ""
        self.results_dir = Path(config.get('results_dir', f'./train_results/gear5_film{phase_suffix}'))
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

        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()

        # Setup loss function (Gear5 style: 'log_l1' or 'importance')
        from utils.gear_losses import LogL1Loss
        self.loss_fn = LogL1Loss()
        self.loss_type = config.get('loss_type', 'log_l1')
        if rank == 0:
            self.logger.info(f"Loss type: {self.loss_type}")

        # Setup wandb
        if config.training.get('wandb', False):
            wandb.init(
                project="flashdepth-gear5-film",
                name=f"gear5_film_phase{self.phase}_{config.training.get('wandb_name', 'experiment')}",
                config=dict(config)
            )

        # Setup visualizer with separate folders
        self.train_visualizer = Gear5FilmVisualizer(save_dir=self.results_dir / "visualizations" / "train")
        self.val_visualizer = Gear5FilmVisualizer(save_dir=self.results_dir / "visualizations" / "valid")

        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_step = 0  # Track which step achieved best validation loss
        self.current_val_loss = None  # Track current validation loss for checkpoint
        self.dataset_losses = None  # Track per-dataset validation losses
        self.num_sequences = None  # Track number of sequences per dataset

        # Validation visualization config: track which sequences to visualize
        # Phase 2: Save 1 sample per dataset at every validation step (for step-by-step comparison)
        # Sintel: 1022×434, Waymo_seg: 1918×1274 (2K resolution)
        if self.phase >= 2:
            self.val_vis_config = {
                'sintel': {'sequences': [0], 'saved': []},  # seq 0 only
                'waymo_seg': {'sequences': [0], 'saved': []}    # seq 0 only
            }
        else:
            # Phase 1: Can afford more samples (sintel: 1022×434, waymo_seg: 784×518)
            self.val_vis_config = {
                'sintel': {'sequences': [0, 4, 7], 'saved': []},  # seq 0, 4, 7 (3 samples)
                'waymo_seg': {'sequences': [0, 1, 2, 3, 4, 5, 6, 7], 'saved': []}  # all 8 sequences
            }

    def _get_canonical_focal_length(self):
        """
        Get canonical focal length from config.

        Returns:
            float: Canonical focal length (default 500.0 for 518×518 resolution)
        """
        return self.config.get('canonical_focal_length', 500.0)

    def _setup_model(self):
        """Initialize FlashDepth with Gear5 FiLM head"""
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
        # Phase 2: Load Phase 1 Gear5-FiLM checkpoint, then overwrite ViT+DPT with FlashDepth-S hybrid
        if self.phase == 1:
            checkpoint_path = self.config.get('load')
        else:
            # Phase 2: Load Gear5-FiLM Phase 1 checkpoint
            checkpoint_path = self.config.get('gear_checkpoint')
            if not checkpoint_path:
                raise ValueError(
                    "Phase 2 (Hybrid) requires 'gear_checkpoint' in config! "
                    "Set gear_checkpoint to your Gear5-FiLM Phase 1 checkpoint path."
                )
            if self.rank == 0:
                self.logger.info(f"Phase {self.phase}: Will load Gear5-FiLM Phase 1 checkpoint from {checkpoint_path}")

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

                # Load ALL parameters from FlashDepth (exclude only gear5_film_head)
                loaded_dict = {}
                excluded_keys = []
                for k, v in state_dict.items():
                    if 'gear5_film_head' in k:
                        excluded_keys.append(k)
                    else:
                        loaded_dict[k] = v

                # Load state dict (strict=False to allow missing gear5_film_head)
                model.load_state_dict(loaded_dict, strict=False)
                self.logger.info(f"Phase 1: Loaded {len(loaded_dict)} parameters from FlashDepth checkpoint")
                self.logger.info(f"  - DINOv2 encoder: ✓ (will be frozen)")
                self.logger.info(f"  - DPT decoder: ✓ (will be frozen)")
                self.logger.info(f"  - Mamba modules: ✓ (will be trainable)")
                self.logger.info(f"  - output_conv1/2: ✓ (both trainable)")
                if excluded_keys:
                    self.logger.info(f"Excluded {len(excluded_keys)} parameters (gear5_film_head will be created)")
            else:
                self.logger.warning(f"Checkpoint {checkpoint_path} not found")

        # Add Gear5 FiLM head (temporal FiLM-style modulation)
        # Uses 2-layer CLS tokens for generating gamma and beta parameters
        model.gear5_film_head = Gear5FilmHead(
            embed_dim=embed_dim,
            dpt_dim=dpt_dim
        )

        # Phase 2: Load Phase 1 checkpoint, then overwrite ViT+DPT with FlashDepth-S hybrid
        if self.phase == 2 and checkpoint_path and checkpoint_path != 'true' and os.path.exists(checkpoint_path):
            # Step 1: Load Phase 1 Gear5-FiLM checkpoint (all components including gear5_film_head)
            self.logger.info(f"Phase 2: Loading Gear5-FiLM Phase 1 checkpoint: {checkpoint_path}")
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
            self.logger.info(f"Phase 2: Loaded Gear5-FiLM Phase 1 checkpoint")
            self.logger.info(f"  - ViT (ViT-L): ✓ (will be overwritten with hybrid)")
            self.logger.info(f"  - DPT: ✓ (will be overwritten with hybrid)")
            self.logger.info(f"  - Mamba: ✓ (kept from Phase 1, trainable)")
            self.logger.info(f"  - output_conv: ✓ (kept from Phase 1, both trainable)")
            self.logger.info(f"  - Gear5FilmHead: ✓ (kept from Phase 1, trainable)")

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

                # Load ONLY ViT and DPT parameters (exclude Mamba, output_conv, gear5_film_head)
                loaded_hybrid = {}
                for k, v in hybrid_state_dict.items():
                    # Include only encoder and DPT (exclude Mamba, output_conv, gear5_film_head)
                    if not any(x in k for x in ['mamba', 'hybrid_fusion', 'teacher_model',
                                                 'output_conv1', 'output_conv2', 'gear5_film_head']):
                        loaded_hybrid[k] = v

                # Overwrite ViT + DPT parameters with hybrid weights
                model.load_state_dict(loaded_hybrid, strict=False)
                self.logger.info(f"Phase 2: Overwritten {len(loaded_hybrid)} ViT + DPT parameters with hybrid weights")
                self.logger.info(f"  - ViT (ViT-L + ViT-S + Cross Attn): ✓")
                self.logger.info(f"  - DPT: ✓")
                self.logger.info(f"  - Kept from Phase 1: Mamba (trainable), output_conv1/2 (trainable), Gear5FilmHead (trainable)")
            else:
                self.logger.warning(f"FlashDepth-S hybrid checkpoint {hybrid_path} not found! Using Phase 1 ViT + DPT.")

        # Enable attention weights storage (for CLS token extraction)
        # Use 2 layers for CLS token extraction
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

        self.logger.info(f"2-layer CLS token extraction: blocks {target_blocks}")

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
        Gear5 FiLM freezing strategy:
            Frozen: ViT encoder, DPT decoder
            Trainable: Gear5FilmHead, Mamba modules, output_conv1/2
        """
        frozen_vit_dpt = 0
        trainable_mamba = 0
        trainable_output_conv = 0
        trainable_film = 0

        for name, param in model.named_parameters():
            # Gear5 FiLM head: trainable
            if 'gear5_film_head' in name:
                param.requires_grad = True
                trainable_film += param.numel()
                self.logger.info(f"Trainable (FiLM): {name} - {param.shape}")

            # Mamba: trainable
            elif 'mamba' in name:
                param.requires_grad = True
                trainable_mamba += param.numel()

            # output_conv (both 1 and 2): trainable
            elif 'output_conv' in name:
                param.requires_grad = True
                trainable_output_conv += param.numel()
                self.logger.info(f"Trainable (output_conv): {name} - {param.shape}")

            # Everything else (ViT encoder, DPT decoder): frozen
            else:
                param.requires_grad = False
                frozen_vit_dpt += param.numel()

        # Log summary
        self.logger.info(f"=== Parameter Configuration (Phase {self.phase}) ===")
        self.logger.info(f"Frozen:")
        self.logger.info(f"  - ViT + DPT: {frozen_vit_dpt:,}")

        self.logger.info(f"Trainable:")
        self.logger.info(f"  - Gear5FilmHead: {trainable_film:,}")
        self.logger.info(f"  - Mamba: {trainable_mamba:,}")
        self.logger.info(f"  - output_conv1/2: {trainable_output_conv:,}")

        total_frozen = frozen_vit_dpt
        total_trainable = trainable_film + trainable_mamba + trainable_output_conv
        self.logger.info(f"Total frozen: {total_frozen:,}")
        self.logger.info(f"Total trainable: {total_trainable:,}")

    def _set_train_mode(self):
        """
        Set model to training mode, but keep frozen parts in eval mode.
        This prevents BatchNorm/Dropout in frozen parts from updating.
        """
        self.model.train()

        # Keep frozen parts in eval mode
        for name, module in self.model.named_modules():
            # Skip empty name (root module)
            if name == '':
                continue

            # Keep trainable parts in train mode
            if any(keyword in name for keyword in ['gear5_film_head', 'mamba', 'output_conv']):
                continue

            # Set frozen parts to eval mode
            module.eval()

    def _setup_data_loaders(self):
        """Setup phase-specific data loaders"""
        if self.phase == 1:
            # Phase 1: 5 datasets, base resolution (aspect ratio preserved)
            # Train: 518×518 (center crop), Val: dataset-specific (sintel: 1022×434, waymo_seg: 784×518)
            train_datasets = self.config.dataset.get('train_datasets',
                ['mvs-synth', 'dynamicreplica', 'tartanair', 'pointodyssey', 'spring'])
            val_datasets = self.config.dataset.get('val_datasets', ['sintel', 'waymo'])
            resolution = 'base'
        else:
            # Phase 2: mvs-synth, spring only, 2K resolution (Hybrid)
            # Val: sintel (1022×434), waymo_seg (1918×1274)
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
        """Setup optimizer with different learning rates for different components"""
        # Three learning rates from config
        film_lr = self.config.training.get('film_lr', 1e-4)
        mamba_lr = self.config.training.get('mamba_lr', 1e-5)
        output_lr = self.config.training.get('output_lr', 1e-5)

        # Create param groups
        film_params = []
        mamba_params = []
        output_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'gear5_film_head' in name:
                    film_params.append(param)
                elif 'mamba' in name:
                    mamba_params.append(param)
                elif 'output_conv' in name:  # Include both output_conv1 and output_conv2
                    output_params.append(param)

        param_groups = [
            {'params': film_params, 'lr': film_lr, 'name': 'film'},
            {'params': mamba_params, 'lr': mamba_lr, 'name': 'mamba'},
            {'params': output_params, 'lr': output_lr, 'name': 'output'}
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            betas=[0.9, 0.95],  # Same as original FlashDepth
            weight_decay=self.config.training.get('weight_decay', 1e-6)
        )

        self.logger.info(f"Optimizer setup:")
        self.logger.info(f"  FiLM: {len(film_params)} params, LR={film_lr}")
        self.logger.info(f"  Mamba: {len(mamba_params)} params, LR={mamba_lr}")
        self.logger.info(f"  Output: {len(output_params)} params, LR={output_lr}")

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

            # Skip if no valid pixels in batch
            if loss_dict is None:
                continue

            # Get learning rates
            lr_film = self.optimizer.param_groups[0]['lr']
            lr_mamba = self.optimizer.param_groups[1]['lr']
            lr_output = self.optimizer.param_groups[2]['lr']

            # Update progress bar (every step)
            postfix_dict = {
                'loss': f'{loss_dict["loss"]:.4f}',
                'lr_f': f'{lr_film:.2e}',
                'lr_m': f'{lr_mamba:.2e}',
                'lr_o': f'{lr_output:.2e}'
            }

            pbar.set_postfix(postfix_dict)

            # WandB logging (every step)
            wandb_dict = {**loss_dict, 'lr_film': lr_film, 'lr_mamba': lr_mamba, 'lr_output': lr_output}
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

                    # Unpack and move to device
                    images, gt_depth, focal_lengths_canonical, focal_lengths_actual, actual_valid_masks, fx_ratio, resize_ratio, dataset_idx = batch
                    images = images.to(self.device)
                    gt_depth = gt_depth.to(self.device)
                    focal_lengths_canonical = focal_lengths_canonical.to(self.device)
                    focal_lengths_actual = focal_lengths_actual.to(self.device)
                    actual_valid_masks = actual_valid_masks.to(self.device)
                    fx_ratio = fx_ratio.to(self.device)
                    resize_ratio = resize_ratio.to(self.device)

                    if gt_depth.ndim == 3:
                        gt_depth = gt_depth.unsqueeze(1)
                    elif gt_depth.ndim == 4 and gt_depth.shape[1] != 1:
                        gt_depth = gt_depth.unsqueeze(2)

                    # GT depth from dataloader is inverse depth (1/m), scale to 100/m for training
                    gt_depth_inverse_100 = gt_depth * 100.0

                    # Get batch size
                    B_orig, T_orig, C, H, W = images.shape

                    # Initialize Mamba sequence
                    if hasattr(model, 'mamba'):
                        model.mamba.start_new_sequence()

                    # Process entire sequence at once
                    images_flat = rearrange(images, 'b t c h w -> (b t) c h w')
                    patch_h, patch_w = H // model.patch_size, W // model.patch_size

                    with torch.no_grad():
                        # Extract features from all frames at once
                        encoder_features = model.pretrained.get_intermediate_layers(
                            images_flat, model.intermediate_layer_idx[model.encoder]
                        )

                        # Extract 2-layer CLS tokens
                        cls_tokens_list = [
                            encoder_features[i][:, 0]  # CLS token: [B*T, embed_dim]
                            for i in self.encoder_indices
                        ]
                        # Reshape to [B, T, embed_dim] for each layer
                        cls_tokens_multi_layer = [
                            rearrange(cls_tokens, '(b t) d -> b t d', b=B_orig, t=T_orig)
                            for cls_tokens in cls_tokens_list
                        ]

                        # Get DPT features (frozen)
                        dpt_features = model.depth_head.get_forward_features(
                            encoder_features, patch_h, patch_w
                        )

                        # Extract attention weights for importance map
                        attention_weights_list = [
                            model.pretrained.blocks[block_idx].attn.attn_weights  # [B*T, num_heads, N+1, N+1]
                            for block_idx in self.target_blocks
                        ]

                        # Apply FiLM modulation (no grad needed for visualization)
                        film_outputs = model.gear5_film_head(
                            cls_tokens_multi_layer,  # List of [B, T, embed_dim]
                            attention_weights_list,  # List of 2 attention weights for importance map
                            dpt_features,  # List of 4 DPT features [B*T, dpt_dim, h, w]
                            patch_h, patch_w
                        )
                        path_1_modulated = film_outputs['path_1_modulated']  # [B*T, dpt_dim, h, w]
                        gamma = film_outputs['gamma']  # [B, T, dpt_dim]
                        beta = film_outputs['beta']  # [B, T, dpt_dim]
                        importance_map = film_outputs['importance_map']  # [B, T, patch_h, patch_w]

                        # Apply Mamba to modulated features (no grad needed for visualization)
                        path_1_temporal = model.dpt_features_to_mamba(
                            input_shape=(B_orig, T_orig, None, H, W),
                            dpt_features=path_1_modulated,  # [B*T, dpt_dim, h, w]
                            in_dpt_layer=0
                        )

                        # Final depth prediction (no grad needed for visualization)
                        out = model.depth_head.scratch.output_conv1(path_1_temporal)
                        out = F.interpolate(out, (H, W), mode="bilinear", align_corners=True)
                        pred_depth_inverse = model.depth_head.scratch.output_conv2(out)  # [B*T, 1, H, W]

                        # Convert to metric depth: 100/m -> m
                        pred_depth_metric = 100.0 / (pred_depth_inverse + 1e-8)

                        # Reshape for visualization
                        pred_depth_inverse_seq = rearrange(pred_depth_inverse, '(b t) 1 h w -> b t 1 h w', b=B_orig, t=T_orig)
                        pred_depth_metric_seq = rearrange(pred_depth_metric, '(b t) 1 h w -> b t 1 h w', b=B_orig, t=T_orig)
                        pred_depth_inverse_vis = pred_depth_inverse_seq[:, 0]  # [B, 1, H, W]
                        pred_depth_metric_vis = pred_depth_metric_seq[:, 0]  # [B, 1, H, W]

                        # GT depth for first frame
                        gt_depth_inverse_vis = gt_depth_inverse_100[:, 0]  # [B, 1, H, W]
                        gt_depth_metric = 100.0 / (gt_depth_inverse_vis + 1e-8)

                        # Compute canonical masks
                        MIN_INVERSE_DEPTH_VIS = 100.0 / 70.0
                        canonical_gt_valid_vis = (gt_depth_inverse_vis > MIN_INVERSE_DEPTH_VIS)

                        MAX_DEPTH_OUTLIER_VIS = 200.0
                        MIN_INVERSE_OUTLIER_VIS = 100.0 / MAX_DEPTH_OUTLIER_VIS
                        canonical_pred_valid_vis = (pred_depth_inverse_vis > MIN_INVERSE_OUTLIER_VIS)

                    # Prepare visualization batch
                    sample_batch = (
                        images[:1, :1].float().cpu(),
                        gt_depth_metric[:1].float().cpu(),
                        dataset_idx,
                        fx_ratio[:1, :1].float().cpu(),
                        resize_ratio[:1, :1].float().cpu()
                    )

                    # Resize importance map to match image resolution (for smooth visualization)
                    importance_map_vis = importance_map[:1, :1]  # [1, 1, patch_h, patch_w]
                    importance_map_resized = F.interpolate(
                        importance_map_vis, size=(H, W), mode='bilinear', align_corners=True
                    )  # [1, 1, H, W]

                    model_outputs_cpu = {
                        'pred_depth': pred_depth_metric_vis[:1].cpu(),
                        'canonical_gt_valid': canonical_gt_valid_vis[:1].cpu(),
                        'canonical_pred_valid': canonical_pred_valid_vis[:1].cpu(),
                        'importance_map': importance_map_resized.cpu(),  # [1, 1, H, W] - upsampled
                        'gamma': gamma[:1, :1].cpu(),  # [1, 1, dpt_dim]
                        'beta': beta[:1, :1].cpu()     # [1, 1, dpt_dim]
                    }

                    self.train_visualizer.create_validation_summary(
                        sample_batch, model_outputs_cpu, step, prefix="training", fps=None, loss_dict=loss_dict, config=self.config
                    )

                    self._set_train_mode()

                except Exception as e:
                    import traceback
                    self.logger.error(f"Failed to save training visualization: {e}")
                    self.logger.error(f"Traceback:\n{traceback.format_exc()}")
                    self._set_train_mode()

            # Validation
            if step % self.config.training.get('val_freq', 1000) == 0:
                if self.rank == 0:
                    val_metrics = self.validate()
                    self.logger.info(f"Validation at step {step}: {val_metrics}")

                    # Update current validation loss for checkpoint
                    self.current_val_loss = val_metrics['loss']
                    self.dataset_losses = val_metrics.get('dataset_losses', None)
                    self.num_sequences = val_metrics.get('num_sequences', None)

                    if self.config.training.get('wandb', False):
                        # Log overall validation metrics
                        wandb_val_dict = {'val/loss': val_metrics['loss']}

                        # Log per-dataset validation losses
                        if 'dataset_losses' in val_metrics:
                            for dataset, loss in val_metrics['dataset_losses'].items():
                                wandb_val_dict[f'val/{dataset}_loss'] = loss

                        # Log number of sequences per dataset
                        if 'num_sequences' in val_metrics:
                            for dataset, count in val_metrics['num_sequences'].items():
                                wandb_val_dict[f'val/{dataset}_sequences'] = count

                        wandb.log(wandb_val_dict, step=step)

                    # Save best model
                    if val_metrics['loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['loss']
                        self.best_step = step
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
        """Single training step with FiLM-style modulation"""
        # Unpack batch
        images, gt_depth, focal_lengths_canonical, focal_lengths_actual, actual_valid_mask, fx_ratio, resize_ratio, dataset_idx = batch
        images = images.to(self.device)
        gt_depth = gt_depth.to(self.device)
        focal_lengths_canonical = focal_lengths_canonical.to(self.device)
        focal_lengths_actual = focal_lengths_actual.to(self.device)
        actual_valid_mask = actual_valid_mask.to(self.device)
        fx_ratio = fx_ratio.to(self.device)
        resize_ratio = resize_ratio.to(self.device)

        # Add channel dimension if needed
        if gt_depth.ndim == 3:
            gt_depth = gt_depth.unsqueeze(1)
        elif gt_depth.ndim == 4 and gt_depth.shape[1] != 1:
            gt_depth = gt_depth.unsqueeze(2)

        B, T = images.shape[:2]

        # Get the actual model (unwrap DDP if needed)
        model = self.model.module if isinstance(self.model, DDP) else self.model

        # GT depth from dataloader is inverse depth (1/m), scale to 100/m
        gt_depth_inverse_100 = gt_depth * 100.0

        # Forward pass
        # Initialize Mamba sequence
        if hasattr(model, 'mamba'):
            model.mamba.start_new_sequence()

        # Reshape video from (B, T, C, H, W) to (B*T, C, H, W)
        B_orig, T_orig, C, H, W = images.shape
        images_flat = rearrange(images, 'b t c h w -> (b t) c h w')

        patch_h, patch_w = H // model.patch_size, W // model.patch_size

        # Use BFloat16 for forward pass
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # Extract features from DINOv2 (frozen, no grad)
            with torch.no_grad():
                encoder_features = model.pretrained.get_intermediate_layers(
                    images_flat, model.intermediate_layer_idx[model.encoder]
                )

                # Extract 2-layer CLS tokens
                cls_tokens_list = [
                    encoder_features[i][:, 0]  # CLS token: [B*T, embed_dim]
                    for i in self.encoder_indices
                ]
                # Reshape to [B, T, embed_dim] for each layer
                cls_tokens_multi_layer = [
                    rearrange(cls_tokens, '(b t) d -> b t d', b=B_orig, t=T_orig)
                    for cls_tokens in cls_tokens_list
                ]

                # Get DPT features (frozen)
                dpt_features = model.depth_head.get_forward_features(
                    encoder_features, patch_h, patch_w
                )  # List of 4 features [B*T, dpt_dim, h, w]

                # Extract attention weights for importance map
                attention_weights_list = [
                    model.pretrained.blocks[block_idx].attn.attn_weights  # [B*T, num_heads, N+1, N+1]
                    for block_idx in self.target_blocks
                ]

            # Apply FiLM modulation (trainable)
            film_outputs = model.gear5_film_head(
                cls_tokens_multi_layer,  # List of [B, T, embed_dim]
                attention_weights_list,  # List of 2 attention weights for importance map
                dpt_features,  # List of 4 DPT features [B*T, dpt_dim, h, w]
                patch_h, patch_w
            )
            path_1_modulated = film_outputs['path_1_modulated']  # [B*T, dpt_dim, h, w]
            gamma = film_outputs['gamma']  # [B, T, dpt_dim]
            beta = film_outputs['beta']  # [B, T, dpt_dim]
            importance_map = film_outputs['importance_map']  # [B, T, patch_h, patch_w]

            # Apply Mamba to modulated features (trainable)
            path_1_temporal = model.dpt_features_to_mamba(
                input_shape=(B_orig, T_orig, None, H, W),
                dpt_features=path_1_modulated,  # [B*T, dpt_dim, h, w]
                in_dpt_layer=0
            )

            # Final depth prediction (both output_conv1/2 trainable)
            out = model.depth_head.scratch.output_conv1(path_1_temporal)  # Trainable
            out = F.interpolate(out, (H, W), mode="bilinear", align_corners=True)
            pred_depth_inverse_100 = model.depth_head.scratch.output_conv2(out)  # Trainable [B*T, 1, H, W]

        # Reshape GT from (B, T, 1, H, W) to (B*T, 1, H, W)
        gt_depth_inverse_flat = rearrange(gt_depth_inverse_100, 'b t 1 h w -> (b t) 1 h w')

        # Flatten for loss computation
        pred_depth_flat = pred_depth_inverse_100.flatten()
        gt_depth_flat = gt_depth_inverse_flat.flatten()

        # Valid mask: GT valid (>0) + Pred positive (no threshold, like Gear2)
        valid_mask = (gt_depth_flat > 0) & (pred_depth_flat > 0)

        if valid_mask.sum() == 0:
            self.logger.warning("No valid pixels in batch! Skipping this step.")
            return None

        # Compute loss based on loss_type (Gear5 style)
        with torch.amp.autocast('cuda', enabled=False):
            epsilon = 1e-3

            # Clamp to positive values BEFORE log to prevent NaN from log(negative) or log(0)
            pred_depth_flat = torch.clamp(pred_depth_flat.float(), min=epsilon)
            gt_depth_flat = torch.clamp(gt_depth_flat.float(), min=epsilon)

            # pred_depth_flat and gt_depth_flat are already inverse depth (100/m)
            # Compute log L1 loss directly in inverse depth space
            loss = torch.abs(
                torch.log(pred_depth_flat + epsilon) -
                torch.log(gt_depth_flat + epsilon)
            )

            if self.loss_type == 'importance':
                # Importance-weighted Log L1 Loss
                # Resize importance_map to image resolution (bilinear for smooth weighting)
                importance_map_resized = F.interpolate(
                    importance_map.view(B_orig * T_orig, 1, patch_h, patch_w),
                    size=(H, W),
                    mode='bilinear',
                    align_corners=True
                )  # [B*T, 1, H, W]
                importance_flat = importance_map_resized.flatten()  # [B*T*H*W]

                # Compute fg_ratio from upsampled importance map (pixel-level)
                importance_threshold = importance_flat.mean()
                fg_mask = (importance_flat > importance_threshold)
                fg_ratio = fg_mask.float().mean().detach()  # Detach to avoid gradient issues

                # Apply importance weighting
                weighted_loss = loss * (1.0 + fg_ratio * importance_flat.float())
                final_loss = weighted_loss[valid_mask].mean()
            else:
                # Regular Log L1 Loss (no importance weighting)
                final_loss = loss[valid_mask].mean()

        # Backward pass (outside autocast for numerical stability)
        self.optimizer.zero_grad()
        final_loss.backward()

        # Compute gradient norm BEFORE clipping
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        return {
            'loss': final_loss.item(),
            'grad_norm': total_norm
        }

    @torch.no_grad()
    def validate(self):
        """Validation loop with visualization"""
        # Clear cache before validation
        torch.cuda.empty_cache()

        self.model.eval()

        # Unwrap DDP if needed
        model = self.model.module if isinstance(self.model, DDP) else self.model

        total_loss = 0
        num_batches = 0

        # Track per-dataset losses
        dataset_losses = {}
        dataset_sequence_counts = {}

        # Reset visualization tracking
        for dataset_name in self.val_vis_config:
            self.val_vis_config[dataset_name]['saved'] = []

        # Track dataset-specific sequence counters
        dataset_sequence_counters = {'sintel': 0, 'waymo_seg': 0}

        # Phase 2: Limit validation batches
        if self.phase >= 2:
            max_val_batches = 6
            dataset_max_sequences = {'sintel': 4, 'waymo_seg': 2}
        else:
            max_val_batches = None
            dataset_max_sequences = {}

        total_processed = 0

        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation", disable=(self.rank != 0))):
            # Skip None batches
            if batch is None:
                continue

            # Unpack batch
            images, gt_depth, focal_lengths_canonical, focal_lengths_actual, actual_valid_mask, fx_ratio, resize_ratio, dataset_idx = batch

            # Get dataset name
            if isinstance(dataset_idx, str):
                current_dataset = dataset_idx
            elif isinstance(dataset_idx, (list, tuple)):
                current_dataset = str(dataset_idx[0])
            elif torch.is_tensor(dataset_idx):
                current_dataset = str(dataset_idx[0].item() if dataset_idx.dim() > 0 else dataset_idx.item())
            else:
                current_dataset = str(dataset_idx)

            # Check dataset-specific limit
            if dataset_max_sequences and current_dataset in dataset_max_sequences:
                if dataset_sequence_counters.get(current_dataset, 0) >= dataset_max_sequences[current_dataset]:
                    if self.rank == 0 and dataset_sequence_counters.get(current_dataset, 0) == dataset_max_sequences[current_dataset]:
                        self.logger.info(f"  [{current_dataset}] Reached max {dataset_max_sequences[current_dataset]} sequences, skipping further...")
                    continue

            # Check total batch limit
            if max_val_batches is not None and total_processed >= max_val_batches:
                break

            # Use BFloat16 autocast with no_grad for validation
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                images = images.to(self.device)
                gt_depth = gt_depth.to(self.device)
                focal_lengths_canonical = focal_lengths_canonical.to(self.device)
                focal_lengths_actual = focal_lengths_actual.to(self.device)
                actual_valid_mask = actual_valid_mask.to(self.device)
                fx_ratio = fx_ratio.to(self.device)
                resize_ratio = resize_ratio.to(self.device)

                # Add channel dimension if needed
                if gt_depth.ndim == 3:
                    gt_depth = gt_depth.unsqueeze(1)
                elif gt_depth.ndim == 4 and gt_depth.shape[1] != 1:
                    gt_depth = gt_depth.unsqueeze(2)

                B, T = images.shape[:2]

                # GT depth from dataloader is inverse depth (1/m), scale to 100/m
                gt_depth_inverse_100 = gt_depth * 100.0

                # Initialize Mamba sequence
                if hasattr(model, 'mamba'):
                    model.mamba.start_new_sequence()

                # Process entire sequence
                B_orig, T_orig, C, H, W = images.shape
                images_flat = rearrange(images, 'b t c h w -> (b t) c h w')
                patch_h, patch_w = H // model.patch_size, W // model.patch_size

                # Extract features
                encoder_features = model.pretrained.get_intermediate_layers(
                    images_flat, model.intermediate_layer_idx[model.encoder]
                )

                # Extract 2-layer CLS tokens
                cls_tokens_list = [
                    encoder_features[i][:, 0]
                    for i in self.encoder_indices
                ]
                cls_tokens_multi_layer = [
                    rearrange(cls_tokens, '(b t) d -> b t d', b=B_orig, t=T_orig)
                    for cls_tokens in cls_tokens_list
                ]

                # Get DPT features
                dpt_features = model.depth_head.get_forward_features(
                    encoder_features, patch_h, patch_w
                )

                # Extract attention weights for importance map
                attention_weights_list = [
                    model.pretrained.blocks[block_idx].attn.attn_weights  # [B*T, num_heads, N+1, N+1]
                    for block_idx in self.target_blocks
                ]

                # Apply FiLM modulation (no grad needed for validation)
                film_outputs = model.gear5_film_head(
                    cls_tokens_multi_layer,
                    attention_weights_list,
                    dpt_features,
                    patch_h, patch_w
                )
                path_1_modulated = film_outputs['path_1_modulated']
                gamma = film_outputs['gamma']
                beta = film_outputs['beta']
                importance_map = film_outputs['importance_map']

                # Apply Mamba (no grad needed for validation)
                path_1_temporal = model.dpt_features_to_mamba(
                    input_shape=(B_orig, T_orig, None, H, W),
                    dpt_features=path_1_modulated,
                    in_dpt_layer=0
                )

                # Final depth prediction (no grad needed for validation)
                out = model.depth_head.scratch.output_conv1(path_1_temporal)
                out = F.interpolate(out, (H, W), mode="bilinear", align_corners=True)
                pred_depth_inverse_100 = model.depth_head.scratch.output_conv2(out)

                pred_depth_inverse = pred_depth_inverse_100

                # Reshape GT
                gt_depth_inverse_flat = rearrange(gt_depth_inverse_100, 'b t 1 h w -> (b t) 1 h w')

                # Interpolate prediction to GT resolution
                gt_shape = gt_depth_inverse_flat.shape[-2:]
                if pred_depth_inverse.shape[-2:] != gt_shape:
                    pred_depth_inverse = F.interpolate(
                        pred_depth_inverse, size=gt_shape, mode="bilinear", align_corners=True
                    )

                H_gt, W_gt = gt_depth_inverse_flat.shape[-2:]

                # Compute validation loss
                pred_depth_flat = pred_depth_inverse.flatten()
                gt_depth_flat = gt_depth_inverse_flat.flatten()

                # Valid mask: GT valid (>0) + Pred positive (no threshold in validation, like Gear2)
                valid_mask = (gt_depth_flat > 0) & (pred_depth_flat > 0)

                # Save masks for visualization
                canonical_gt_valid = (gt_depth_inverse_flat > 0).cpu()
                canonical_pred_valid = (pred_depth_inverse > 0).cpu()

                if valid_mask.sum() > 0:
                    # Compute validation loss based on loss_type
                    epsilon = 1e-3
                    pred_valid = pred_depth_flat[valid_mask].float()
                    gt_valid = gt_depth_flat[valid_mask].float()

                    positive_mask = (pred_valid > 0) & (gt_valid > 0)
                    if positive_mask.sum() == 0:
                        self.logger.warning(f"VALIDATION Step {self.global_step} - No positive depth values!")
                        frame_losses = []
                    else:
                        pred_positive = pred_valid[positive_mask]
                        gt_positive = gt_valid[positive_mask]

                        # Compute log L1 loss
                        loss = torch.abs(
                            torch.log(pred_positive + epsilon) -
                            torch.log(gt_positive + epsilon)
                        )

                        if self.loss_type == 'importance':
                            # Importance-weighted Log L1 Loss
                            # Resize importance_map to image resolution (bilinear for smooth weighting)
                            importance_map_resized = F.interpolate(
                                importance_map.view(B_orig * T_orig, 1, patch_h, patch_w),
                                size=(H_gt, W_gt),
                                mode='bilinear',
                                align_corners=True
                            )  # [B*T, 1, H_gt, W_gt]
                            importance_flat = importance_map_resized.flatten()
                            importance_valid = importance_flat[valid_mask].float()
                            importance_positive = importance_valid[positive_mask]

                            # Compute fg_ratio from upsampled importance map (pixel-level)
                            importance_threshold = importance_flat.mean()
                            fg_ratio_positive = (importance_positive > importance_threshold).float().mean()

                            # Apply importance weighting
                            weighted_loss = loss * (1.0 + fg_ratio_positive * importance_positive)
                            loss_batch = weighted_loss.mean()
                        else:
                            # Standard Log L1 Loss
                            loss_batch = loss.mean()

                        if torch.isnan(loss_batch):
                            self.logger.error(f"VALIDATION Step {self.global_step} - NaN loss detected!")
                            frame_losses = []
                        else:
                            frame_losses = [loss_batch.float()]
                else:
                    frame_losses = []

                # Validation visualization
                pred_depth_inverse_seq = rearrange(pred_depth_inverse, '(b t) 1 h w -> b t 1 h w', b=B_orig, t=T_orig)

                if self.val_visualizer and self.rank == 0:
                    if current_dataset in self.val_vis_config:
                        config = self.val_vis_config[current_dataset]
                        seq_num = dataset_sequence_counters[current_dataset]

                        should_save = (
                            seq_num in config['sequences'] and
                            seq_num not in config['saved']
                        )

                        if should_save:
                            try:
                                pred_depth_inverse_vis = pred_depth_inverse_seq[:, 0]
                                gt_depth_inverse_vis = gt_depth_inverse_100[:, 0]

                                pred_depth_metric = (100.0 / (pred_depth_inverse_vis.float() + 1e-8)).cpu()
                                gt_depth_metric = (100.0 / (gt_depth_inverse_vis.float() + 1e-8)).cpu()

                                gt_h, gt_w = gt_depth_inverse_vis.shape[-2:]

                                img_vis = images[:, 0]
                                img_t_resized = F.interpolate(
                                    img_vis, size=(gt_h, gt_w), mode='bilinear', align_corners=True
                                )

                                canonical_gt_valid_seq = rearrange(canonical_gt_valid, '(b t) 1 h w -> b t 1 h w', b=B_orig, t=T_orig)
                                canonical_pred_valid_seq = rearrange(canonical_pred_valid, '(b t) 1 h w -> b t 1 h w', b=B_orig, t=T_orig)
                                canonical_gt_valid_vis = canonical_gt_valid_seq[:, 0]
                                canonical_pred_valid_vis = canonical_pred_valid_seq[:, 0]

                                # Resize importance map to GT resolution (for smooth visualization)
                                importance_map_vis = importance_map[:, 0:1]  # [B, 1, patch_h, patch_w]
                                importance_map_resized_vis = F.interpolate(
                                    importance_map_vis, size=(gt_h, gt_w), mode='bilinear', align_corners=True
                                )  # [B, 1, gt_h, gt_w]

                                model_outputs = {
                                    'pred_depth': pred_depth_metric,
                                    'canonical_gt_valid': canonical_gt_valid_vis,
                                    'canonical_pred_valid': canonical_pred_valid_vis,
                                    'importance_map': importance_map_resized_vis.float().cpu(),  # [B, 1, gt_h, gt_w] - upsampled
                                    'gamma': gamma[:, 0:1].cpu(),
                                    'beta': beta[:, 0:1].cpu()
                                }

                                sample_batch = (
                                    img_t_resized.unsqueeze(1).float().cpu(),
                                    gt_depth_metric.unsqueeze(1),
                                    dataset_idx,
                                    fx_ratio[:, 0:1].float().cpu(),
                                    resize_ratio[:, 0:1].float().cpu()
                                )

                                val_loss_dict = {
                                    'val_loss': loss_batch.item() if len(frame_losses) > 0 else 0.0
                                }

                                save_name = f"validation_{current_dataset}_seq{seq_num:03d}_step_{self.global_step:06d}"
                                self.val_visualizer.create_validation_summary(
                                    sample_batch, model_outputs, self.global_step,
                                    save_name=save_name, fps=None, loss_dict=val_loss_dict, dataset_name=current_dataset, config=self.config
                                )
                                config['saved'].append(seq_num)
                                self.logger.info(f"Saved validation visualization: {current_dataset} sequence {seq_num} ({len(config['saved'])}/{len(config['sequences'])})")
                            except Exception as e:
                                if self.rank == 0:
                                    self.logger.warning(f"Failed to save validation visualization for {current_dataset} seq {seq_num}: {e}")

                    # Clear intermediate tensors
                    del encoder_features, dpt_features, path_1_temporal
                    del pred_depth_inverse
                    torch.cuda.empty_cache()

            # Track sequence count
            if current_dataset not in dataset_sequence_counts:
                dataset_sequence_counts[current_dataset] = 0
            dataset_sequence_counts[current_dataset] += 1

            # Track loss
            if len(frame_losses) > 0:
                avg_loss = sum(frame_losses) / len(frame_losses)
                total_loss += avg_loss.item()
                num_batches += 1

                if current_dataset not in dataset_losses:
                    dataset_losses[current_dataset] = []
                dataset_losses[current_dataset].append(avg_loss.item())

            # Increment sequence counter
            if current_dataset in dataset_sequence_counters:
                dataset_sequence_counters[current_dataset] += 1

            # Increment total processed count
            total_processed += 1

            # Clear batch memory
            del images, gt_depth, gt_depth_inverse_100
            torch.cuda.empty_cache()

        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')

        # Log detailed statistics
        if self.rank == 0:
            self.logger.info("=" * 80)
            self.logger.info(f"VALIDATION SUMMARY (Step {self.global_step})")
            self.logger.info("=" * 80)
            self.logger.info(f"Total batches with valid pixels: {num_batches}")

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
            'best_step': self.best_step,
            'current_val_loss': self.current_val_loss,
            'dataset_losses': self.dataset_losses,
            'num_sequences': self.num_sequences,
            'config': OmegaConf.to_container(self.config, resolve=True),
            'phase': self.phase
        }

        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")


@hydra.main(version_base=None, config_path="configs/gear5_film", config_name="config")
def main(config: DictConfig):
    """Main entry point"""
    # Initialize distributed training
    rank, world_size, local_rank = init_distributed()

    # Create trainer
    trainer = Gear5FilmTrainer(config, rank, world_size, local_rank)
    trainer.train()

    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
