"""
Gear5 Training Script: Two-Stage Global + Foreground Modulation

Two-stage training approach:
    Step 1: Global Scale & Shift Prediction (GSP)
        - Input: Multi-layer CLS tokens [4, 11, 17, 23]
        - Output: Global scale and shift
        - Trainable: GSP + Mamba + Final head
        - Frozen: ViT + DPT
        - Loss: gt valid & pred inlier pixels

    Step 2: Foreground-only Modulation
        - Input: Globally-modulated DPT + Multi-layer attention [4, 11, 17, 23]
        - Output: FG-modulated features
        - Trainable: FG modulation + Mamba + Final head
        - Frozen: ViT + DPT + GSP (from Step 1)
        - Loss: gt valid & pred inlier & FG pixels

Key features:
    - Two-stage training: --step 1, --step 2, or --step 1,2
    - Global modulation from multi-layer CLS tokens
    - FG-only modulation (BG keeps global modulation)
    - Canonical space normalization (focal_length=1000)
    - Resolution: 518×518 for both steps
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
from utils.gear3_upgrade_visualization import Gear3UpgradeVisualizer  # Can reuse for now
from flashdepth.gear5_modules import (
    GlobalScalePredictorMultiLayer,
    ForegroundOnlyModulationHead,
    Gear5MetricHead
)
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
    Trainer for Gear5 two-stage metric depth learning.

    Step 1:
        Frozen: ViT + DPT
        Trainable: Global GSP + Mamba + Final head

    Step 2:
        Frozen: ViT + DPT + GSP (from Step 1)
        Trainable: FG modulation + Mamba + Final head
    """
    def __init__(self, config, rank, world_size, local_rank):
        self.config = config
        self.step = config.get('step', 1)  # Training step (1 or 2)
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank

        # Detect phase based on config_variant (similar to Gear3 Upgrade)
        # Phase 1: 518×518, config_l or config_s (Step 1 or 2)
        # Phase 2: 2K, config_hybrid (Step 2 only)
        config_variant = config.get('config_variant', 'l')
        self.phase = 2 if config_variant == 'hybrid' else 1

        if self.phase == 2 and self.step != 2:
            raise ValueError("Phase 2 (Hybrid) requires step=2!")

        # Setup device
        self.device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)

        if rank == 0:
            logging.info(f"Training Step {self.step}, Phase {self.phase} on {world_size} GPU(s)")
            if self.phase == 2:
                logging.info(f"  Phase 2 (Hybrid): 2K resolution, Gear5-S weights + FlashDepth-hybrid")

        # Setup results directory (only rank 0)
        phase_suffix = f"_phase{self.phase}" if self.phase > 1 else ""
        step_suffix = f"_step{self.step}"
        self.results_dir = Path(config.get('results_dir', f'./train_results/gear5{step_suffix}{phase_suffix}'))
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
            self.logger.info(f"Training step: {self.step}")

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
                name=f"gear5_step{self.step}_{config.training.get('wandb_name', 'experiment')}",
                config=dict(config)
            )

        # Setup visualizer with separate folders
        self.train_visualizer = Gear3UpgradeVisualizer(save_dir=self.results_dir / "visualizations" / "train")
        self.val_visualizer = Gear3UpgradeVisualizer(save_dir=self.results_dir / "visualizations" / "valid")

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
        Get canonical focal length (fixed at 1000.0 for all resolutions).

        Returns:
            float: Canonical focal length (always 1000.0)
        """
        return 1000.0

    def _setup_model(self):
        """Initialize FlashDepth with Gear5 modules (GSP + FG modulation)"""
        # Gear5 ALWAYS uses multi_layer separation method
        self.separation_method = 'multi_layer'

        # Create base FlashDepth model
        model_config = dict(self.config.model)
        model_config['batch_size'] = self.config.training.batch_size
        model_config['use_metric_head'] = False  # Don't use original GSP head

        model = FlashDepth(**model_config)

        # Get model architecture info
        embed_dim = 1024 if model.encoder == 'vitl' else 384
        dpt_dim = 256 if model.encoder == 'vitl' else 64

        # Load pre-trained checkpoint
        # Phase 1, Step 1: Load FlashDepth weights (DINOv2 + DPT + Mamba + Final head)
        # Phase 1, Step 2: Load Step 1 checkpoint (all modules including Gear5)
        # Phase 2, Step 2: Load Gear5-S checkpoint (will load later after creating gear5_metric_head)
        if self.step == 1:
            checkpoint_path = self.config.get('load')
        elif self.phase == 2:
            # Phase 2 (Hybrid): Load Gear5-S checkpoint after creating gear5_metric_head
            checkpoint_path = self.config.get('gear_checkpoint')
            if not checkpoint_path:
                raise ValueError(
                    "Phase 2 (Hybrid) requires 'gear_checkpoint' in config! "
                    "Set gear_checkpoint to your Gear5-S Phase 1 checkpoint path."
                )
            if self.rank == 0:
                self.logger.info(f"Phase {self.phase}: Will load Gear5-S checkpoint from {checkpoint_path}")
        else:
            # Phase 1, Step 2: Load Step 1 checkpoint
            checkpoint_path = self.config.get('gear_checkpoint')
            if not checkpoint_path:
                raise ValueError(
                    "Step 2 requires 'gear_checkpoint' in config! "
                    "Set gear_checkpoint to your Step 1 checkpoint path."
                )
            if self.rank == 0:
                self.logger.info(f"Step {self.step}: Loading Step 1 checkpoint from {checkpoint_path}")

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

                if self.step == 1:
                    # Phase 1, Step 1: Load ALL parameters from FlashDepth (DINOv2, DPT, Mamba, output_conv)
                    # Only exclude gear5_metric_head (will be created and trained from scratch)
                    loaded_dict = {}
                    excluded_keys = []
                    for k, v in state_dict.items():
                        # Exclude only gear5_metric_head (doesn't exist in FlashDepth checkpoint)
                        if 'gear5_metric_head' in k:
                            excluded_keys.append(k)
                        else:
                            loaded_dict[k] = v

                    # Load state dict (strict=False to allow missing gear5_metric_head)
                    model.load_state_dict(loaded_dict, strict=False)
                    self.logger.info(f"Phase 1, Step 1: Loaded {len(loaded_dict)} parameters from FlashDepth checkpoint")
                    self.logger.info(f"  - DINOv2 encoder: ✓ (will be frozen)")
                    self.logger.info(f"  - DPT projects/resize/refinenet: ✓ (will be frozen)")
                    self.logger.info(f"  - Mamba: ✓ (will be trained)")
                    self.logger.info(f"  - output_conv1/2: ✓ (will be trained)")
                    if excluded_keys:
                        self.logger.info(f"Excluded {len(excluded_keys)} parameters (gear5_metric_head will be created)")
                elif self.phase == 2:
                    # Phase 2: Skip loading here, will load after creating gear5_metric_head
                    pass
                else:
                    # Phase 1, Step 2: Load ALL parameters including gear5_metric_head
                    # But need to add gear5_metric_head first before loading
                    pass  # Will load after gear5_metric_head is created
            else:
                self.logger.warning(f"Checkpoint {checkpoint_path} not found")

        # Add Gear3 Upgrade metric head
        embed_dim = 1024 if model.encoder == 'vitl' else 384
        dpt_dim = 256 if model.encoder == 'vitl' else 64
        num_heads = 16 if model.encoder == 'vitl' else 6

        # Gear5 uses its own metric head (global GSP + FG modulation)
        # GSP fusion: uniform mix (25:25:25:25), consistent with FFM
        model.gear5_metric_head = Gear5MetricHead(
            embed_dim=embed_dim,
            dpt_dim=dpt_dim
        )

        # Phase 1, Step 2: Load ALL parameters after gear5_metric_head is created
        if self.step == 2 and self.phase == 1 and checkpoint_path and checkpoint_path != 'true' and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            # Load all parameters including gear5_metric_head from Step 1
            model.load_state_dict(state_dict, strict=False)
            self.logger.info(f"Phase 1, Step 2: Loaded ALL parameters from Step 1 checkpoint")
            self.logger.info(f"  - DINOv2, DPT, Mamba, output_conv, Gear5 (GSP + FFM): ✓")
            self.logger.info(f"  - DINOv2 and DPT are already frozen from Step 1 (same as FlashDepth checkpoint)")

        # Phase 2, Step 1 (Hybrid): Load GSP+Mamba+output from Phase1 Step2, overwrite DINOv2+DPT from FlashDepth-hybrid (NO FFM)
        elif self.step == 1 and self.phase == 2 and checkpoint_path and checkpoint_path != 'true' and os.path.exists(checkpoint_path):
            # Step 1: Load Phase1 Step2 checkpoint (GSP, Mamba, output_conv) - EXCLUDE FFM
            self.logger.info(f"Phase 2, Step 1: Loading from Phase1 Step2 checkpoint (excluding FFM): {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            # Filter out FFM keys (gear5_metric_head.fg_modulation_head.*)
            filtered_state_dict = {}
            excluded_ffm_keys = []
            for k, v in state_dict.items():
                if k.startswith('gear5_metric_head.fg_modulation_head.'):
                    excluded_ffm_keys.append(k)
                else:
                    filtered_state_dict[k] = v

            # Load filtered state dict (GSP, Mamba, output_conv, but NO FFM)
            model.load_state_dict(filtered_state_dict, strict=False)
            self.logger.info(f"Phase 2, Step 1: Loaded {len(filtered_state_dict)} parameters from Phase1 Step2")
            self.logger.info(f"  - GSP (gear5_metric_head.gsp): ✓")
            self.logger.info(f"  - Mamba: ✓")
            self.logger.info(f"  - output_conv1/2: ✓")
            self.logger.info(f"  - FFM: ✗ (excluded {len(excluded_ffm_keys)} keys, will NOT be used in Step 1)")

            # Step 2: Overwrite DINOv2 + DPT with FlashDepth-hybrid weights
            hybrid_path = self.config.get('load', 'configs/flashdepth/iter_43002.pth')
            if os.path.exists(hybrid_path):
                self.logger.info(f"Phase 2, Step 1: Loading FlashDepth-hybrid weights from {hybrid_path}")
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

                # Load ONLY DINOv2 and DPT parameters (overwrite Phase1 weights with Hybrid)
                loaded_hybrid = {}
                for k, v in hybrid_state_dict.items():
                    # Include only encoder and DPT refinement (exclude Mamba, output_conv, gear5_metric_head)
                    if not any(x in k for x in ['mamba', 'hybrid_fusion', 'teacher_model',
                                                 'output_conv1', 'output_conv2', 'gear5_metric_head']):
                        loaded_hybrid[k] = v

                # Overwrite DINOv2 + DPT parameters
                model.load_state_dict(loaded_hybrid, strict=False)
                self.logger.info(f"Phase 2, Step 1: Overwritten {len(loaded_hybrid)} DINOv2 + DPT parameters with Hybrid weights")
                self.logger.info(f"  - DINOv2 (ViT-L + ViT-S + Cross Attn): ✓")
                self.logger.info(f"  - DPT: ✓")
                self.logger.info(f"  - Kept from Phase1 Step2: GSP, Mamba, output_conv")
                self.logger.info(f"  - FFM: Will NOT be loaded (Step 1 trains without FFM)")
            else:
                self.logger.warning(f"FlashDepth-hybrid checkpoint {hybrid_path} not found! Using Phase1 DINOv2 + DPT.")

        # Phase 2, Step 2 (Hybrid): Load from Phase2 Step1, add FFM from Phase1 Step2
        elif self.step == 2 and self.phase == 2 and checkpoint_path and checkpoint_path != 'true' and os.path.exists(checkpoint_path):
            # Step 1: Load Phase2 Step1 checkpoint (GSP, Mamba, output_conv, DINOv2, DPT - but NO FFM yet)
            self.logger.info(f"Phase 2, Step 2: Loading from Phase2 Step1 checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            # Load all parameters from Phase2 Step1 (should NOT have FFM yet)
            model.load_state_dict(state_dict, strict=False)
            self.logger.info(f"Phase 2, Step 2: Loaded base model from Phase2 Step1")
            self.logger.info(f"  - DINOv2 + DPT (from FlashDepth-hybrid): ✓")
            self.logger.info(f"  - GSP + Mamba + output_conv (from Phase1 Step2): ✓")
            self.logger.info(f"  - FFM: Not yet loaded (will load from Phase1 Step2 next)")

            # Step 2: Load FFM from Phase1 Step2 checkpoint
            phase1_step2_path = self.config.get('phase1_step2_checkpoint')
            if phase1_step2_path and phase1_step2_path != 'true' and os.path.exists(phase1_step2_path):
                self.logger.info(f"Phase 2, Step 2: Loading FFM from Phase1 Step2 checkpoint: {phase1_step2_path}")
                phase1_checkpoint = torch.load(phase1_step2_path, map_location='cpu')

                # Extract state dict
                if isinstance(phase1_checkpoint, dict) and 'model' in phase1_checkpoint:
                    phase1_state_dict = phase1_checkpoint['model']
                elif isinstance(phase1_checkpoint, dict) and 'state_dict' in phase1_checkpoint:
                    phase1_state_dict = phase1_checkpoint['state_dict']
                else:
                    phase1_state_dict = phase1_checkpoint

                # Remove module. prefix if present
                phase1_state_dict = {k.replace('module.', ''): v for k, v in phase1_state_dict.items()}

                # Extract ONLY FFM parameters
                ffm_state_dict = {}
                for k, v in phase1_state_dict.items():
                    if k.startswith('gear5_metric_head.fg_modulation_head.'):
                        ffm_state_dict[k] = v

                # Load FFM parameters
                if ffm_state_dict:
                    model.load_state_dict(ffm_state_dict, strict=False)
                    self.logger.info(f"Phase 2, Step 2: Loaded {len(ffm_state_dict)} FFM parameters from Phase1 Step2")
                    self.logger.info(f"  - FFM (fg_modulation_head): ✓")
                    self.logger.info(f"  - Now ready to train full model with FFM!")
                else:
                    self.logger.warning(f"No FFM parameters found in Phase1 Step2 checkpoint!")
            else:
                self.logger.warning(f"Phase1 Step2 checkpoint not provided or not found. FFM will be randomly initialized!")

        # Enable attention weights storage
        # Gear5 ALWAYS uses multi_layer separation method
        # - Step 1 (GSP): [4, 11, 17, 23] - all 4 DPT layers
        # - Step 2 (FG): [11, 17] - mid 2 DPT layers only
        if self.step == 1:
            # Step 1: GSP module uses all 4 DPT layers
            multi_layer_blocks = {
                'vitl': [4, 11, 17, 23],
                'vits': [2, 5, 8, 11]
            }
        else:  # Step 2
            # Step 2: FG feature module uses mid 2 DPT layers only
            multi_layer_blocks = {
                'vitl': [11, 17],
                'vits': [5, 8]
            }
        target_blocks = multi_layer_blocks[model.encoder]

        for i, block in enumerate(model.pretrained.blocks):
            if i in target_blocks:
                block.attn.store_attn_weights = True
                self.logger.info(f"Enabled attention weights storage for block {i}")
            else:
                block.attn.store_attn_weights = False

        self.logger.info(f"Multi-layer attention fusion (Step {self.step}): storing attention from blocks {target_blocks}")

        # Store target blocks and compute encoder_features indices
        # encoder_features from get_intermediate_layers returns features at intermediate_layer_idx
        # For vitl: intermediate_layer_idx = [4, 11, 17, 23], encoder_features has indices [0, 1, 2, 3]
        # For vits: intermediate_layer_idx = [2, 5, 8, 11], encoder_features has indices [0, 1, 2, 3]
        intermediate_idx = model.intermediate_layer_idx[model.encoder]

        # Map target_blocks to encoder_features indices
        # e.g., target_blocks = [4, 11, 17, 23], intermediate_idx = [4, 11, 17, 23]
        #       encoder_indices = [0, 1, 2, 3] (all 4 for Step 1)
        # e.g., target_blocks = [11, 17], intermediate_idx = [4, 11, 17, 23]
        #       encoder_indices = [1, 2] (middle 2 for Step 2)
        encoder_indices = [intermediate_idx.index(block) for block in target_blocks]
        self.encoder_indices = encoder_indices
        self.logger.info(f"Encoder features indices for Step {self.step}: {encoder_indices}")

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
        Freeze configuration based on step and phase:

        Step 1:
            Frozen: DINOv2, DPT
            Trainable: GSP, Mamba, output_conv

        Step 2 (Phase 1 or 2):
            Frozen: DINOv2, DPT, GSP
            Trainable: FFM, Mamba, output_conv
        """
        frozen_params = 0
        mamba_params = 0
        output_conv_params = 0
        gsp_params = 0
        ffm_params = 0

        for name, param in model.named_parameters():
            # Gear5 metric head - step-specific freezing
            if 'gear5_metric_head.global_gsp' in name:
                # GSP: trainable only in Step 1
                param.requires_grad = (self.step == 1)
                gsp_params += param.numel()
                if param.requires_grad:
                    self.logger.info(f"Trainable (GSP): {name} - {param.shape}")

            elif 'gear5_metric_head.fg_modulation_head' in name:
                # FFM: trainable only in Step 2
                param.requires_grad = (self.step == 2)
                ffm_params += param.numel()
                if param.requires_grad:
                    self.logger.info(f"Trainable (FFM): {name} - {param.shape}")

            elif 'mamba' in name:
                # Mamba: always trainable
                param.requires_grad = True
                mamba_params += param.numel()

            elif 'output_conv' in name:
                # DPT output head: always trainable
                param.requires_grad = True
                output_conv_params += param.numel()
                self.logger.info(f"Trainable (output_conv): {name} - {param.shape}")

            else:
                # Everything else (DINOv2, DPT refinement): frozen
                param.requires_grad = False
                frozen_params += param.numel()

        # Log summary
        self.logger.info(f"=== Parameter Configuration (Step {self.step}, Phase {self.phase}) ===")
        self.logger.info(f"Frozen:")
        self.logger.info(f"  - DINOv2 + DPT: {frozen_params:,}")
        if self.step == 2:
            self.logger.info(f"  - GSP: {gsp_params:,}")

        self.logger.info(f"Trainable:")
        if self.step == 1:
            self.logger.info(f"  - GSP: {gsp_params:,}")
        else:
            self.logger.info(f"  - FFM: {ffm_params:,}")
        self.logger.info(f"  - Mamba: {mamba_params:,}")
        self.logger.info(f"  - output_conv: {output_conv_params:,}")

        total_frozen = frozen_params + (gsp_params if self.step == 2 else 0)
        total_trainable = mamba_params + output_conv_params + (gsp_params if self.step == 1 else ffm_params)
        self.logger.info(f"Total frozen: {total_frozen:,}")
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
            if any(keyword in name for keyword in ['gear5_metric_head', 'mamba', 'output_conv']):
                continue

            # Set frozen parts to eval mode
            module.eval()

    def _setup_data_loaders(self):
        """Setup phase-specific data loaders"""
        if self.step == 1:
            # Phase 1: 5 datasets, 518×518
            train_datasets = self.config.dataset.get('train_datasets',
                ['mvs-synth', 'dynamicreplica', 'tartanair', 'pointodyssey', 'spring'])
            val_datasets = self.config.dataset.get('val_datasets', ['sintel', 'waymo'])
            resolution = 'base'  # 518×518
        elif self.step == 1.5:
            # Phase 1.5: nuScenes fine-tuning, 518×518 (optional)
            train_datasets = ['nuscenes']
            val_datasets = ['nuscenes']
            resolution = 'base'  # 518×518 (same as Phase 1)
        elif self.step == 2:
            # Phase 2: mvs-synth, spring only, 2K resolution (Hybrid)
            train_datasets = self.config.dataset.get('train_datasets', ['mvs-synth', 'spring'])
            val_datasets = self.config.dataset.get('val_datasets', ['sintel', 'waymo_seg'])
            resolution = '2k'
        else:
            raise ValueError(f"Invalid step: {self.step}. Must be 1, 1.5, or 2.")

        if self.rank == 0:
            self.logger.info(f"Step {self.step} - Train datasets: {train_datasets}")
            self.logger.info(f"Step {self.step} - Val datasets: {val_datasets}")
            self.logger.info(f"Step {self.step} - Resolution: {resolution}")

        # Training dataset
        train_dataset = CombinedDataset(
            root_dir=self.config.dataset.data_root,
            enable_dataset_flags=train_datasets,
            resolution=resolution,  # Use phase-specific resolution
            split='train',
            video_length=self.config.dataset.video_length,
            color_aug=False  # No augmentation for metric training
        )

        # Validation dataset: use shorter sequence for Phase 2/3 to save memory
        # Phase 2/3 use 2K resolution which requires much more memory
        val_video_length = 1 if self.step >= 2 else self.config.dataset.video_length
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
            self.logger.info(f"Val batch size: 1, video_length: {val_video_length} {'(reduced for 2K)' if self.step >= 2 else '(like original FlashDepth)'}")

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
                    images, gt_depth, focal_lengths, dataset_idx = batch
                    images = images.to(self.device)
                    gt_depth = gt_depth.to(self.device)
                    focal_lengths = focal_lengths.to(self.device)

                    if gt_depth.ndim == 3:
                        gt_depth = gt_depth.unsqueeze(1)
                    elif gt_depth.ndim == 4 and gt_depth.shape[1] != 1:
                        gt_depth = gt_depth.unsqueeze(2)

                    # GT depth from dataloader is inverse depth (1/m), scale to 100/m for training
                    gt_depth_inverse_100 = gt_depth * 100.0  # (1/m) * 100 = 100/m

                    # Get batch size for Mamba initialization
                    B_orig, T_orig, C, H, W = images.shape

                    # Apply canonical space transformation if enabled (for visualization consistency)
                    if self.config.get('use_canonical_space', False):
                        CANONICAL_FX = self._get_canonical_focal_length()

                        # Transform inverse depth directly to canonical space
                        # inverse_canonical = inverse_actual * (fx_actual / CANONICAL_FX)
                        fx_actual = focal_lengths.view(B_orig, T_orig, 1, 1, 1)
                        gt_depth_inverse_100 = gt_depth_inverse_100 * (fx_actual / CANONICAL_FX)

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
                        # Use encoder_indices to index into encoder_features
                        # Step 1: encoder_indices = [0, 1, 2, 3] (all 4 layers)
                        # Step 2: encoder_indices = [1, 2] (middle 2 layers)
                        cls_tokens_list = [
                            encoder_features[i][:, 0]  # CLS token from each layer
                            for i in self.encoder_indices
                        ]
                        attention_weights_multi_layer = [
                            model.pretrained.blocks[block_idx].attn.attn_weights
                            for block_idx in self.target_blocks
                        ]

                        # Apply Gear5 modulation BEFORE Mamba
                        # Returns: dict with keys based on step
                        gear5_outputs = model.gear5_metric_head(
                            cls_tokens_list=cls_tokens_list,
                            patch_tokens=patch_tokens,
                            attention_weights_multi_layer=attention_weights_multi_layer,
                            dpt_features=path_1,
                            patch_h=patch_h,
                            patch_w=patch_w,
                            step=self.step
                        )

                        path_1_modulated = gear5_outputs['modulated_features']
                        scale = gear5_outputs['scale']
                        shift = gear5_outputs['shift']

                        # Step 2: Extract additional outputs
                        if self.step == 2:
                            importance_map = gear5_outputs['importance_map']
                            fg_features = gear5_outputs['fg_features']
                            fg_mask = gear5_outputs['fg_mask']
                            bg_mask = 1.0 - fg_mask  # Derive BG mask from FG mask
                        else:
                            # Step 1: Set to None
                            importance_map = None
                            fg_features = None
                            fg_mask = None
                            bg_mask = None

                        # Apply Mamba temporal modeling to modulated feature
                        path_1_temporal = model.dpt_features_to_mamba(
                            input_shape=(B_orig, T_orig, None, H, W),
                            dpt_features=path_1_modulated,
                            in_dpt_layer=0
                        )

                        out = model.depth_head.scratch.output_conv1(path_1_temporal)
                        out = F.interpolate(out, (H, W), mode="bilinear", align_corners=True)
                        out = model.depth_head.scratch.output_conv2(out)

                        # Save prediction inverse depth for mask calculation
                        pred_depth_inverse = out  # [B*T, 1, H, W]

                        # Convert prediction to metric depth: 100/m -> m
                        # Output shape: (B*T, 1, H, W)
                        pred_depth_metric = 100.0 / (out + 1e-8)

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
                            focal_lengths[:1, :1].float().cpu()  # [1, 1] - resized focal length
                        )

                        # Step 2: Include importance map and masks
                        if self.step == 2:
                            # Reshape outputs from (B*T, ...) to (B, T, ...)
                            importance_map_seq = rearrange(importance_map, '(b t) c h w -> b t c h w', b=B_orig, t=T_orig)
                            fg_mask_seq = rearrange(fg_mask, '(b t) c h w -> b t c h w', b=B_orig, t=T_orig)
                            bg_mask_seq = rearrange(bg_mask, '(b t) c h w -> b t c h w', b=B_orig, t=T_orig)

                            # Select first frame for visualization
                            importance_map_vis = importance_map_seq[:, 0]  # [B, C, H, W]
                            fg_mask_vis = fg_mask_seq[:, 0]  # [B, 1, H, W]
                            bg_mask_vis = bg_mask_seq[:, 0]  # [B, 1, H, W]

                            # Resize masks to match image resolution
                            importance_map_resized = F.interpolate(
                                importance_map_vis, size=(H, W), mode='bilinear', align_corners=True
                            )
                            fg_mask_resized = F.interpolate(
                                fg_mask_vis, size=(H, W), mode='bilinear', align_corners=True
                            )
                            bg_mask_resized = F.interpolate(
                                bg_mask_vis, size=(H, W), mode='bilinear', align_corners=True
                            )

                            # Step 2: Include all outputs + canonical masks
                            model_outputs_cpu = {
                                'pred_depth': pred_depth_metric_vis[:1].cpu(),  # [1, 1, H, W]
                                'importance_map': importance_map_resized[:1].cpu(),  # [1, 1, H, W]
                                'fg_mask': fg_mask_resized[:1].cpu(),  # [1, 1, H, W]
                                'bg_mask': bg_mask_resized[:1].cpu(),   # [1, 1, H, W]
                                'canonical_gt_valid': canonical_gt_valid_vis[:1].cpu(),  # [1, 1, H, W] - canonical space mask
                                'canonical_pred_valid': canonical_pred_valid_vis[:1].cpu()  # [1, 1, H, W] - canonical space mask
                            }
                        else:
                            # Step 1: Only pred_depth + canonical masks (no FG/BG masks)
                            model_outputs_cpu = {
                                'pred_depth': pred_depth_metric_vis[:1].cpu(),  # [1, 1, H, W]
                                'canonical_gt_valid': canonical_gt_valid_vis[:1].cpu(),  # [1, 1, H, W] - canonical space mask
                                'canonical_pred_valid': canonical_pred_valid_vis[:1].cpu()  # [1, 1, H, W] - canonical space mask
                            }

                        # FPS removed from training (only measured in test_gear3.py)
                        current_fps = None

                        # Extract layer_weights for visualization (multi_layer separation only, Step 2 only)
                        layer_weights = None
                        if self.separation_method == 'multi_layer' and self.step == 2:
                            try:
                                # Handle DDP wrapping
                                model = self.model.module if hasattr(self.model, 'module') else self.model
                                # Gear5: multi_layer_fusion is in fg_modulation_head (Step 2 only)
                                fusion_weights = model.gear5_metric_head.fg_modulation_head.multi_layer_fusion.fusion_weights
                                layer_weights = torch.softmax(fusion_weights, dim=0).detach().cpu().numpy()
                            except Exception as e:
                                self.logger.warning(f"Failed to extract layer_weights: {e}")

                        # Pass loss_dict and layer_weights for visualization
                        self.train_visualizer.create_validation_summary(
                            sample_batch, model_outputs_cpu, step, prefix="training", fps=current_fps, loss_dict=loss_dict, layer_weights=layer_weights, config=self.config
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
                self.save_checkpoint(f'checkpoint_step{step}_step{self.step}.pth')

        # Final save
        self.save_checkpoint('last.pth')
        self.logger.info("Training completed!")

    def train_step(self, batch):
        """Single training step with BFloat16 autocast"""
        # Unpack batch (updated to include focal_lengths)
        images, gt_depth, focal_lengths, dataset_idx = batch
        images = images.to(self.device)
        gt_depth = gt_depth.to(self.device)
        focal_lengths = focal_lengths.to(self.device)  # Shape: (B, T)

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

        # Apply canonical space transformation if enabled
        if self.config.get('use_canonical_space', False):
            CANONICAL_FX = self._get_canonical_focal_length()

            # Transform inverse depth directly to canonical space
            # inverse_canonical = inverse_actual * (fx_actual / CANONICAL_FX)
            fx_actual = focal_lengths.view(B, T, 1, 1, 1)
            gt_depth_inverse_100 = gt_depth_inverse_100 * (fx_actual / CANONICAL_FX)

            if self.global_step < 5:
                self.logger.info(f"DEBUG - Canonical space enabled (CANONICAL_FX={CANONICAL_FX})")
                self.logger.info(f"DEBUG - fx_actual range: {fx_actual.min():.1f} - {fx_actual.max():.1f}")
                self.logger.info(f"DEBUG - After canonical transform: min={gt_depth_inverse_100.min():.4f}, max={gt_depth_inverse_100.max():.4f}")

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

            # Gear5: Collect CLS tokens and attention weights from multiple layers
            # Use encoder_indices to index into encoder_features
            # Step 1: encoder_indices = [0, 1, 2, 3] (all 4 layers)
            # Step 2: encoder_indices = [1, 2] (middle 2 layers)
            with torch.no_grad():
                cls_tokens_list = [
                    encoder_features[i][:, 0]  # CLS token from each layer
                    for i in self.encoder_indices
                ]
                attention_weights_multi_layer = [
                    model.pretrained.blocks[block_idx].attn.attn_weights.detach()
                    for block_idx in self.target_blocks
                ]

            # Apply Gear5 modulation to path_1 (BEFORE Mamba)
            # This makes the feature metric-aware before temporal modeling
            # Inputs: (B*T, ...), Output: (B*T, ...)
            gear5_outputs = model.gear5_metric_head(
                cls_tokens_list=cls_tokens_list,
                patch_tokens=patch_tokens,
                attention_weights_multi_layer=attention_weights_multi_layer,
                dpt_features=path_1,
                patch_h=patch_h,
                patch_w=patch_w,
                step=self.step
            )

            path_1_modulated = gear5_outputs['modulated_features']
            scale = gear5_outputs['scale']
            shift = gear5_outputs['shift']

            # Step 2: Extract additional outputs
            if self.step == 2:
                importance_map = gear5_outputs['importance_map']
                fg_features = gear5_outputs['fg_features']
                fg_mask = gear5_outputs['fg_mask']

            # Note: attention_weights_multi_layer will be garbage collected automatically
            # No need to manually set to None (validation also needs attention weights)

            # Apply Mamba temporal modeling to modulated (metric-aware) feature
            # Mamba is trainable in Phase 1+, so gradients flow through it
            path_1_temporal = model.dpt_features_to_mamba(
                input_shape=(B_orig, T_orig, None, H, W),
                dpt_features=path_1_modulated,
                in_dpt_layer=0  # Single layer index
            )  # Returns (B*T, dpt_dim, h, w) with temporal consistency

            # Pass through DPT output head (trainable)
            out = model.depth_head.scratch.output_conv1(path_1_temporal)
            out = F.interpolate(out, (H, W), mode="bilinear", align_corners=True)
            out = model.depth_head.scratch.output_conv2(out)

            # Prediction is already positive (Softplus activation in output_conv2)
            # Shape: (B*T, 1, H, W)
            pred_depth_inverse = out

        # Compute loss (outside autocast for stability)
        # Reshape GT from (B, T, 1, H, W) to (B*T, H, W)
        gt_depth_inverse_flat = rearrange(gt_depth_inverse_100, 'b t 1 h w -> (b t) h w')

        # Remove channel dimension from prediction
        pred_depth_inverse_flat = pred_depth_inverse.squeeze(1)  # (B*T, H, W)

        # DEBUG: Check values in first few steps
        if self.global_step < 5:
            self.logger.info(f"DEBUG - Pred inverse depth: min={pred_depth_inverse_flat.min():.4f}, max={pred_depth_inverse_flat.max():.4f}, mean={pred_depth_inverse_flat.mean():.4f}")

        # Clamp prediction to reasonable range to prevent NaN
        pred_depth_inverse_flat = torch.clamp(pred_depth_inverse_flat, min=1e-3, max=1e4)

        # Compute valid mask: GT valid + Pred outlier filtering
        # GT valid: Only compute loss where GT is valid (70m threshold)
        # Pred outlier: Filter out extreme predictions (>200m outliers)
        if self.global_step < 100:
            MIN_INVERSE_DEPTH = 100.0 / 200.0  # Relaxed: 200m threshold for first 100 steps
        else:
            MIN_INVERSE_DEPTH = 100.0 / 70.0   # Normal: 70m threshold after warmup

        # GT valid mask: where GT depth is within valid range
        gt_valid_mask = (gt_depth_inverse_flat > MIN_INVERSE_DEPTH)

        # Pred outlier mask: filter extreme predictions (>200m is outlier)
        MAX_DEPTH_OUTLIER = 200.0
        MIN_INVERSE_OUTLIER = 100.0 / MAX_DEPTH_OUTLIER
        pred_outlier_mask = (pred_depth_inverse_flat > MIN_INVERSE_OUTLIER)

        # Final mask: GT valid AND pred not outlier
        valid_mask = gt_valid_mask & pred_outlier_mask

        if valid_mask.sum() == 0:
            self.logger.error("No valid GT & Pred pixels in batch!")
            return {'loss': 0.0}

        # Compute loss (like original FlashDepth)
        with torch.amp.autocast('cuda', enabled=False):
            loss = self.loss_fn(
                pred_depth_inverse_flat.float(),
                gt_depth_inverse_flat.float(),
                valid_mask.float()
            )

        # Backward pass (outside autocast for numerical stability)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        return {
            'loss': loss.item()
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

            # Unpack batch (updated to include focal_lengths)
            images, gt_depth, focal_lengths, dataset_idx = batch

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
                focal_lengths = focal_lengths.to(self.device)  # Shape: (B, T)

                # Add channel dimension if needed
                if gt_depth.ndim == 3:
                    gt_depth = gt_depth.unsqueeze(1)
                elif gt_depth.ndim == 4 and gt_depth.shape[1] != 1:
                    gt_depth = gt_depth.unsqueeze(2)

                B, T = images.shape[:2]

                # GT depth from dataloader is inverse depth (1/m), scale to 100/m for training
                gt_depth_inverse_100 = gt_depth * 100.0  # (1/m) * 100 = 100/m

                # Apply canonical space transformation if enabled
                if self.config.get('use_canonical_space', False):
                    CANONICAL_FX = self._get_canonical_focal_length()

                    # Transform inverse depth directly to canonical space
                    # inverse_canonical = inverse_actual * (fx_actual / CANONICAL_FX)
                    fx_actual = focal_lengths.view(B, T, 1, 1, 1)
                    gt_depth_inverse_100 = gt_depth_inverse_100 * (fx_actual / CANONICAL_FX)

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

                # Gear5: Collect CLS tokens and attention weights from multiple layers
                # Use encoder_indices to index into encoder_features
                # Step 1: encoder_indices = [0, 1, 2, 3] (all 4 layers)
                # Step 2: encoder_indices = [1, 2] (middle 2 layers)
                cls_tokens_list = [
                    encoder_features[i][:, 0]  # CLS token from each layer
                    for i in self.encoder_indices
                ]
                attention_weights_multi_layer = [
                    model.pretrained.blocks[block_idx].attn.attn_weights.detach()
                    for block_idx in self.target_blocks
                ]

                # Apply Gear5 modulation BEFORE Mamba (metric-aware feature)
                # Inputs/outputs: (B*T, ...)
                gear5_outputs = model.gear5_metric_head(
                    cls_tokens_list=cls_tokens_list,
                    patch_tokens=patch_tokens,
                    attention_weights_multi_layer=attention_weights_multi_layer,
                    dpt_features=path_1,
                    patch_h=patch_h,
                    patch_w=patch_w,
                    step=self.step
                )

                path_1_modulated = gear5_outputs['modulated_features']
                scale = gear5_outputs['scale']
                shift = gear5_outputs['shift']

                # Step 2: Extract additional outputs
                if self.step == 2:
                    importance_map = gear5_outputs['importance_map']
                    fg_features = gear5_outputs['fg_features']
                    fg_mask = gear5_outputs['fg_mask']
                    bg_mask = 1.0 - fg_mask  # Derive BG mask from FG mask
                else:
                    # Step 1: Set to None
                    importance_map = None
                    fg_features = None
                    fg_mask = None
                    bg_mask = None

                # Note: attention_weights_multi_layer will be garbage collected automatically
                # No need to manually set to None (validation also needs attention weights)

                # Apply Mamba temporal modeling to modulated feature
                path_1_temporal = model.dpt_features_to_mamba(
                    input_shape=(B_orig, T_orig, None, H, W),
                    dpt_features=path_1_modulated,
                    in_dpt_layer=0
                )  # Returns (B*T, dpt_dim, h, w) with temporal consistency

                # Get depth (at model resolution)
                out = model.depth_head.scratch.output_conv1(path_1_temporal)
                out = F.interpolate(out, (H, W), mode="bilinear", align_corners=True)
                out = model.depth_head.scratch.output_conv2(out)

                pred_depth_inverse = out  # [B*T, 1, H, W] at model resolution

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

                # Compute loss for entire sequence
                # Validation: Use same threshold (70m) for both GT and Pred for fair evaluation
                MIN_INVERSE_DEPTH = 100.0 / 70.0  # 70m threshold (consistent with test)

                # GT valid mask: where GT depth is within valid range
                gt_valid_mask = (gt_depth_inverse_flat >= MIN_INVERSE_DEPTH)

                # Pred valid mask: same threshold as GT for fair evaluation
                pred_valid_mask = (pred_depth_inverse >= MIN_INVERSE_DEPTH)

                # Final mask: GT valid AND pred valid (both use 70m threshold)
                valid_mask = (gt_valid_mask & pred_valid_mask).float()

                # Save canonical masks for visualization (before any reshaping)
                canonical_gt_valid = gt_valid_mask.cpu()  # [B*T, 1, H, W]
                canonical_pred_valid = pred_valid_mask.cpu()  # [B*T, 1, H, W]

                if valid_mask.sum() > 0:
                    loss_batch = self.loss_fn(pred_depth_inverse, gt_depth_inverse_flat, valid_mask)
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

                                # Step 2: Include importance map and masks
                                if self.step == 2:
                                    # Reshape importance/mask maps to (B, T, ...) and select first frame
                                    importance_map_seq = rearrange(importance_map, '(b t) c h w -> b t c h w', b=B_orig, t=T_orig)
                                    fg_mask_seq = rearrange(fg_mask, '(b t) c h w -> b t c h w', b=B_orig, t=T_orig)
                                    bg_mask_seq = rearrange(bg_mask, '(b t) c h w -> b t c h w', b=B_orig, t=T_orig)

                                    importance_map_vis = importance_map_seq[:, 0]  # [B, C, h, w]
                                    fg_mask_vis = fg_mask_seq[:, 0]
                                    bg_mask_vis = bg_mask_seq[:, 0]

                                    # Resize importance map to GT resolution
                                    importance_map_resized = F.interpolate(
                                        importance_map_vis, size=(gt_h, gt_w), mode='bilinear', align_corners=True
                                    )

                                    # Resize FG/BG masks to GT resolution
                                    fg_mask_resized = F.interpolate(
                                        fg_mask_vis, size=(gt_h, gt_w), mode='bilinear', align_corners=True
                                    )
                                    bg_mask_resized = F.interpolate(
                                        bg_mask_vis, size=(gt_h, gt_w), mode='bilinear', align_corners=True
                                    )

                                    # Step 2: Include all outputs
                                    model_outputs = {
                                        'pred_depth': pred_depth_metric,  # [B, 1, gt_h, gt_w] at GT resolution
                                        'importance_map': importance_map_resized.float().cpu(),  # [B, 1, gt_h, gt_w]
                                        'fg_mask': fg_mask_resized.float().cpu(),  # [B, 1, gt_h, gt_w]
                                        'bg_mask': bg_mask_resized.float().cpu(),   # [B, 1, gt_h, gt_w]
                                        'canonical_gt_valid': canonical_gt_valid_vis,  # [B, 1, H, W] - canonical space mask
                                        'canonical_pred_valid': canonical_pred_valid_vis  # [B, 1, H, W] - canonical space mask
                                    }
                                else:
                                    # Step 1: Only pred_depth and canonical masks (no importance map/fg/bg masks)
                                    model_outputs = {
                                        'pred_depth': pred_depth_metric,  # [B, 1, gt_h, gt_w] at GT resolution
                                        'canonical_gt_valid': canonical_gt_valid_vis,  # [B, 1, H, W] - canonical space mask
                                        'canonical_pred_valid': canonical_pred_valid_vis  # [B, 1, H, W] - canonical space mask
                                    }

                                # For visualization, we need [B, T, ...] format like training
                                # But we only have one frame (t=0), so unsqueeze T dimension
                                sample_batch = (
                                    img_t_resized.unsqueeze(1).float().cpu(),  # [B, 1, C, gt_h, gt_w] at GT resolution
                                    gt_depth_metric.unsqueeze(1),  # [B, 1, gt_h, gt_w]
                                    dataset_idx,
                                    focal_lengths[:, 0:1].float().cpu()  # [B, 1] - resized focal length for first frame
                                )

                                # FPS removed from training (only measured in test_gear3.py)
                                current_fps = None

                                # Create loss_dict with current frame loss (for visualization)
                                val_loss_dict = {
                                    'val_loss': loss_batch.item() if len(frame_losses) > 0 else 0.0
                                }

                                # Extract layer_weights for visualization (multi_layer separation only, Step 2 only)
                                layer_weights = None
                                if self.separation_method == 'multi_layer' and self.step == 2:
                                    try:
                                        # Handle DDP wrapping
                                        model = self.model.module if hasattr(self.model, 'module') else self.model
                                        # Gear5: multi_layer_fusion is in fg_modulation_head (Step 2 only)
                                        fusion_weights = model.gear5_metric_head.fg_modulation_head.multi_layer_fusion.fusion_weights
                                        layer_weights = torch.softmax(fusion_weights, dim=0).detach().cpu().numpy()
                                    except Exception as e:
                                        self.logger.warning(f"Failed to extract layer_weights: {e}")

                                # Save with dataset and sequence-specific name
                                save_name = f"validation_{current_dataset}_seq{seq_num:03d}_step_{self.global_step:06d}"
                                self.val_visualizer.create_validation_summary(
                                    sample_batch, model_outputs, self.global_step,
                                    save_name=save_name, fps=current_fps, loss_dict=val_loss_dict, dataset_name=current_dataset, layer_weights=layer_weights, config=self.config
                                )
                                config['saved'].append(seq_num)
                                self.logger.info(f"Saved validation visualization: {current_dataset} sequence {seq_num} ({len(config['saved'])}/{len(config['sequences'])})")
                            except Exception as e:
                                if self.rank == 0:
                                    self.logger.warning(f"Failed to save validation visualization for {current_dataset} seq {seq_num}: {e}")

                    # Clear intermediate tensors to free memory after each sequence
                    del encoder_features, attention_weights, patch_tokens, dpt_features, path_1, path_1_temporal
                    del path_1_modulated, pred_depth_inverse
                    # Clear step-2-only tensors (importance_map, fg_features, fg_mask, bg_mask)
                    if self.step == 2:
                        del importance_map, fg_features, fg_mask, bg_mask
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

            # WARNING: Check if validation set is too small (Phase 2/3)
            if self.step >= 2 and num_batches < 20:
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
            'phase': self.step
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
