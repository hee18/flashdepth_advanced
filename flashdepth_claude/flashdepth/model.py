import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import time
from contextlib import nullcontext
from einops import rearrange
from PIL import Image
import logging
from .dinov2 import DINOv2

from .mamba import MambaModel
from .rnn_transformer import TransformerRNN

from .original_dpt import DPTHead
from .hybrid_fusion import HybridFusion

from .util.loss import ScaleAndShiftInvariantLoss
from utils.helpers import *

from utils.eval_metrics.metrics import compute_depth_metrics
from .heads import GlobalScalePredictor, MetricDepthLoss
from .onepiece_modules import UnifiedGlobalMamba, OnepieceMetricHead, SceneCutDetector




class FlashDepth(nn.Module):
    def __init__(
        self,
        vit_size='vitl',
        dpt_dim=256,
        out_channels=[256, 512, 1024, 1024],
        patch_size=14,
        batch_size=4,
        **kwargs
    ):
        super(FlashDepth, self).__init__()

        encoder = vit_size
        model_configs = {
            'vits': {'encoder': 'vits', 'dpt_dim': 64, 'out_channels': [48, 96, 192, 384]},
            'vitl': {'encoder': 'vitl', 'dpt_dim': 256, 'out_channels': [256, 512, 1024, 1024]},
        }

        dpt_dim = model_configs[encoder]['dpt_dim']
        out_channels = model_configs[encoder]['out_channels']

        self.patch_size = patch_size
        
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23], 
        }

        
        self.hybrid_configs = kwargs.get('hybrid_configs')
        if self.hybrid_configs is None or self.hybrid_configs.use_hybrid is False:
            self.hybrid_configs = None
        else:
            self.teacher_model = nn.Module()
            self.teacher_model.pretrained = DINOv2(model_name='vitl', patch_size=patch_size)
            self.teacher_model.depth_head = DPTHead(self.teacher_model.pretrained.embed_dim, dpt_dim=256, out_channels=[256, 512, 1024, 1024])
            self.teacher_model.eval()

            self.hybrid_fusion = HybridFusion(d_model=64, **self.hybrid_configs)


        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder, patch_size=patch_size)

        self.use_mamba = kwargs['use_mamba']
        if self.use_mamba:
            self.downsample_mamba = kwargs['downsample_mamba']
            self.mamba_in_dpt_layer = kwargs['mamba_in_dpt_layer']


            if kwargs.get('use_xlstm', False): 
                from .xlstm_block import xLSTMModel
                self.mamba = xLSTMModel(dpt_dim, training_mode=kwargs['training'], **kwargs)
            
            elif kwargs.get('use_transformer_rnn', False):
                self.mamba = TransformerRNN(dpt_dim, **kwargs)
            else:
                # Ensure required parameters are in kwargs
                mamba_kwargs = kwargs.copy()
                if 'mamba_type' not in mamba_kwargs:
                    mamba_kwargs['mamba_type'] = 'add'  # Default from original
                if 'num_mamba_layers' not in mamba_kwargs:
                    mamba_kwargs['num_mamba_layers'] = 4  # Default for ViT-L from original
                if 'batch_size' not in mamba_kwargs:
                    mamba_kwargs['batch_size'] = batch_size  # From function parameter

                self.mamba = MambaModel(dpt_dim, **mamba_kwargs)
            
            logging.info(f"downsample_mamba: {self.downsample_mamba}")
            logging.info(f"mamba_in_dpt_layer: {self.mamba_in_dpt_layer}")
            
        self.depth_head = DPTHead(self.pretrained.embed_dim, dpt_dim=dpt_dim, out_channels=out_channels)

        # Add Global Scale Predictor for metric depth estimation
        self.use_metric_head = kwargs.get('use_metric_head', False)
        if self.use_metric_head:
            self.gsp_head = GlobalScalePredictor(input_dim=self.pretrained.embed_dim)
            logging.info("Global Scale Predictor initialized for metric depth estimation")

        # Onepiece: Unified Global Mamba + OnepieceMetricHead + SceneCutDetector
        self.use_onepiece = kwargs.get('use_onepiece', False)
        if self.use_onepiece:
            onepiece_mamba_layers = kwargs.get('unified_mamba_layers', 2)
            onepiece_d_state = kwargs.get('unified_mamba_d_state', 64)
            onepiece_d_conv = kwargs.get('unified_mamba_d_conv', 4)

            # CLS(1024) + GAP(256) = 1280 for ViT-L
            cls_dim = self.pretrained.embed_dim  # 1024 for ViT-L
            gap_dim = dpt_dim  # 256 for ViT-L
            unified_dim = cls_dim + gap_dim  # 1280

            self.unified_global_mamba = UnifiedGlobalMamba(
                d_input=unified_dim,
                num_layers=onepiece_mamba_layers,
                d_state=onepiece_d_state,
                d_conv=onepiece_d_conv,
                expand=2,
                headdim=64,
                max_batch_size=batch_size
            )
            self.onepiece_metric_head = OnepieceMetricHead(
                input_dim=unified_dim,
                dpt_dim=gap_dim
            )
            self.scene_cut_detector = SceneCutDetector(
                tau=kwargs.get('scene_cut_tau', 0.05),
                k=kwargs.get('scene_cut_k', 80)
            )
            logging.info(
                f"Onepiece initialized: unified_dim={unified_dim}, "
                f"mamba_layers={onepiece_mamba_layers}, "
                f"d_state={onepiece_d_state}, d_conv={onepiece_d_conv}"
            )
           


    def dpt_features_to_mamba(self, input_shape, dpt_features, in_dpt_layer):
        # reshape to (B, T*h*w, c) for mamba
        if len(input_shape)==4:
            B, C, H, W = input_shape
            T = 1
        else:
            B, T, C, H, W = input_shape
        BT, c, h, w = dpt_features.shape
        assert BT == B*T, f"Expected batch dimension {B*T}, got {BT}" # sanity check

        downsample_factor = self.downsample_mamba[in_dpt_layer]


        if downsample_factor != 1.0:
            original_dpt_features = dpt_features.clone()
            original_dpt_features = rearrange(original_dpt_features, '(b t) c h w -> b t c h w', b=B, t=T)
            dpt_features = F.adaptive_avg_pool2d(dpt_features, (int(h*downsample_factor), int(w*downsample_factor)))

        
        dpt_features = rearrange(dpt_features, '(b t) c h w -> b t (h w) c', b=B, t=T)
        
        mamba_kwargs = dict(Thw = (1, H, W), dpt_shape=(h,w), downsample_factor=downsample_factor, in_dpt_layer=in_dpt_layer)

        # mamba_out = torch.zeros_like(dpt_features)
        mamba_out = []

        for i in range(T):
            seq_out = self.mamba.forward_single_frame(dpt_features[:,i,...], **mamba_kwargs)
            if downsample_factor != 1.0:
                assert self.mamba.mamba_type == 'add'
                if seq_out.ndim == 3:
                    spatial_out = rearrange(seq_out, 'b (h w) c -> b c h w', h=int(h*downsample_factor), w=int(w*downsample_factor))
                else:
                    spatial_out = seq_out
                spatial_out = F.interpolate(spatial_out, (h,w), mode="bilinear", align_corners=True) 
                
                seq_out = rearrange(spatial_out, 'b c h w -> b (h w) c')
                seq_out = self.mamba.final_layer(seq_out)

                spatial_out = rearrange(seq_out, 'b (h w) c -> b c h w', h=h, w=w)
                spatial_out = spatial_out + original_dpt_features[:,i,...]
                seq_out = rearrange(spatial_out, 'b c h w -> b (h w) c')
            
            mamba_out.append(seq_out)

        # reshape back to spatial format (B*T, c, h, w)
        mamba_out = torch.stack(mamba_out, dim=1)
        mamba_out = rearrange(mamba_out, 'b t (h w) c -> (b t) c h w', h=h, w=w, b=B)

       
        return mamba_out

    def get_dpt_features(self, x, input_shape=None):

        self.input_resolution = (x.shape[-1], x.shape[-2]) # w,h
       
        patch_h, patch_w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size

        if self.hybrid_configs is None:
            intermediate_features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder])
            
            if self.use_mamba:
                out = self.depth_head.forward_with_mamba(intermediate_features, patch_h, patch_w, temporal_layer=self.mamba_in_dpt_layer, mamba_fn=self.dpt_features_to_mamba, shape_placeholder=input_shape)
            else:
                out = self.depth_head(intermediate_features, patch_h, patch_w)

            # logging.info(f'out: {out}')

        else:
            # using hybrid model                
            # input resolution: (w,h), assuming height is short side, change if needed
            base_resolution = self.hybrid_configs['teacher_resolution']
            if self.input_resolution[1]>base_resolution:
                main_w = int((base_resolution/self.input_resolution[1])*self.input_resolution[0])
                main_w = (main_w // 14) * 14 # multiple of 14
                main_x = F.interpolate(x, (base_resolution, main_w), mode="bilinear", align_corners=True)
                high_res_x = x
            else:
                ## TODO: resolution < 518, directly run teacher model stream
                main_x = x
                high_res_x = x
            
            # STEP 2: get intermediate features
            student_intermediate_features = self.pretrained.get_intermediate_layers(high_res_x, self.intermediate_layer_idx[self.encoder])
            teacher_intermediate_features = self.teacher_model.pretrained.get_intermediate_layers(main_x, self.intermediate_layer_idx['vitl'])
        
            # STEP 3: get path_4s for fusion
            teacher_dpt_features = self.teacher_model.depth_head.get_path4(teacher_intermediate_features, main_x.shape[-2]//self.patch_size, main_x.shape[-1]//self.patch_size) 
            student_path4 = self.depth_head.get_path4(student_intermediate_features, patch_h, patch_w)
            fused_path4 = self.hybrid_fusion(student_path4, teacher_dpt_features, path_idx=0)


            # STEP 4: run DPT decoder and mamba using fused path_4
            out = self.depth_head.forward_with_mamba(student_intermediate_features, patch_h, patch_w, temporal_layer=self.mamba_in_dpt_layer, mamba_fn=self.dpt_features_to_mamba, shape_placeholder=input_shape,
                                                     fused_path4=fused_path4)

        return out

    def get_cls_token(self, x):
        """
        Extract [CLS] token from DINOv2 encoder

        Args:
            x: Input image tensor [B, C, H, W] or [BT, C, H, W]

        Returns:
            cls_token: [CLS] token features [B, embed_dim] or [BT, embed_dim]
        """
        # Get features from DINOv2 encoder
        features = self.pretrained.forward_features(x)
        cls_token = features['x_norm_clstoken']  # [B, embed_dim] or [BT, embed_dim]

        return cls_token

    def forward_with_metric_head(self, batch, use_mamba=True, **kwargs):
        """
        Forward pass with metric depth estimation using GSP head

        Args:
            batch: Input batch (video, gt_depth) or just video
            use_mamba: Whether to use mamba for temporal processing
            **kwargs: Additional arguments

        Returns:
            Dictionary containing relative and metric depth predictions
        """
        if not self.use_metric_head:
            raise ValueError("Metric head is not enabled. Set use_metric_head=True during initialization.")

        # Handle batch format
        if isinstance(batch, list) or isinstance(batch, tuple):
            if len(batch) == 3:
                video, gt_depth, dataset_idx = batch
            elif len(batch) == 2:
                video, gt_depth = batch
            else:
                video = batch[0]
                gt_depth = None
        elif isinstance(batch, torch.Tensor):
            video = batch
            gt_depth = None
        else:
            video = batch
            gt_depth = None

        # Move video to GPU
        video = video.to(torch.cuda.current_device())
        if gt_depth is not None:
            gt_depth = gt_depth.to(torch.cuda.current_device())

        if use_mamba:
            self.mamba.start_new_sequence()

        B, T, C, H, W = video.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        # Use chunking for large videos to avoid INT_MAX tensor limit
        chunk_size = 50  # Process 50 frames at a time
        relative_depths = []
        cls_tokens = []

        for start_idx in range(0, T, chunk_size):
            end_idx = min(start_idx + chunk_size, T)
            video_chunk = video[:, start_idx:end_idx]  # [B, chunk_T, C, H, W]
            chunk_T = video_chunk.shape[1]

            # Reshape chunk for processing: [B, chunk_T, C, H, W] -> [B*chunk_T, C, H, W]
            video_chunk_flat = rearrange(video_chunk, 'b t c h w -> (b t) c h w')

            # Path A (Frozen): Get relative depth through original FlashDepth pipeline
            with torch.no_grad():
                dpt_features = self.get_dpt_features(video_chunk_flat, input_shape=(B, chunk_T, C, H, W))
                relative_depth_chunk = self.final_head(dpt_features, patch_h, patch_w)  # [B*chunk_T, H, W]
                relative_depths.append(relative_depth_chunk)

            # Path B (Trainable): Get CLS tokens for GSP head
            cls_tokens_chunk = self.get_cls_token(video_chunk_flat)  # [B*chunk_T, embed_dim]
            cls_tokens.append(cls_tokens_chunk)

        # Concatenate all chunks
        relative_depth = torch.cat(relative_depths, dim=0)  # [BT, H, W]
        cls_tokens = torch.cat(cls_tokens, dim=0)  # [BT, embed_dim]

        # Debug: Check relative depth statistics
        if torch.isnan(relative_depth).any():
            print(f"WARNING: NaN values in relative depth!")
        if torch.isinf(relative_depth).any():
            print(f"WARNING: Inf values in relative depth!")

        rel_min, rel_max = relative_depth.min().item(), relative_depth.max().item()
        rel_mean, rel_std = relative_depth.mean().item(), relative_depth.std().item()
        print(f"Relative depth stats - Min: {rel_min:.6f}, Max: {rel_max:.6f}, Mean: {rel_mean:.6f}, Std: {rel_std:.6f}")

        # Debug: Check CLS token statistics
        cls_min, cls_max = cls_tokens.min().item(), cls_tokens.max().item()
        cls_mean, cls_std = cls_tokens.mean().item(), cls_tokens.std().item()
        print(f"CLS token stats - Min: {cls_min:.6f}, Max: {cls_max:.6f}, Mean: {cls_mean:.6f}, Std: {cls_std:.6f}")

        # Predict global scale and shift parameters
        scale, shift = self.gsp_head(cls_tokens)  # Each: [BT, 1]

        # Debug: Check scale and shift values
        scale_mean, shift_mean = scale.mean().item(), shift.mean().item()
        scale_std, shift_std = scale.std().item(), shift.std().item()
        print(f"Scale - Mean: {scale_mean:.6f}, Std: {scale_std:.6f}")
        print(f"Shift - Mean: {shift_mean:.6f}, Std: {shift_std:.6f}")

        # Convert relative depth to metric depth
        metric_depth = self.gsp_head.predict_metric_depth(
            relative_depth, scale, shift
        )  # [BT, H, W]

        # Debug: Check metric depth statistics
        metric_min, metric_max = metric_depth.min().item(), metric_depth.max().item()
        metric_mean, metric_std = metric_depth.mean().item(), metric_depth.std().item()
        print(f"Metric depth stats - Min: {metric_min:.6f}, Max: {metric_max:.6f}, Mean: {metric_mean:.6f}, Std: {metric_std:.6f}")

        # Reshape back to video format if needed
        relative_depth = rearrange(relative_depth, '(b t) h w -> b t h w', b=B, t=T)
        metric_depth = rearrange(metric_depth, '(b t) h w -> b t h w', b=B, t=T)
        scale = rearrange(scale, '(b t) 1 -> b t', b=B, t=T)
        shift = rearrange(shift, '(b t) 1 -> b t', b=B, t=T)

        return {
            'relative_depth': relative_depth,
            'metric_depth': metric_depth,
            'scale': scale,
            'shift': shift
        }

    def train_metric_head(self, batch, loss_fn=None, **kwargs):
        """
        Training function specifically for GSP head fine-tuning

        Args:
            batch: Training batch containing (video, gt_metric_depth)
            loss_fn: Loss function (uses MetricDepthLoss if not provided)
            **kwargs: Additional arguments

        Returns:
            loss: Training loss
            metrics: Dictionary of training metrics
        """
        if loss_fn is None:
            loss_fn = MetricDepthLoss(loss_type='l1')

        video, gt_metric_depth, dataset_idx = batch
        video = video.to(torch.cuda.current_device())
        gt_metric_depth = gt_metric_depth.to(torch.cuda.current_device())

        # Forward pass
        outputs = self.forward_with_metric_head(batch, **kwargs)
        pred_metric_depth = outputs['metric_depth']

        # Reshape for loss computation
        pred_flat = rearrange(pred_metric_depth, 'b t h w -> (b t) h w')
        gt_flat = rearrange(gt_metric_depth, 'b t h w -> (b t) h w')

        # For TartanAir: GT is already metric depth (meters)
        # For other datasets: GT is inverse depth, need conversion
        # TODO: Handle dataset-specific GT format in the future
        # For now, assuming TartanAir (metric depth)
        gt_metric_flat = gt_flat

        # Create valid mask considering both GT and pred ranges to prevent extreme values
        gt_valid_mask = gt_flat > 0  # GT valid pixels
        pred_valid_mask = (pred_flat > 0) & (pred_flat < 1000.0)  # Pred in reasonable range
        valid_mask = gt_valid_mask & pred_valid_mask

        # Compute loss in metric depth space
        loss = loss_fn(pred_flat, gt_metric_flat, valid_mask)

        # Compute metrics for logging
        metrics = {
            'metric_loss': loss.item(),
            'mean_scale': outputs['scale'].mean().item(),
            'mean_shift': outputs['shift'].mean().item(),
            'valid_pixels': valid_mask.sum().item(),
        }

        return loss, metrics

    def _get_intermediate_layers_with_cls(self, x, layer_indices, cls_layer_indices=None):
        """
        Single-pass extraction of both intermediate layer features and the CLS token.
        Avoids calling the ViT encoder twice (get_intermediate_layers + forward_features).

        Args:
            x: Input tensor [B*T, C, H, W]
            layer_indices: List of block indices to extract features from
            cls_layer_indices: Optional list of 0-indexed indices into the intermediate layers
                              for multi-layer CLS averaging. If None, uses last layer only.

        Returns:
            intermediate_features: List of patch token tensors (CLS stripped)
            cls_token: Normalized CLS token [B*T, embed_dim]
                       (averaged across selected layers, or last layer if cls_layer_indices is None)
        """
        # Use the internal method to get raw outputs (including CLS at position 0)
        raw_outputs = self.pretrained._get_intermediate_layers_not_chunked(x, layer_indices)

        # Normalize
        normed_outputs = [self.pretrained.norm(out) for out in raw_outputs]

        # Extract CLS token(s)
        if cls_layer_indices is not None and len(cls_layer_indices) > 0:
            selected_cls = [normed_outputs[idx][:, 0] for idx in cls_layer_indices]
            cls_token = torch.stack(selected_cls, dim=0).mean(dim=0)  # [B*T, embed_dim]
        else:
            cls_token = normed_outputs[-1][:, 0]  # [B*T, embed_dim]

        # Strip CLS and register tokens (same as get_intermediate_layers)
        num_reg = self.pretrained.num_register_tokens  # 0 for DINOv2
        intermediate_features = [out[:, 1 + num_reg:] for out in normed_outputs]

        return intermediate_features, cls_token

    def forward_with_onepiece(self, batch, phase=1, no_shift=False,
                              cls_layer_indices=None, **kwargs):
        """
        Forward pass with Onepiece metric depth estimation.

        Pipeline:
            1. (Frozen) DINOv2 → CLS tokens [B*T, 1024] + intermediate features
            2. (Frozen/Phase2) DPT → dpt_features [B*T, 256, h, w]
            3. GAP(dpt_features) → [B*T, 256]
            4. (Trainable) Concat(CLS, GAP) → [B, T, 1280] → UnifiedGlobalMamba → [B, T, 1280]
            5a. (Trainable) refined_global → MetricHead → scale, shift
            5b. (Trainable) refined_global → FiLM → gamma, beta → modulate dpt_features
            6. (Frozen/Phase2) final_head(modulated_features) → relative_depth
            7. metric_depth = scale * depth_from_relative(relative_depth) + shift

        Args:
            batch: (video, gt_depth, ...) or just video tensor
            phase: 1 or 2 (controls freezing)
            no_shift: If True, zero out shift (scale-only mode)
            cls_layer_indices: Optional list of 0-indexed indices into intermediate layers
                              for multi-layer CLS averaging. If None, uses last layer only.
            **kwargs: Additional arguments

        Returns:
            dict with: relative_depth, metric_depth, scale, shift,
                       dpt_features, cls_tokens, scene_cut_weights, d_cls
        """
        if not self.use_onepiece:
            raise ValueError("Onepiece is not enabled. Set use_onepiece=True.")

        # Handle batch format (Gear5 8-element format)
        if isinstance(batch, (list, tuple)):
            video = batch[0]
        elif isinstance(batch, torch.Tensor):
            video = batch
        else:
            video = batch

        video = video.to(torch.cuda.current_device())
        B, T, C, H, W = video.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        # Reshape: [B, T, C, H, W] → [B*T, C, H, W]
        video_flat = rearrange(video, 'b t c h w -> (b t) c h w')

        # ===== Step 1: DINOv2 encoder (frozen, single pass) =====
        with torch.no_grad():
            encoder_features, cls_tokens_flat = self._get_intermediate_layers_with_cls(
                video_flat, self.intermediate_layer_idx[self.encoder],
                cls_layer_indices=cls_layer_indices
            )

        # ===== Step 2: DPT → dpt_features (no Spatial Mamba) =====
        if phase == 1:
            with torch.no_grad():
                dpt_features = self.depth_head(encoder_features, patch_h, patch_w)
        else:
            dpt_features = self.depth_head(encoder_features, patch_h, patch_w)

        dpt_h, dpt_w = dpt_features.shape[2], dpt_features.shape[3]

        # ===== Step 3: GAP =====
        gap_features = F.adaptive_avg_pool2d(dpt_features, 1).squeeze(-1).squeeze(-1)  # [B*T, 256]

        # ===== Step 4: Unified Global Mamba =====
        # Concat CLS + GAP → [B*T, 1280]
        global_tokens = torch.cat([cls_tokens_flat, gap_features], dim=-1)  # [B*T, 1280]
        global_tokens = rearrange(global_tokens, '(b t) d -> b t d', b=B, t=T)  # [B, T, 1280]

        # Temporal processing through Unified Global Mamba (always trainable)
        refined_global = self.unified_global_mamba(global_tokens)  # [B, T, 1280]
        refined_global_flat = rearrange(refined_global, 'b t d -> (b t) d')  # [B*T, 1280]

        # ===== Step 5a: Scale/Shift prediction =====
        scale, shift, gamma, beta = self.onepiece_metric_head(refined_global_flat)
        # scale: [B*T, 1], shift: [B*T, 1], gamma: [B*T, 256], beta: [B*T, 256]

        # No-shift mode: zero out shift so only scale is used
        if no_shift:
            shift = torch.zeros_like(shift)

        # ===== Step 5b: FiLM modulation on DPT features =====
        gamma_spatial = gamma.unsqueeze(-1).unsqueeze(-1)  # [B*T, 256, 1, 1]
        beta_spatial = beta.unsqueeze(-1).unsqueeze(-1)    # [B*T, 256, 1, 1]
        modulated_features = gamma_spatial * dpt_features + beta_spatial  # [B*T, 256, h, w]

        # ===== Step 6: Final head → relative depth =====
        if phase == 1:
            with torch.no_grad():
                relative_depth = self.final_head(modulated_features, patch_h, patch_w)  # [B*T, H, W]
        else:
            relative_depth = self.final_head(modulated_features, patch_h, patch_w)  # [B*T, H, W]

        # ===== Step 7: Metric depth conversion =====
        # FlashDepth outputs inverse depth * 100, so relative_depth is in 100/m scale
        # depth_from_relative: 100/m → meters: 100 / relative_depth
        depth_from_relative = 100.0 / (relative_depth + 1e-8)  # [B*T, H, W] in meters

        # Apply scale and shift
        scale_spatial = scale.unsqueeze(-1)  # [B*T, 1, 1]
        shift_spatial = shift.unsqueeze(-1)  # [B*T, 1, 1]
        metric_depth = scale_spatial * depth_from_relative + shift_spatial  # [B*T, H, W]

        # ===== Scene cut detection =====
        cls_tokens_seq = rearrange(cls_tokens_flat, '(b t) d -> b t d', b=B, t=T)
        scene_cut_weights, d_cls = self.scene_cut_detector(cls_tokens_seq)

        # Reshape outputs
        relative_depth = rearrange(relative_depth, '(b t) h w -> b t h w', b=B, t=T)
        metric_depth = rearrange(metric_depth, '(b t) h w -> b t h w', b=B, t=T)
        scale = rearrange(scale, '(b t) 1 -> b t', b=B, t=T)
        shift = rearrange(shift, '(b t) 1 -> b t', b=B, t=T)
        dpt_features_seq = rearrange(dpt_features, '(b t) c h w -> b t c h w', b=B, t=T)

        return {
            'relative_depth': relative_depth,       # [B, T, H, W]
            'metric_depth': metric_depth,            # [B, T, H, W]
            'scale': scale,                          # [B, T]
            'shift': shift,                          # [B, T]
            'dpt_features': dpt_features_seq,        # [B, T, 256, h, w]
            'cls_tokens': cls_tokens_seq,            # [B, T, 1024]
            'scene_cut_weights': scene_cut_weights,  # [B, T-1]
            'd_cls': d_cls,                          # [B, T-1]
        }

    def forward_with_onepiece_streaming(self, video, phase=2, no_shift=False,
                                         cls_layer_indices=None):
        """
        Frame-by-frame streaming inference with Mamba temporal state.

        Mathematically equivalent to forward_with_onepiece (Mamba2 parallel scan
        == sequential recurrence), but processes one frame at a time for fair FPS
        measurement and real-world streaming scenarios.

        Args:
            video: [B, T, C, H, W] video tensor (already on GPU)
            phase: 1 or 2 (controls DPT/final_head freezing)
            no_shift: If True, zero out shift (scale-only mode)
            cls_layer_indices: Optional list of 0-indexed indices into intermediate
                              layers for multi-layer CLS averaging.

        Returns:
            dict with: relative_depth, metric_depth, scale, shift,
                       cls_tokens, scene_cut_weights, d_cls
        """
        if not self.use_onepiece:
            raise ValueError("Onepiece is not enabled. Set use_onepiece=True.")

        B, T, C, H, W = video.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        # Reset Mamba hidden state for new sequence
        self.unified_global_mamba.start_new_sequence()

        all_metric_depth = []
        all_scale = []
        all_shift = []
        all_relative_depth = []
        all_cls_tokens = []

        ctx = torch.no_grad() if phase == 1 else nullcontext()

        for t in range(T):
            frame = video[:, t]  # [B, 3, H, W]

            # Step 1: DINOv2 encoder (always frozen)
            with torch.no_grad():
                encoder_features, cls_token = self._get_intermediate_layers_with_cls(
                    frame, self.intermediate_layer_idx[self.encoder],
                    cls_layer_indices=cls_layer_indices
                )  # cls_token: [B, 1024]

            # Step 2: DPT (no spatial Mamba in Onepiece)
            with ctx:
                dpt_features = self.depth_head(encoder_features, patch_h, patch_w)  # [B, 256, h, w]

            # Step 3: GAP
            gap = F.adaptive_avg_pool2d(dpt_features, 1).squeeze(-1).squeeze(-1)  # [B, 256]

            # Step 4: Mamba streaming (single frame with hidden state)
            global_token = torch.cat([cls_token, gap], dim=-1).unsqueeze(1)  # [B, 1, 1280]
            refined = self.unified_global_mamba.forward_single_frame(global_token).squeeze(1)  # [B, 1280]

            # Step 5a: MetricHead → scale, shift, FiLM params
            scale, shift, gamma, beta = self.onepiece_metric_head(refined)

            if no_shift:
                shift = torch.zeros_like(shift)

            # Step 5b: FiLM modulation on DPT features
            modulated = gamma.unsqueeze(-1).unsqueeze(-1) * dpt_features + beta.unsqueeze(-1).unsqueeze(-1)

            # Step 6: final_head → relative depth
            with ctx:
                relative_depth = self.final_head(modulated, patch_h, patch_w)  # [B, H, W]

            # Step 7: Metric depth conversion
            depth_from_relative = 100.0 / (relative_depth + 1e-8)
            metric_depth = scale.unsqueeze(-1) * depth_from_relative + shift.unsqueeze(-1)

            all_metric_depth.append(metric_depth)
            all_scale.append(scale.squeeze(-1))
            all_shift.append(shift.squeeze(-1))
            all_relative_depth.append(relative_depth)
            all_cls_tokens.append(cls_token)

        # Stack results along time dimension
        metric_depth = torch.stack(all_metric_depth, dim=1)   # [B, T, H, W]
        scale = torch.stack(all_scale, dim=1)                  # [B, T]
        shift = torch.stack(all_shift, dim=1)                  # [B, T]
        relative_depth = torch.stack(all_relative_depth, dim=1)  # [B, T, H, W]
        cls_tokens_seq = torch.stack(all_cls_tokens, dim=1)    # [B, T, 1024]

        # Scene cut detection (post-hoc on all CLS tokens, logging only)
        scene_cut_weights, d_cls = self.scene_cut_detector(cls_tokens_seq)

        return {
            'relative_depth': relative_depth,       # [B, T, H, W]
            'metric_depth': metric_depth,            # [B, T, H, W]
            'scale': scale,                          # [B, T]
            'shift': shift,                          # [B, T]
            'cls_tokens': cls_tokens_seq,            # [B, T, 1024]
            'scene_cut_weights': scene_cut_weights,  # [B, T-1]
            'd_cls': d_cls,                          # [B, T-1]
        }

    def final_head(self, x, patch_h, patch_w):

        out  = self.depth_head.scratch.output_conv1(x)

        bs = out.shape[0]
        target_h = int(patch_h * self.patch_size)
        target_w = int(patch_w * self.patch_size)

        # Process in batches of 30 frames to avoid memory issues
        # int max is 2147483647; for B,C=128,H=518,W=518, can only handle 60 frames
        # for vit-s using raw 2k resolution, can only handle 30 frames (2147483647/(32*1064*1904)=33)
        outputs = []
        for i in range(0, bs, 30):
            batch = out[i:i+30]  # Take up to 30 frames
            batch_out = F.interpolate(batch, (target_h, target_w),
                                    mode="bilinear", align_corners=True)
            outputs.append(batch_out)

        out = torch.cat(outputs, dim=0)
        out = self.depth_head.scratch.output_conv2(out)
        # if out.max() <=0:
        #     logging.warning("Depth is all zeros")
        depth = F.relu(out).squeeze(1)

        return depth


    def train_sequence(self, batch, loss_type='l1', vis_training=False, savedir='debug_training', **kwargs):
        # both have shape (B, T, C, H, W)
        video, gt_depth = batch 
        video = video.to(torch.cuda.current_device())
        
        # multiplying gt disparity by 100 -> 1/meters to 100/meters; 
        # magic number for training stability because gt is in meters but depthanythingv2 original output values in the hundreds
        gt_depth = gt_depth.to(torch.cuda.current_device())* 100 

        self.mamba.start_new_sequence()

        B, T, C, H, W = video.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        # reshape to (B*T, C, H, W) for ViT
        video = rearrange(video, 'b t c h w -> (b t) c h w')
        dpt_features = self.get_dpt_features(video, input_shape=(B, T, C, H, W)) # (B*T, c, h, w) where c=dpt_dim, h,w are also downsampled versions
        pred_depth = self.final_head(dpt_features, patch_h, patch_w) # (B*T, H, W)

        # loss
        gt_depth = rearrange(gt_depth, 'b t h w -> (b t) h w')
        valid_mask = gt_depth >=0


        if 'l1' in loss_type: 
            loss = F.l1_loss(pred_depth[valid_mask], gt_depth[valid_mask])
        elif 'scaleshift' in loss_type:
            loss = ScaleAndShiftInvariantLoss()(pred_depth, gt_depth, mask=valid_mask)

        # tried implementing temporal loss from Video Depth Anything paper; didn't seem to help results
        # keeping for reference
        if 'temporal' in loss_type and kwargs['timestep'] > 500:
            l1_loss = loss 
            
            # Reshape back to B,T,H,W for temporal processing
            pred_temporal = rearrange(pred_depth, '(b t) h w -> b t h w', b=B)
            gt_temporal = rearrange(gt_depth, '(b t) h w -> b t h w', b=B)
            valid_temporal = rearrange(valid_mask, '(b t) h w -> b t h w', b=B)
            

            temporal_loss = 0
            K = int(loss_type.split('k')[1])  # # Maximum time step difference to consider (e.g., 2 means up to t and t+2)

            for k in range(1, K + 1):
                pred_diff_k = pred_temporal[:, k:] - pred_temporal[:, :-k]
                gt_diff_k = gt_temporal[:, k:] - gt_temporal[:, :-k]
                
                valid_diff_k = (valid_temporal[:, k:] & valid_temporal[:, :-k])
                
                # Small change condition based on mean depth
                # for each pixel independently, it checks if the depth change (gt_diff_k) is less than 20% of that pixel’s mean depth (mean_depth_k).
                mean_depth_k = (gt_temporal[:, k:] + gt_temporal[:, :-k]) / 2
                # relative_threshold_k = 0.2 * mean_depth_k
                # small_change_mask_k = torch.abs(gt_diff_k) < relative_threshold_k
                relative_change_k   = gt_diff_k.abs() / (mean_depth_k.abs() + 1e-6)  
                small_change_mask_k = (relative_change_k < 0.2) 
                
                temporal_mask_k = valid_diff_k & small_change_mask_k
                
                if temporal_mask_k.sum() > 0:
                    loss_k = F.l1_loss(pred_diff_k[temporal_mask_k], gt_diff_k[temporal_mask_k])
                    temporal_loss += loss_k
            
            loss = l1_loss + 0.5 * temporal_loss
            # loss = dict(total=loss, l1_loss=l1_loss, temporal_loss=temporal_loss*0.5)
       
       
        
        ### debug, visualize training data
        grid = None
        if vis_training:
            if dist.get_rank() == 0:
                with torch.no_grad():
                    pred_depth_vis = rearrange(pred_depth.clone().cpu(), '(b t) h w -> b t h w', b=B)
                    gt_depth_vis = rearrange(gt_depth.clone().cpu(), '(b t) h w -> b t h w', b=B)
                    video_vis = rearrange(video.clone().cpu(), '(b t) c h w -> b t c h w', b=B)
                
                    import os; os.makedirs(savedir, exist_ok=True)
                    for i in range(B):
                        try:
                            pred_save = depth_to_np_arr(pred_depth_vis[i])
                            video_save = torch_batch_to_np_arr(video_vis[i])
                            gt_save = depth_to_np_arr(gt_depth_vis[i])
                            grid = save_gifs_as_grid(video_save, pred_frames=pred_save, gt_frames=gt_save, duration=160,
                                                output_path=f'{savedir}/{vis_training}_{i}.gif', fixed_height=224)
                        except Exception as e:
                            logging.info(f"Visualization error for iter {vis_training}: {e}") 
                            pass
            dist.barrier()
 
 
        return loss, grid
    
    
    @torch.no_grad()
    def forward(self, batch, use_mamba, gif_path, resolution, out_mp4 ,save_depth_npy=False, save_vis_map=False, **kwargs):

        # both have shape (B, T, C, H, W)
        if isinstance(batch, list) or isinstance(batch, tuple):
            video, gt_depth = batch
        elif isinstance(batch, torch.Tensor):
            video = batch
            gt_depth = None

        # For fair FPS measurement: pre-load entire video to GPU (exclude data transfer from measurement)
        if kwargs.get('print_time', False):
            video = video.to(torch.cuda.current_device())
            if gt_depth is not None:
                gt_depth = gt_depth.to(torch.cuda.current_device())

        preds = []

        loss = 0
        if use_mamba:
            self.mamba.start_new_sequence()

        for i in range(video.shape[1]):
            warmup_frames = 5
            if kwargs.get('print_time', False) and i==warmup_frames:
                torch.cuda.synchronize()
                start = time.time()

            # For FPS measurement: video already on GPU, just index
            if kwargs.get('print_time', False):
                frame = video[:, i, :, :, :]
            else:
                frame = video[:, i, :, :, :].to(torch.cuda.current_device())
            B, C, H, W = frame.shape


            patch_h, patch_w = frame.shape[-2] // self.patch_size, frame.shape[-1] // self.patch_size

            # dpt_features = self.get_dpt_features(frame)
            dpt_features = self.get_dpt_features(frame, input_shape=(B,C,H,W))

            pred_depth = self.final_head(dpt_features, patch_h, patch_w)
            pred_depth = torch.clip(pred_depth, min=0)


            if gt_depth is not None and pred_depth.shape != gt_depth[:, i, :, :].shape:
                pred_depth = F.interpolate(pred_depth.unsqueeze(1), gt_depth[:, i, :, :].unsqueeze(1).shape[-2:], mode="bilinear", align_corners=True).squeeze(1)

            if gt_depth is not None:
                # For FPS measurement: gt_depth already on GPU
                if kwargs.get('print_time', False):
                    gt_frame = gt_depth[:, i, :, :]
                else:
                    gt_frame = gt_depth[:, i, :, :].to(torch.cuda.current_device())
                valid_mask = gt_frame >=0
                # if loss_type == 'l1':
                #     loss += F.l1_loss(pred_depth[valid_mask], gt_frame[valid_mask])
                # elif loss_type == 'scaleshift':
                #     # loss += ScaleAndShiftInvariantLoss()(pred_depth, gt_frame, mask=valid_mask)
                #     loss += F.l1_loss(pred_depth[valid_mask], gt_frame[valid_mask])
                # else:
                loss += F.l1_loss(pred_depth[valid_mask], gt_frame[valid_mask])

            preds.append(pred_depth)

        if kwargs.get('print_time', False):
            try:
                torch.cuda.synchronize()
                end = time.time()
                logging.info(f'Inference FPS (data pre-loaded): {((video.shape[1]-warmup_frames) / (end - start)):.2f} | wall time: {end - start:.2f}s | num frames: {video.shape[1]-warmup_frames}')
            except Exception as e:
                logging.info(f"Error in printing time: {e}")
                pass
        
        if kwargs.get('dummy_timing', False):
            return 0,0


        return self.save_and_return(video, gt_depth, preds, loss, save_depth_npy, gif_path, save_vis_map, out_mp4, resolution, kwargs)




    @torch.compiler.disable
    def save_and_return(self, video, gt_depth, preds, loss, save_depth_npy, gif_path, save_vis_map, out_mp4, resolution, kwargs):

        grid = None
        if gt_depth is not None and kwargs.get('use_metrics', True):
            l1_loss = loss / video.shape[1]

            # calculating metrics across sequence
            preds_tensor = torch.stack(preds, dim=1).cpu().float() # (1, T, H, W)
            gt_depth = gt_depth.cpu().float() # (1, T, H, W)
            loss = compute_depth_metrics(preds_tensor.squeeze(0), gt_depth.squeeze(0))
            loss['l1_loss'] = l1_loss.item()

        
        if save_depth_npy:
            test_idx = gif_path.rstrip('.gif').split('_')[-1]
            npy_path = os.path.join(os.path.dirname(gif_path), 'depth_npy_files') #, test_idx)
            os.makedirs(npy_path, exist_ok=True)
            for i in range(len(preds)):
                np.save(f'{npy_path}/frame_{i}.npy', preds[i].cpu().float().numpy().squeeze(0))
        
        if kwargs.get('out_video', True):
            try:
                pred0 = []
                for i in range(len(preds)):
                    pred0.append(preds[i][0].cpu())
                pred0 = torch.stack(pred0)
                inverse_colormap = kwargs.get('inverse', False)
                pred_save = depth_to_np_arr(pred0, inverse=inverse_colormap)
                video_save = torch_batch_to_np_arr(video[0])
                if gt_depth is not None:
                    gt_save = depth_to_np_arr(gt_depth[0], inverse=inverse_colormap)
                else:
                    gt_save = None

                # inferno heat map
                if save_vis_map:
                    test_idx = gif_path.rstrip('.gif').split('_')[-1]
                    vis_map_path = os.path.join(os.path.dirname(gif_path), 'vis_maps') #, test_idx)
                    os.makedirs(vis_map_path, exist_ok=True)
                    for i in range(len(pred_save)):
                        Image.fromarray(pred_save[i]).save(f'{vis_map_path}/frame_{i}.png')

                os.makedirs(os.path.dirname(gif_path), exist_ok=True)
                if not out_mp4:
                    grid = save_gifs_as_grid(video_save,gt_save,pred_save, output_path=gif_path, fixed_height=resolution)
                else:
                    grid = save_grid_to_mp4(video_save,gt_save,pred_save, output_path=gif_path.replace('.gif', '.mp4'), fixed_height=video.shape[-2])
            except Exception as e:
                logging.info(f"Error in saving video: {e}")
                pass
    
        return loss, grid



    # not using mamba
    def train_single(self, batch, loss_type='l1', vis_training=False, savedir='debug_training'):

        images, gt_depth = batch
        images = images.to(torch.cuda.current_device()).squeeze(1)
        gt_depth = gt_depth.to(torch.cuda.current_device()).squeeze(1)*100

        assert images.ndim == 4, f"{images.shape}; image ndim should only be 4"

        patch_h, patch_w = images.shape[-2] // self.patch_size, images.shape[-1] // self.patch_size

        dpt_features = self.get_dpt_features(images)
        pred_depth = self.final_head(dpt_features, patch_h, patch_w) # (B, H, W)

        valid_mask = gt_depth >=0
        loss = F.l1_loss(pred_depth[valid_mask], gt_depth[valid_mask])
   
        grid = None
        if vis_training:
            if dist.get_rank() == 0:
                with torch.no_grad():
                    import os; os.makedirs(savedir, exist_ok=True)
                    try:
                        pred_depth = torch.clip(pred_depth, min=0)
                        pred_save = depth_to_np_arr(pred_depth)
                        video_save = torch_batch_to_np_arr(images)
                        gt_save = depth_to_np_arr(gt_depth)
                        grid = save_gifs_as_grid(video_save, pred_frames=pred_save, gt_frames=gt_save, 
                                            output_path=f'{savedir}/{vis_training}.gif', fixed_height=224)
                    except Exception as e:
                        logging.info(f"Visualization error for iter {vis_training}: {e}") 
                        pass
            dist.barrier()
      

        return loss, grid
