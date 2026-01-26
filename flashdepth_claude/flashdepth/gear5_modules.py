"""
Gear5 Modules: GRU-based Temporal Scale Prediction with Importance-weighted Loss

This module implements a unified single-stage metric depth training approach:

Single Stage: Temporal Scale & Shift Prediction
- Input: 2-layer CLS tokens [Layers 11, 23 for ViT-L / 5, 11 for ViT-S]
- Processing: GRU for temporal consistency
- Output: Frame-wise scale and shift parameters
- Applied to: Final relative depth output (after output_conv2)
- Formula: D_metric = Scale × D_relative + Shift

Key Features:
1. GRU provides temporal consistency across frames
2. Importance map for loss weighting (attention-based)
3. Frozen FlashDepth components (ViT, DPT, Mamba, output_conv)
4. Only ~132K trainable parameters

Architecture:
    ViT (frozen) → DPT (frozen) → Mamba (frozen) → output_conv (frozen) → Relative Depth
                                                                              ↓
    CLS tokens → TemporalScalePredictor → Scale/Shift → Metric Depth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


# ==================== Importance Map Generator ====================

class ImportanceMapGenerator(nn.Module):
    """
    Generate importance map from multi-layer CLS attention weights.

    Pipeline:
        1. Extract CLS-to-patch attention from multiple layers
        2. Average across layers
        3. Resize to spatial dimensions
        4. Normalize to [0, 1]

    Input: List of attention weights from [Layer 11, 23] (or [5, 11] for ViT-S)
    Output: Importance map [B, T, H, W]
    """
    def __init__(self, num_layers=2):
        super().__init__()
        self.num_layers = num_layers

        logging.info(f"ImportanceMapGenerator: Averaging {num_layers} attention layers")

    def forward(self, attention_weights_list, patch_h, patch_w):
        """
        Args:
            attention_weights_list: List of [B, num_heads, N+1, N+1] attention weights
                                   Length = num_layers (e.g., 2 for layers [11, 23])
            patch_h, patch_w: Spatial patch dimensions (e.g., 37×37 for 518×518)

        Returns:
            importance_map: [B, 1, patch_h, patch_w] normalized importance scores
        """
        # Extract CLS-to-patch attention from each layer
        cls_to_patch_list = []
        for attn in attention_weights_list:
            # attn: [B, num_heads, N+1, N+1]
            # CLS row: attn[:, :, 0, 1:]  -> [B, num_heads, N]
            cls_to_patch = attn[:, :, 0, 1:]  # [B, num_heads, num_patches]
            cls_to_patch = cls_to_patch.mean(dim=1)  # Average over heads: [B, num_patches]
            cls_to_patch_list.append(cls_to_patch)

        # Average across layers
        cls_attention = torch.stack(cls_to_patch_list, dim=0).mean(dim=0)  # [B, num_patches]

        # Reshape to spatial dimensions
        num_patches = cls_attention.shape[1]
        expected_patches = patch_h * patch_w

        if num_patches != expected_patches:
            logging.warning(
                f"Patch mismatch: got {num_patches}, expected {expected_patches}. "
                f"Interpolating to {patch_h}×{patch_w}."
            )
            # Linear interpolation
            cls_attention = F.interpolate(
                cls_attention.unsqueeze(1), size=expected_patches,
                mode='linear', align_corners=True
            ).squeeze(1)

        # Reshape to 2D: [B, patch_h, patch_w]
        importance_map = cls_attention.view(-1, patch_h, patch_w).unsqueeze(1)  # [B, 1, patch_h, patch_w]

        # Remove register token (highest attention patch) with 3×3 inpainting
        B = importance_map.shape[0]
        for b in range(B):
            attn_2d = importance_map[b, 0]  # [patch_h, patch_w]

            # Find the patch with maximum attention (register token)
            max_val = attn_2d.max()
            outlier_mask = (attn_2d == max_val)  # Only the single highest patch

            # Inpaint with local average (3×3 box filter at patch level)
            kernel = torch.ones(1, 1, 3, 3, device=importance_map.device) / 9
            attn_smoothed = F.conv2d(
                importance_map[b:b+1], kernel, padding=1
            )
            importance_map[b, 0] = torch.where(
                outlier_mask,
                attn_smoothed[0, 0],
                importance_map[b, 0]
            )

        # Percentile normalization (1-99 percentile) to [0, 1]
        # More robust than min-max: reduces sensitivity to remaining outliers
        for b in range(B):
            attn_flat = importance_map[b].flatten()
            attn_p1 = torch.quantile(attn_flat, 0.01)   # 1st percentile
            attn_p99 = torch.quantile(attn_flat, 0.99)  # 99th percentile

            # Normalize to [0, 1] and clip
            importance_map[b] = (importance_map[b] - attn_p1) / (attn_p99 - attn_p1 + 1e-8)
            importance_map[b] = torch.clamp(importance_map[b], 0.0, 1.0)

        return importance_map


# ==================== Temporal Scale Predictor (GRU-based) ====================

class TemporalScalePredictor(nn.Module):
    """
    Temporal scale and shift predictor for metric depth conversion.
    Supports both GRU and Mamba2 backends.

    Architecture:
        CLS tokens [B, T, 1024]
            ↓ Feature Extractor
        Features [B, T, 256]
            ↓ GRU or Mamba2 (temporal modeling)
        Hidden states [B, T, 128]
            ↓ Scale/Shift Heads
        Scale [B, T], Shift [B, T]

    Key Benefits:
        1. GRU: Lightweight, fast for short sequences (T<50)
        2. Mamba2: Better long-range modeling, efficient for T>100
        3. Temporal consistency across frames
        4. Lightweight (~132K params for GRU, ~400K for Mamba2)

    Parameters:
        - Feature Net: (1024 × 256) + 256 = 262,400
        - GRU: ~100K (input=256, hidden=128, layers=1)
        - Mamba2: ~200K (d_model=256, expand=2)
        - Scale Head: (128 × 1) + 1 = 129
        - Shift Head: (128 × 1) + 1 = 129
        - Total: ~362K (GRU) or ~462K (Mamba2) parameters
    """
    def __init__(self, embed_dim=1024, feature_dim=256, hidden_dim=128, num_layers=1, use_mamba=False):
        super().__init__()

        self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_mamba = use_mamba

        # 1. Feature Extractor: Reduce CLS token dimensionality
        self.feature_net = nn.Sequential(
            nn.Linear(embed_dim, feature_dim),
            nn.ReLU(inplace=True)
        )

        # 2. Temporal modeling: GRU or Mamba2
        if use_mamba:
            # Mamba2: Use MambaBlock for temporal modeling
            from .mamba import MambaBlock

            # NOTE: This is the NEW Mamba2 for TemporalScalePredictor ONLY
            # The original FlashDepth Mamba modules are FROZEN during training
            self.temporal_mamba = MambaBlock(
                d_model=feature_dim,  # 256
                layer_idx=0,  # Single layer
                expand=2,  # Standard expansion
                d_state=64,
                d_conv=4,
                headdim=64,
                use_hydra=False
            )

            # Project Mamba output to hidden_dim
            self.mamba_proj = nn.Linear(feature_dim, hidden_dim)

            logging.info(f"TemporalScalePredictor: Using Mamba2 for temporal modeling")
        else:
            # GRU: Lightweight temporal modeling
            self.temporal_gru = nn.GRU(
                input_size=feature_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True  # Input: [B, T, feature_dim]
            )
            logging.info(f"TemporalScalePredictor: Using GRU for temporal modeling")

        # 3. Scale/Shift Heads
        self.scale_head = nn.Linear(hidden_dim, 1)
        self.shift_head = nn.Linear(hidden_dim, 1)

        # Parameter count
        total_params = sum(p.numel() for p in self.parameters())
        logging.info(f"TemporalScalePredictor: {total_params:,} parameters")
        logging.info(f"  Feature Net: {embed_dim} → {feature_dim}")
        if use_mamba:
            logging.info(f"  Mamba2: d_model={feature_dim}, expand=2")
            logging.info(f"  Projection: {feature_dim} → {hidden_dim}")
        else:
            logging.info(f"  GRU: input={feature_dim}, hidden={hidden_dim}, layers={num_layers}")
        logging.info(f"  Heads: hidden={hidden_dim} → 1 (scale), 1 (shift)")

    def forward(self, cls_tokens):
        """
        Args:
            cls_tokens: [B, T, embed_dim] - 2-layer averaged CLS tokens

        Returns:
            scale: [B, T] - positive scale factors
            shift: [B, T] - shift values
        """
        B, T, embed_dim = cls_tokens.shape

        # 1. Feature extraction
        features = self.feature_net(cls_tokens)  # [B, T, feature_dim]

        # 2. Temporal modeling
        if self.use_mamba:
            # Mamba2 path
            # MambaBlock expects [B, T, d_model]
            mamba_output = self.temporal_mamba(features)  # [B, T, feature_dim]
            # Project to hidden_dim
            hidden_states = self.mamba_proj(mamba_output)  # [B, T, hidden_dim]
        else:
            # GRU path
            # Output: [B, T, hidden_dim], hidden: [num_layers, B, hidden_dim]
            hidden_states, _ = self.temporal_gru(features)  # [B, T, hidden_dim]

        # 3. Predict scale and shift from hidden states
        scale_logits = self.scale_head(hidden_states).squeeze(-1)  # [B, T]
        shift_logits = self.shift_head(hidden_states).squeeze(-1)  # [B, T]

        # 4. Ensure positive scale with Softplus
        scale = F.softplus(scale_logits)  # [B, T]
        shift = shift_logits  # [B, T] - any value

        return scale, shift


# ==================== Gear5 Metric Head ====================

class Gear5MetricHead(nn.Module):
    """
    Unified Gear5 metric head with temporal scale prediction and importance mapping.

    Pipeline:
        1. Extract 2-layer CLS tokens
        2. Generate importance map from attention weights
        3. Predict scale/shift with GRU
        4. Return for application to final depth

    Note: This head does NOT modify DPT features. Scale/shift are applied
          AFTER output_conv2 in the training/inference loop.
    """
    def __init__(self, embed_dim=1024, feature_dim=256, hidden_dim=128, use_mamba=False):
        super().__init__()

        # Temporal scale predictor (GRU or Mamba2-based)
        self.temporal_scale_predictor = TemporalScalePredictor(
            embed_dim=embed_dim,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=1,
            use_mamba=use_mamba  # NEW: Support Mamba2 option
        )

        # Importance map generator
        self.importance_map_generator = ImportanceMapGenerator(num_layers=2)

        # Total parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(f"Gear5MetricHead: {trainable_params:,} / {total_params:,} trainable parameters")

    def forward(self, cls_tokens, attention_weights_list, patch_h, patch_w):
        """
        Args:
            cls_tokens: [B, T, embed_dim] - 2-layer averaged CLS tokens
            attention_weights_list: List of [B*T, num_heads, N+1, N+1] from 2 layers
            patch_h, patch_w: Spatial patch dimensions

        Returns:
            dict with:
                - scale: [B, T]
                - shift: [B, T]
                - importance_map: [B, T, patch_h, patch_w]
        """
        # 1. Predict scale and shift
        scale, shift = self.temporal_scale_predictor(cls_tokens)  # [B, T], [B, T]

        # 2. Generate importance map
        # Note: attention_weights_list is [B*T, ...], need to handle temporal dimension
        importance_map = self.importance_map_generator(
            attention_weights_list, patch_h, patch_w
        )  # [B*T, 1, patch_h, patch_w]

        # Reshape importance map to [B, T, patch_h, patch_w]
        BT = importance_map.shape[0]
        T = cls_tokens.shape[1]
        B = BT // T
        importance_map = importance_map.view(B, T, patch_h, patch_w)  # [B, T, patch_h, patch_w]

        return {
            'scale': scale,
            'shift': shift,
            'importance_map': importance_map
        }


# ==================== Bankai Mode: Unified Mamba for Temporal Depth + Metric ====================

class UnifiedMamba(nn.Module):
    """
    Unified Mamba for Gear5 Bankai mode.

    Processes spatial tokens (from DPT path_1) + CLS token through a single Mamba2,
    achieving both temporal depth consistency and metric scale/shift prediction.

    Architecture:
        Input: path_1 (148×148, C) + CLS token (1, C)
            ↓ Downsample path_1 (e.g., 0.1 → 14×14 = 196 tokens)
            ↓ Flatten → [h×w, C]
            ↓ CLS projection → [1, C]
            ↓ Concat → [h×w + 1, C] (CLS at END for aggregation)
            ↓ Mamba2 (per-frame processing with hidden state propagation)
            ↓ Split → Spatial [h×w, C], CLS [1, C]
            ↓
        Spatial output → final_layer + Upsample + Residual → Relative Depth enhancement
        CLS output → Scale/Shift MLP → Metric conversion

    Key Design Decisions:
        - CLS at END: Receives full spatial context for scale/shift prediction
          (Mamba2 is causal, so CLS does NOT affect spatial tokens)
        - Hidden state propagation: Ensures temporal consistency
        - Residual connection: Preserves original DPT quality
        - Compatible with FlashDepth MambaModel: blocks structure matches for weight loading

    Weight Loading:
        - blocks[0][0..3]: Loaded from FlashDepth pretrained Mamba (mamba.blocks.0.*)
        - final_layer: Loaded from FlashDepth pretrained Mamba (mamba.final_layer.*)
        - cls_proj, scale_head, shift_head: Random init (new components)
    """
    def __init__(self,
                 dpt_dim=256,
                 cls_embed_dim=1024,
                 num_layers=4,
                 downsample_factor=0.1,
                 d_state=256,  # Match FlashDepth default
                 d_conv=4,     # Match FlashDepth config
                 use_hydra=False):
        super().__init__()

        self.dpt_dim = dpt_dim
        self.cls_embed_dim = cls_embed_dim
        self.num_layers = num_layers
        self.downsample_factor = downsample_factor
        self.d_state = d_state
        self.d_conv = d_conv

        # Determine Mamba2 hyperparameters based on dpt_dim
        if dpt_dim == 64:  # ViT-S / Hybrid
            headdim = 32
            expand = 4
        else:  # ViT-L (dpt_dim=256)
            headdim = 64
            expand = 2

        self.headdim = headdim
        self.expand = expand

        # 1. CLS projection: embed_dim → dpt_dim (NEW - not from FlashDepth)
        self.cls_proj = nn.Sequential(
            nn.Linear(cls_embed_dim, dpt_dim),
            nn.ReLU(inplace=True)
        )

        # 2. Mamba2 blocks - MATCHING FlashDepth MambaModel structure for weight loading
        # FlashDepth: self.blocks = nn.ModuleList([nn.ModuleList([MambaBlock(...)])])
        # We use blocks[0] to match FlashDepth's mamba.blocks.0.*
        from .mamba import MambaBlock, InferenceParams

        self.blocks = nn.ModuleList([
            nn.ModuleList([
                MambaBlock(
                    d_model=dpt_dim,
                    layer_idx=layer_idx,
                    expand=expand,
                    d_state=d_state,
                    d_conv=d_conv,
                    headdim=headdim,
                    use_hydra=use_hydra
                ) for layer_idx in range(num_layers)
            ])
        ])

        # 3. final_layer - MATCHING FlashDepth MambaModel (mamba_type='add')
        # This is loaded from FlashDepth pretrained weights
        self.final_layer = nn.Sequential(
            nn.GELU(),
            nn.Linear(dpt_dim, dpt_dim)
        )
        # Zero-init for training stability (same as FlashDepth)
        nn.init.zeros_(self.final_layer[1].weight)
        nn.init.zeros_(self.final_layer[1].bias)

        # 4. Scale/Shift heads for metric depth (NEW - not from FlashDepth)
        self.scale_head = nn.Sequential(
            nn.Linear(dpt_dim, dpt_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dpt_dim // 2, 1)
        )
        self.shift_head = nn.Sequential(
            nn.Linear(dpt_dim, dpt_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dpt_dim // 2, 1)
        )

        # 5. Inference params for hidden state tracking
        self.max_seqlen = 60000
        self.max_batch_size = 32  # Will be updated during forward
        self.inference_params = None

        # Log parameter count
        total_params = sum(p.numel() for p in self.parameters())
        blocks_params = sum(p.numel() for p in self.blocks.parameters())
        final_layer_params = sum(p.numel() for p in self.final_layer.parameters())
        new_params = total_params - blocks_params - final_layer_params
        logging.info(f"UnifiedMamba (Bankai): {total_params:,} parameters")
        logging.info(f"  From FlashDepth (blocks + final_layer): {blocks_params + final_layer_params:,}")
        logging.info(f"  New components (cls_proj, scale/shift): {new_params:,}")
        logging.info(f"  Mamba2: d_model={dpt_dim}, d_state={d_state}, d_conv={d_conv}, expand={expand}, headdim={headdim}, layers={num_layers}")
    
    def start_new_sequence(self, batch_size=None):
        """Reset hidden states for new video sequence"""
        from .mamba import InferenceParams
        
        if batch_size is not None:
            self.max_batch_size = batch_size
        self.inference_params = InferenceParams(
            max_seqlen=self.max_seqlen, 
            max_batch_size=self.max_batch_size
        )
    
    def forward_single_frame(self, spatial_tokens, cls_token):
        """
        Process a single frame with hidden state propagation.

        Args:
            spatial_tokens: [B, h*w, dpt_dim] - Downsampled spatial features
            cls_token: [B, cls_embed_dim] - CLS token from DINOv2

        Returns:
            spatial_out: [B, h*w, dpt_dim] - Enhanced spatial features
            scale: [B, 1] - Scale factor
            shift: [B, 1] - Shift factor
        """
        B = spatial_tokens.shape[0]

        # 1. Project CLS token to dpt_dim
        cls_proj = self.cls_proj(cls_token)  # [B, dpt_dim]
        cls_proj = cls_proj.unsqueeze(1)  # [B, 1, dpt_dim]

        # 2. Concatenate spatial + CLS (CLS at END)
        x = torch.cat([spatial_tokens, cls_proj], dim=1)  # [B, h*w + 1, dpt_dim]

        # 3. Process through Mamba2 blocks (blocks[0] to match FlashDepth structure)
        for block in self.blocks[0]:
            x = block(x, inference_params=self.inference_params)

        # 4. Split spatial and CLS
        spatial_out = x[:, :-1, :]  # [B, h*w, dpt_dim]
        cls_out = x[:, -1, :]  # [B, dpt_dim]

        # 5. Apply final_layer to spatial features (loaded from FlashDepth)
        spatial_out = self.final_layer(spatial_out)  # [B, h*w, dpt_dim]

        # 6. Predict scale and shift from CLS
        scale_logits = self.scale_head(cls_out)  # [B, 1]
        shift_logits = self.shift_head(cls_out)  # [B, 1]

        # Ensure positive scale with Softplus
        scale = F.softplus(scale_logits)  # [B, 1]
        shift = shift_logits  # [B, 1]

        # Update sequence length offset
        if self.inference_params is not None:
            self.inference_params.seqlen_offset += x.shape[1]

        return spatial_out, scale, shift
    
    def forward(self, path_1, cls_tokens, input_shape):
        """
        Process entire video sequence through unified Mamba.
        
        Args:
            path_1: [B*T, dpt_dim, h, w] - DPT path_1 features
            cls_tokens: [B, T, cls_embed_dim] - CLS tokens for all frames
            input_shape: (B, T, C, H, W) - Original input shape
        
        Returns:
            dict with:
                - spatial_out: [B*T, dpt_dim, h, w] - Enhanced spatial features
                - scale: [B, T] - Scale factors for each frame
                - shift: [B, T] - Shift factors for each frame
        """
        B, T, C, H, W = input_shape
        BT, dpt_dim, h, w = path_1.shape
        assert BT == B * T, f"Expected {B*T}, got {BT}"
        
        # Start new sequence
        self.start_new_sequence(batch_size=B)
        
        # Store original path_1 for residual
        original_path_1 = path_1.clone()
        original_path_1 = original_path_1.view(B, T, dpt_dim, h, w)
        
        # Apply downsampling
        if self.downsample_factor != 1.0:
            h_down = int(h * self.downsample_factor)
            w_down = int(w * self.downsample_factor)
            path_1_down = F.adaptive_avg_pool2d(path_1, (h_down, w_down))
        else:
            h_down, w_down = h, w
            path_1_down = path_1
        
        # Reshape for per-frame processing
        path_1_down = path_1_down.view(B, T, dpt_dim, h_down, w_down)
        
        # Process each frame
        spatial_outs = []
        scales = []
        shifts = []
        
        for t in range(T):
            # Get frame features
            frame_spatial = path_1_down[:, t]  # [B, dpt_dim, h_down, w_down]
            frame_cls = cls_tokens[:, t]  # [B, cls_embed_dim]
            
            # Flatten spatial features
            frame_spatial_flat = frame_spatial.permute(0, 2, 3, 1).reshape(B, h_down * w_down, dpt_dim)
            
            # Process through unified Mamba
            spatial_out, scale, shift = self.forward_single_frame(frame_spatial_flat, frame_cls)
            
            spatial_outs.append(spatial_out)
            scales.append(scale)
            shifts.append(shift)
        
        # Stack outputs
        spatial_outs = torch.stack(spatial_outs, dim=1)  # [B, T, h_down*w_down, dpt_dim]
        scales = torch.cat(scales, dim=1)  # [B, T]
        shifts = torch.cat(shifts, dim=1)  # [B, T]
        
        # Reshape spatial outputs back to spatial format
        spatial_outs = spatial_outs.view(B * T, h_down * w_down, dpt_dim)
        spatial_outs = spatial_outs.permute(0, 2, 1).reshape(B * T, dpt_dim, h_down, w_down)
        
        # Upsample if downsampled
        if self.downsample_factor != 1.0:
            spatial_outs = F.interpolate(spatial_outs, (h, w), mode='bilinear', align_corners=True)
        
        # Add residual (1:1 ratio)
        spatial_outs = spatial_outs + original_path_1.view(B * T, dpt_dim, h, w)
        
        return {
            'spatial_out': spatial_outs,  # [B*T, dpt_dim, h, w]
            'scale': scales,  # [B, T]
            'shift': shifts  # [B, T]
        }


class BankaiMetricHead(nn.Module):
    """
    Unified Bankai metric head: combines temporal depth enhancement and metric prediction.
    
    This head replaces both the original FlashDepth Mamba (F-Mamba) and 
    Gear5's TemporalScalePredictor (T-Mamba) with a single unified Mamba.
    
    Architecture (Bankai):
        ┌─────────────────────────────────────────────────────────────────┐
        │                        DINOv2 (Frozen)                          │
        │  ViT-L: CLS tokens [17, 23] → avg → Fused CLS-L [1, 1024]      │
        │  ViT-S: CLS tokens [8, 11]  → avg → Fused CLS-S [1, 384]       │
        └─────────────────────────────────────────────────────────────────┘
                                      ↓
                [Hybrid only] CrossAttn(Q:CLS-S, KV:CLS-L)
                                      ↓
                      Fused CLS → Linear + ReLU → [1, C]
        
        ┌─────────────────────────────────────────────────────────────────┐
        │                         DPT (Frozen)                            │
        │  path_1 (148×148, C) → Downsample → Flatten → [h×w, C]         │
        └─────────────────────────────────────────────────────────────────┘
                                      ↓
                            Concat → [h×w + 1, C]
                            (Spatial tokens + CLS at END)
                                      ↓
        ┌─────────────────────────────────────────────────────────────────┐
        │                    Unified Mamba (Trainable)                    │
        │  - Per-frame processing + hidden state propagation              │
        │  - Temporal consistency via hidden state                        │
        └─────────────────────────────────────────────────────────────────┘
                                      ↓
                    ┌─────────────────┴─────────────────┐
                    ↓                                   ↓
            Spatial [h×w, C]                     CLS [1, C]
                    ↓                                   ↓
            Unflatten + Upsample                 MLP Head
            + original_path_1 (residual)              ↓
                    ↓                           Scale (Softplus)
            output_conv (trainable)             Shift (Clamped)
                    ↓
              Relative Depth
                    ↓
            Metric Depth = Scale × Relative + Shift
    
    Model Variants:
        - Large: dpt_dim=256, cls_embed_dim=1024, downsample=0.1
        - Small: dpt_dim=64, cls_embed_dim=384, downsample=0.05
        - Hybrid: dpt_dim=64, cls_embed_dim=384 (with CrossAttn), downsample=0.05
    """
    def __init__(self,
                 dpt_dim=256,
                 cls_embed_dim=1024,
                 num_mamba_layers=4,
                 downsample_factor=0.1,
                 use_hybrid_cls_fusion=False,
                 teacher_cls_dim=1024,
                 d_state=256,  # Must match FlashDepth for weight loading
                 d_conv=4):    # Must match FlashDepth for weight loading
        super().__init__()

        self.dpt_dim = dpt_dim
        self.cls_embed_dim = cls_embed_dim
        self.use_hybrid_cls_fusion = use_hybrid_cls_fusion
        self.teacher_cls_dim = teacher_cls_dim

        # Hybrid mode: CrossAttention for CLS fusion
        if use_hybrid_cls_fusion:
            self.cls_cross_attn = nn.MultiheadAttention(
                embed_dim=cls_embed_dim,  # Query dim (student)
                num_heads=4,
                kdim=teacher_cls_dim,  # Key dim (teacher)
                vdim=teacher_cls_dim,  # Value dim (teacher)
                batch_first=True
            )
            logging.info(f"BankaiMetricHead: CrossAttn for CLS fusion (Q:{cls_embed_dim}, KV:{teacher_cls_dim})")

        # Unified Mamba for temporal processing
        # Structure matches FlashDepth MambaModel for weight loading compatibility
        self.unified_mamba = UnifiedMamba(
            dpt_dim=dpt_dim,
            cls_embed_dim=cls_embed_dim,
            num_layers=num_mamba_layers,
            downsample_factor=downsample_factor,
            d_state=d_state,
            d_conv=d_conv
        )
        
        # Importance map generator (reuse from original Gear5)
        self.importance_map_generator = ImportanceMapGenerator(num_layers=2)
        
        # Log parameter count
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(f"BankaiMetricHead: {trainable_params:,} / {total_params:,} trainable parameters")
    
    def forward(self, 
                path_1,
                cls_tokens, 
                attention_weights_list, 
                input_shape,
                teacher_cls_tokens=None):
        """
        Args:
            path_1: [B*T, dpt_dim, h, w] - DPT path_1 features (BEFORE Mamba in original)
            cls_tokens: [B, T, cls_embed_dim] - Multi-layer averaged CLS tokens
            attention_weights_list: List of [B*T, num_heads, N+1, N+1] from 2 layers
            input_shape: (B, T, C, H, W) - Original input shape
            teacher_cls_tokens: [B, T, teacher_cls_dim] - Teacher CLS tokens (Hybrid only)
        
        Returns:
            dict with:
                - spatial_out: [B*T, dpt_dim, h, w] - Enhanced spatial features (for output_conv)
                - scale: [B, T] - Scale factors
                - shift: [B, T] - Shift factors
                - importance_map: [B, T, patch_h, patch_w]
        """
        B, T, C, H, W = input_shape
        patch_h, patch_w = H // 14, W // 14  # Assuming patch_size=14
        
        # Hybrid mode: Fuse CLS tokens via CrossAttention
        if self.use_hybrid_cls_fusion and teacher_cls_tokens is not None:
            # cls_tokens: [B, T, cls_embed_dim] as Query
            # teacher_cls_tokens: [B, T, teacher_cls_dim] as Key/Value
            BT_cls = B * T
            
            # Reshape for attention: [B*T, 1, dim]
            q = cls_tokens.reshape(BT_cls, 1, -1)  # [B*T, 1, cls_embed_dim]
            kv = teacher_cls_tokens.reshape(BT_cls, 1, -1)  # [B*T, 1, teacher_cls_dim]
            
            # Cross-attention
            fused_cls, _ = self.cls_cross_attn(q, kv, kv)  # [B*T, 1, cls_embed_dim]
            
            # Reshape back to [B, T, cls_embed_dim]
            cls_tokens = fused_cls.reshape(B, T, -1)
        
        # Process through unified Mamba
        mamba_outputs = self.unified_mamba(path_1, cls_tokens, input_shape)
        
        spatial_out = mamba_outputs['spatial_out']  # [B*T, dpt_dim, h, w]
        scale = mamba_outputs['scale']  # [B, T]
        shift = mamba_outputs['shift']  # [B, T]
        
        # Generate importance map
        importance_map = self.importance_map_generator(
            attention_weights_list, patch_h, patch_w
        )  # [B*T, 1, patch_h, patch_w]
        
        # Reshape importance map to [B, T, patch_h, patch_w]
        importance_map = importance_map.view(B, T, patch_h, patch_w)
        
        return {
            'spatial_out': spatial_out,
            'scale': scale,
            'shift': shift,
            'importance_map': importance_map
        }
