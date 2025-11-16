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
