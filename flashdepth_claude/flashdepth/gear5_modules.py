"""
Gear5 Modules: Global Scale Prediction + FG-only Modulation

This module implements a two-stage metric depth training approach:

Stage 1: Global Scale & Shift Prediction
- Input: Multi-layer CLS tokens [Layers 4, 11, 17, 23]
- Output: Global scale and shift parameters
- Applied to: DPT output features (before Mamba)
- Formula: DPT_global = DPT × scale + shift

Stage 2: Foreground-only Modulation
- Input: Globally-modulated DPT features + Multi-layer attention
- Output: FG-only modulated features
- Applied to: Foreground pixels only (Background keeps global modulation)
- Formula: DPT_fg = gamma × DPT_global_fg + beta

Architecture:
    ViT → DPT → Global GSP → Mamba → Final Head (Step 1)
    ViT → DPT → Global GSP → FG Modulation → Mamba → Final Head (Step 2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# Import from gear3_upgrade_modules
from flashdepth.gear3_upgrade_modules import (
    MultiLayerAttentionFusion,
    process_attention_to_importance
)


# ==================== Step 1: Global Scale Predictor ====================

class GlobalScalePredictorMultiLayer(nn.Module):
    """
    Predict global scale and shift from multi-layer CLS tokens.

    This module extracts CLS tokens from multiple ViT layers and fuses them
    to predict scene-level scale and shift parameters for metric depth conversion.

    Input: CLS tokens from layers [4, 11, 17, 23] (encoder output layers)
    Output: scale (positive), shift (any value)

    Architecture:
        Concat [CLS_4, CLS_11, CLS_17, CLS_23] → 4*embed_dim
        ↓
        MLP: 4*embed_dim → 1024 → 256 → 2
        ↓
        [scale (Softplus), shift]
    """
    def __init__(self, embed_dim=1024, num_layers=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # Concatenate all CLS tokens: 4 × embed_dim → 4096 (for ViT-L)
        input_dim = embed_dim * num_layers

        # MLP to predict scale and shift
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)  # Output: [scale, shift]
        )

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        logging.info(f"GlobalScalePredictorMultiLayer: {total_params:,} parameters")
        logging.info(f"  Input: {num_layers} CLS tokens × {embed_dim} = {input_dim}")
        logging.info(f"  Output: scale (positive), shift (any)")

    def forward(self, cls_tokens_list):
        """
        Args:
            cls_tokens_list: List of [B, embed_dim] CLS tokens
                            [CLS_4, CLS_11, CLS_17, CLS_23]

        Returns:
            scale: [B] - positive scale factor
            shift: [B] - shift value (any)
        """
        # Concatenate all CLS tokens: [B, 4*embed_dim]
        cls_concat = torch.cat(cls_tokens_list, dim=-1)

        # Predict scale and shift
        params = self.predictor(cls_concat)  # [B, 2]

        # Ensure positive scale with Softplus
        scale = F.softplus(params[:, 0])  # [B]
        shift = params[:, 1]  # [B]

        return scale, shift


# ==================== Step 2: Foreground-only Modulation ====================

class ForegroundFeatureNetwork(nn.Module):
    """
    Extract foreground-only features from patch tokens using FG mask.

    Unlike Gear3 which extracts both FG and BG features, this only extracts FG.
    Uses weighted pooling with importance map for soft attention weighting.
    """
    def __init__(self, embed_dim=1024, feature_dim=256):
        super().__init__()

        # FG feature network (same as Gear3)
        self.fg_net = nn.Sequential(
            nn.Linear(embed_dim, feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True)
        )

        logging.info(f"ForegroundFeatureNetwork: {embed_dim} -> {feature_dim}")

    def forward(self, patch_tokens, fg_mask, importance_map):
        """
        Args:
            patch_tokens: [B, num_patches, embed_dim]
            fg_mask: [B, 1, patch_h, patch_w] - binary FG mask
            importance_map: [B, 1, patch_h, patch_w] - soft importance scores

        Returns:
            fg_features: [B, feature_dim]
        """
        B, num_patches, embed_dim = patch_tokens.shape

        # Flatten masks
        fg_mask_flat = fg_mask.flatten(2).squeeze(1)  # [B, mask_patches]
        importance_flat = importance_map.flatten(2).squeeze(1)  # [B, map_patches]

        # Handle dimension mismatch (resize to num_patches)
        if fg_mask_flat.shape[1] != num_patches:
            fg_mask_flat = F.interpolate(
                fg_mask_flat.unsqueeze(1), size=num_patches,
                mode='linear', align_corners=True
            ).squeeze(1)

        if importance_flat.shape[1] != num_patches:
            importance_flat = F.interpolate(
                importance_flat.unsqueeze(1), size=num_patches,
                mode='linear', align_corners=True
            ).squeeze(1)

        # Soft weighting: importance × FG mask
        fg_weights = importance_flat * fg_mask_flat  # [B, num_patches]

        # Normalize weights
        fg_weights = fg_weights / (fg_weights.sum(dim=1, keepdim=True) + 1e-8)

        # Weighted pooling
        fg_pooled = (patch_tokens * fg_weights.unsqueeze(-1)).sum(dim=1)  # [B, embed_dim]

        # Pass through network
        fg_features = self.fg_net(fg_pooled)  # [B, feature_dim]

        return fg_features


class ForegroundModulationNetwork(nn.Module):
    """
    Generate FiLM parameters (gamma, beta) for FG-only modulation.

    Input: FG features [B, feature_dim]
    Output: gamma [B, dpt_dim], beta [B, dpt_dim]
    """
    def __init__(self, feature_dim=256, dpt_dim=256):
        super().__init__()
        self.dpt_dim = dpt_dim

        # FG modulation network
        self.fg_modulation = nn.Sequential(
            nn.Linear(feature_dim, dpt_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(dpt_dim * 2, dpt_dim * 2)  # [gamma, beta]
        )

        logging.info(f"ForegroundModulationNetwork: {feature_dim} -> gamma/beta ({dpt_dim})")

    def forward(self, fg_features):
        """
        Args:
            fg_features: [B, feature_dim]

        Returns:
            fg_gamma: [B, dpt_dim]
            fg_beta: [B, dpt_dim]
        """
        # Generate modulation parameters
        fg_params = self.fg_modulation(fg_features)  # [B, dpt_dim * 2]

        # Split into gamma and beta
        fg_gamma = fg_params[:, :self.dpt_dim]
        fg_beta = fg_params[:, self.dpt_dim:]

        return fg_gamma, fg_beta


class ForegroundOnlyModulator(nn.Module):
    """
    Apply FiLM modulation to foreground pixels only.

    Modulation formula:
        For FG pixels: modulated = gamma × feature + beta
        For BG pixels: modulated = feature (no change)

    This is different from Gear3 which blends FG/BG modulation.
    """
    def __init__(self):
        super().__init__()

    def forward(self, features, fg_mask, fg_gamma, fg_beta):
        """
        Args:
            features: [B, C, H, W] - DPT features (already globally-modulated)
            fg_mask: [B, 1, patch_h, patch_w] - binary FG mask
            fg_gamma: [B, C] - FG modulation scale
            fg_beta: [B, C] - FG modulation shift

        Returns:
            modulated_features: [B, C, H, W]
        """
        B, C, H, W = features.shape

        # Resize FG mask to match feature spatial dimensions
        if fg_mask.shape[2:] != (H, W):
            fg_mask = F.interpolate(
                fg_mask, size=(H, W), mode='bilinear', align_corners=True
            )  # [B, 1, H, W]

        # Expand gamma and beta to spatial dimensions
        fg_gamma = fg_gamma.view(B, C, 1, 1)  # [B, C, 1, 1]
        fg_beta = fg_beta.view(B, C, 1, 1)

        # Apply FiLM to FG pixels only
        # Ensure fg_mask matches dtype for BFloat16 compatibility
        fg_mask = fg_mask.to(features.dtype)

        # FG modulation: gamma × feature + beta
        fg_modulated = fg_gamma * features + fg_beta

        # Blend: FG modulation for FG pixels, original for BG pixels
        # modulated = fg_mask × (gamma × feature + beta) + (1 - fg_mask) × feature
        modulated_features = fg_mask * fg_modulated + (1.0 - fg_mask) * features

        return modulated_features


class ForegroundOnlyModulationHead(nn.Module):
    """
    Foreground-only modulation head for Step 2.

    Pipeline:
        1. Multi-layer attention fusion → importance map
        2. FG mask generation (mean threshold)
        3. FG feature extraction (weighted pooling)
        4. FG modulation parameters (gamma, beta)
        5. Apply FiLM to FG pixels only

    Key difference from Gear3:
        - Only FG modulation (no BG modulation)
        - BG pixels keep globally-modulated values
    """
    def __init__(self, embed_dim=1024, dpt_dim=256):
        super().__init__()

        self.embed_dim = embed_dim

        # Multi-layer attention fusion (reuse from Gear3)
        self.multi_layer_fusion = MultiLayerAttentionFusion(num_layers=4)

        # FG feature extraction
        self.fg_feature_network = ForegroundFeatureNetwork(
            embed_dim=embed_dim, feature_dim=256
        )

        # FG modulation network
        self.fg_modulation_network = ForegroundModulationNetwork(
            feature_dim=256, dpt_dim=dpt_dim
        )

        # FG-only modulator
        self.fg_modulator = ForegroundOnlyModulator()

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(f"ForegroundOnlyModulationHead: {trainable_params:,} / {total_params:,} trainable parameters")

    def forward(self, patch_tokens, attention_weights_multi_layer,
                dpt_features_global, patch_h, patch_w):
        """
        Args:
            patch_tokens: [B, num_patches+1, embed_dim] from Layer 23 (includes CLS)
            attention_weights_multi_layer: List of [B, num_heads, N+1, N+1]
                                          from layers [4, 11, 17, 23] (same as CLS layers)
            dpt_features_global: [B, dpt_dim, H, W] - globally-modulated DPT features
            patch_h, patch_w: Spatial dimensions (e.g., 37×37 for 518×518)

        Returns:
            path_1_fg_modulated: [B, dpt_dim, H, W]
            importance_map: [B, 1, patch_h, patch_w]
            fg_features: [B, 256]
            fg_mask: [B, 1, patch_h, patch_w]
        """
        # Remove CLS token
        patch_tokens_only = patch_tokens[:, 1:, :]  # [B, num_patches, embed_dim]

        # Step 1: Multi-layer attention fusion → importance map
        importance_map = self.multi_layer_fusion(
            attention_weights_multi_layer, patch_h, patch_w
        )  # [B, 1, patch_h, patch_w]

        # Step 2: Generate FG mask (mean threshold)
        importance_flat = importance_map.flatten(2).squeeze(1)  # [B, num_patches]
        threshold = importance_flat.mean(dim=1, keepdim=True)  # Per-sample adaptive
        fg_mask = (importance_flat > threshold).float().reshape(importance_map.shape)

        # Step 3: Extract FG features
        fg_features = self.fg_feature_network(
            patch_tokens_only, fg_mask, importance_map
        )  # [B, 256]

        # Step 4: Generate FG modulation parameters
        fg_gamma, fg_beta = self.fg_modulation_network(fg_features)  # [B, dpt_dim]

        # Step 5: Apply FG-only modulation
        path_1_fg_modulated = self.fg_modulator(
            dpt_features_global, fg_mask, fg_gamma, fg_beta
        )  # [B, dpt_dim, H, W]

        return path_1_fg_modulated, importance_map, fg_features, fg_mask


# ==================== Combined Head for Step 2 ====================

class Gear5MetricHead(nn.Module):
    """
    Combined Gear5 head with both Global GSP and FG modulation.

    Used in Step 2 training where:
        - Global GSP is frozen (from Step 1)
        - FG modulation is trainable
        - Mamba + Final head are trainable (from Step 1)

    Pipeline:
        ViT → DPT → Global GSP (frozen) → FG Modulation (trainable) → Mamba → Final
    """
    def __init__(self, embed_dim=1024, dpt_dim=256):
        super().__init__()

        # Global scale predictor (will be frozen in Step 2)
        self.global_gsp = GlobalScalePredictorMultiLayer(
            embed_dim=embed_dim, num_layers=4
        )

        # FG-only modulation (trainable in Step 2)
        self.fg_modulation_head = ForegroundOnlyModulationHead(
            embed_dim=embed_dim, dpt_dim=dpt_dim
        )

        # Count parameters
        gsp_params = sum(p.numel() for p in self.global_gsp.parameters())
        fg_params = sum(p.numel() for p in self.fg_modulation_head.parameters())
        total_params = gsp_params + fg_params

        logging.info(f"Gear5MetricHead: {total_params:,} total parameters")
        logging.info(f"  Global GSP: {gsp_params:,} (frozen in Step 2)")
        logging.info(f"  FG Modulation: {fg_params:,} (trainable in Step 2)")

    def forward(self, cls_tokens_list, patch_tokens, attention_weights_multi_layer,
                dpt_features, patch_h, patch_w, step=2):
        """
        Args:
            cls_tokens_list: List of [B, embed_dim] CLS tokens [Layer 4, 11, 17, 23]
            patch_tokens: [B, num_patches+1, embed_dim] from Layer 23
            attention_weights_multi_layer: List from layers [4, 11, 17, 23] (same as CLS)
            dpt_features: [B, dpt_dim, H, W] - original DPT path_1 features
            patch_h, patch_w: Spatial dimensions
            step: 1 (global only) or 2 (global + FG)

        Returns:
            modulated_features: [B, dpt_dim, H, W]
            scale: [B] (for monitoring)
            shift: [B] (for monitoring)
            importance_map: [B, 1, patch_h, patch_w] (if step==2)
            fg_features: [B, 256] (if step==2)
            fg_mask: [B, 1, patch_h, patch_w] (if step==2)
        """
        B = dpt_features.shape[0]

        # Step 1: Global scale prediction
        scale, shift = self.global_gsp(cls_tokens_list)  # [B], [B]

        # Apply global modulation to DPT features
        dpt_features_global = dpt_features * scale.view(B, 1, 1, 1) + shift.view(B, 1, 1, 1)

        if step == 1:
            # Step 1 training: only global modulation
            return {
                'modulated_features': dpt_features_global,
                'scale': scale,
                'shift': shift
            }

        elif step == 2:
            # Step 2 training: global + FG modulation
            path_1_fg_modulated, importance_map, fg_features, fg_mask = \
                self.fg_modulation_head(
                    patch_tokens, attention_weights_multi_layer,
                    dpt_features_global, patch_h, patch_w
                )

            return {
                'modulated_features': path_1_fg_modulated,
                'scale': scale,
                'shift': shift,
                'importance_map': importance_map,
                'fg_features': fg_features,
                'fg_mask': fg_mask
            }

        else:
            raise ValueError(f"Invalid step: {step}. Must be 1 or 2.")
