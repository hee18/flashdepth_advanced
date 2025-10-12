"""
Gear3 Modules: Feature-level Metric Injection via FiLM-style Modulation

This module implements the following architecture:
1. [REMOVED] ImportancePredictor: Now uses raw attention directly
2. ForegroundBackgroundNetworks: Generates FG/BG semantic features
3. ModulationNetworks: Generates gamma and beta for FiLM-style modulation
4. FeatureModulator: Applies hierarchical modulation to DPT features

Key Changes:
- Importance map = DINOv2 CLS→patch attention (averaged over heads)
- Register token (highest attention patch) removed via 3×3 inpainting
- Percentile normalization (1-99) to [0,1] for robustness
- FG/BG split uses mean (adaptive) instead of median (fixed 50:50)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import logging


def process_attention_to_importance(attention_weights, patch_h, patch_w, remove_outliers=True):
    """
    Convert raw attention weights to importance map.

    Steps:
    1. Extract CLS→patch attention
    2. Average over heads
    3. Remove register token (highest attention patch)
    4. Percentile normalization (1-99 percentile) to [0, 1]

    Args:
        attention_weights: [B, num_heads, num_patches+1, num_patches+1]
        patch_h, patch_w: Spatial dimensions
        remove_outliers: Whether to remove register token (default: True)

    Returns:
        importance_map: [B, 1, patch_h, patch_w] in range [0, 1]
    """
    B = attention_weights.shape[0]

    # Extract CLS→patch attention
    cls_to_patches = attention_weights[:, :, 0, 1:]  # [B, num_heads, num_patches]

    # Average over heads
    attn_scores = cls_to_patches.mean(dim=1)  # [B, num_patches]

    # Reshape to spatial
    attn_map = attn_scores.reshape(B, 1, patch_h, patch_w)  # [B, 1, patch_h, patch_w]

    if remove_outliers:
        # Remove register token (single highest attention patch)
        # Based on empirical observation: DINOv2 has 1 register token with extremely high attention
        for b in range(B):
            attn_2d = attn_map[b, 0]  # [patch_h, patch_w]

            # Find the patch with maximum attention (register token)
            max_val = attn_2d.max()
            outlier_mask = (attn_2d == max_val)  # Only the single highest patch

            # Inpaint with local average (3×3 box filter at patch level)
            # Replace register token with average of surrounding 8 patches
            kernel = torch.ones(1, 1, 3, 3, device=attn_map.device) / 9
            attn_smoothed = F.conv2d(
                attn_map[b:b+1], kernel, padding=1
            )
            attn_map[b, 0] = torch.where(
                outlier_mask,
                attn_smoothed[0, 0],
                attn_map[b, 0]
            )

    # Percentile-based normalization to [0, 1] (1-99 percentile)
    # More robust than min-max: reduces sensitivity to remaining outliers
    for b in range(B):
        attn_flat = attn_map[b].flatten()
        attn_p1 = torch.quantile(attn_flat, 0.01)   # 1st percentile
        attn_p99 = torch.quantile(attn_flat, 0.99)  # 99th percentile

        # Normalize to [0, 1] and clip
        attn_map[b] = (attn_map[b] - attn_p1) / (attn_p99 - attn_p1 + 1e-8)
        attn_map[b] = torch.clamp(attn_map[b], 0.0, 1.0)  # Clip outliers to [0, 1]

    return attn_map


class ForegroundBackgroundNetworks(nn.Module):
    """
    Generates foreground and background semantic features from patch tokens.

    Option 3: Attention-based Pooling
    - Uses CLS→patch attention to distinguish important (FG) vs context (BG) regions
    - Top attention patches → FG network
    - Bottom attention patches → BG network

    Input: Patch tokens [B, num_patches, embed_dim], Attention weights
    Output: FG features [B, feature_dim], BG features [B, feature_dim]
    """
    def __init__(self, embed_dim=1024, feature_dim=256):
        super().__init__()

        # Foreground network (focus on salient objects)
        self.fg_net = nn.Sequential(
            nn.Linear(embed_dim, feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True)
        )

        # Background network (focus on context)
        self.bg_net = nn.Sequential(
            nn.Linear(embed_dim, feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True)
        )

        logging.info(f"FG/BG Networks (Attention-based Pooling): {embed_dim} -> {feature_dim}")

    def forward(self, patch_tokens, attention_weights, importance_map):
        """
        Args:
            patch_tokens: [B, num_patches, embed_dim]
            attention_weights: [B, num_heads, num_patches+1, num_patches+1]
                              (from last DINOv2 block)
            importance_map: [B, 1, patch_h, patch_w] - Processed attention
                           (outliers removed, min-max normalized)

        Returns:
            fg_features: [B, feature_dim] - Weighted by high attention
            bg_features: [B, feature_dim] - Weighted by low attention
        """
        B, num_patches, embed_dim = patch_tokens.shape

        # Use processed importance map (outliers removed, normalized [0,1])
        # Flatten spatial dimensions to match patch_tokens shape
        attn_scores = importance_map.flatten(2).squeeze(1)  # [B, num_patches]

        # Compute mean for FG/BG split from CLEANED attention (adaptive, not fixed 50:50)
        # Register patches are already removed, so mean is not distorted by outliers
        attn_mean = attn_scores.mean(dim=1, keepdim=True)  # [B, 1]

        # Create masks (top attention = FG, bottom attention = BG)
        fg_mask = (attn_scores > attn_mean).float()  # [B, num_patches]
        bg_mask = (attn_scores <= attn_mean).float()  # [B, num_patches]

        # Weighted pooling (attention-weighted average)
        fg_weights = attn_scores * fg_mask  # [B, num_patches]
        bg_weights = (1.0 - attn_scores) * bg_mask  # Inverse for BG

        # Normalize weights
        fg_weights = fg_weights / (fg_weights.sum(dim=1, keepdim=True) + 1e-8)  # [B, num_patches]
        bg_weights = bg_weights / (bg_weights.sum(dim=1, keepdim=True) + 1e-8)

        # Weighted sum
        fg_pooled = (patch_tokens * fg_weights.unsqueeze(-1)).sum(dim=1)  # [B, embed_dim]
        bg_pooled = (patch_tokens * bg_weights.unsqueeze(-1)).sum(dim=1)  # [B, embed_dim]

        # Pass through networks
        fg_features = self.fg_net(fg_pooled)  # [B, feature_dim]
        bg_features = self.bg_net(bg_pooled)  # [B, feature_dim]

        return fg_features, bg_features


class ModulationNetworks(nn.Module):
    """
    Generates gamma and beta for FiLM-style modulation for path_1 (Layer 23 features).

    Input: FG/BG features [B, feature_dim]
    Output: Gamma [B, dpt_dim], Beta [B, dpt_dim] for FG and BG separately
    """
    def __init__(self, feature_dim=256, dpt_dim=256):
        super().__init__()
        self.dpt_dim = dpt_dim

        # Single modulation network for path_1 (Layer 23)
        # FG modulation: Layer 23 features → gamma, beta
        self.fg_modulation = nn.Sequential(
            nn.Linear(feature_dim, dpt_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(dpt_dim * 2, dpt_dim * 2)  # First half: gamma, second half: beta
        )

        # BG modulation: Layer 23 features → gamma, beta
        self.bg_modulation = nn.Sequential(
            nn.Linear(feature_dim, dpt_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(dpt_dim * 2, dpt_dim * 2)  # First half: gamma, second half: beta
        )

        logging.info("Modulation Networks initialized for path_1 (Layer 23) only")

    def forward(self, fg_features, bg_features):
        """
        Args:
            fg_features: [B, feature_dim]
            bg_features: [B, feature_dim]

        Returns:
            fg_gamma: [B, dpt_dim]
            fg_beta: [B, dpt_dim]
            bg_gamma: [B, dpt_dim]
            bg_beta: [B, dpt_dim]
        """
        # FG modulation
        fg_params = self.fg_modulation(fg_features)  # [B, dpt_dim * 2]
        fg_gamma = fg_params[:, :self.dpt_dim]
        fg_beta = fg_params[:, self.dpt_dim:]

        # BG modulation
        bg_params = self.bg_modulation(bg_features)  # [B, dpt_dim * 2]
        bg_gamma = bg_params[:, :self.dpt_dim]
        bg_beta = bg_params[:, self.dpt_dim:]

        return fg_gamma, fg_beta, bg_gamma, bg_beta


class FeatureModulator(nn.Module):
    """
    Applies hierarchical FiLM-style modulation to DPT features.

    Modulation formula:
        gamma[x,y] = importance[x,y] * fg_gamma + (1 - importance[x,y]) * bg_gamma
        beta[x,y] = importance[x,y] * fg_beta + (1 - importance[x,y]) * bg_beta
        modulated[x,y] = gamma[x,y] ⊙ feature[x,y] + beta[x,y]
    """
    def __init__(self):
        super().__init__()

    def forward(self, features, importance_map, fg_gamma, fg_beta, bg_gamma, bg_beta):
        """
        Args:
            features: [B, C, H, W] DPT layer features
            importance_map: [B, 1, H', W'] (will be resized to match features)
            fg_gamma, fg_beta: [B, C] foreground modulation params
            bg_gamma, bg_beta: [B, C] background modulation params

        Returns:
            modulated_features: [B, C, H, W]
        """
        B, C, H, W = features.shape

        # Resize importance map to match feature spatial dimensions
        if importance_map.shape[2:] != (H, W):
            importance_map = F.interpolate(
                importance_map, size=(H, W), mode='bilinear', align_corners=True
            )  # [B, 1, H, W]

        # Expand gamma and beta to spatial dimensions
        fg_gamma = fg_gamma.view(B, C, 1, 1)  # [B, C, 1, 1]
        fg_beta = fg_beta.view(B, C, 1, 1)
        bg_gamma = bg_gamma.view(B, C, 1, 1)
        bg_beta = bg_beta.view(B, C, 1, 1)

        # Memory-efficient computation using torch.lerp (linear interpolation)
        # gamma = (1 - importance_map) * bg_gamma + importance_map * fg_gamma
        # Ensure importance_map matches dtype of gamma/beta (for BFloat16 compatibility)
        importance_map = importance_map.to(bg_gamma.dtype)
        gamma = torch.lerp(bg_gamma, fg_gamma, importance_map)  # [B, C, H, W]
        beta = torch.lerp(bg_beta, fg_beta, importance_map)  # [B, C, H, W]

        # Apply FiLM modulation
        modulated_features = gamma * features + beta

        return modulated_features


class Gear3MetricHead(nn.Module):
    """
    Complete Gear3 metric depth head combining all modules.

    Architecture:
        1. Attention-based importance map: DINOv2 attention -> importance map (no learnable params)
        2. FG/BG Networks: patch tokens -> FG/BG features
        3. Modulation Networks: FG/BG features -> gamma/beta for path_1
        4. Feature Modulator: Apply modulation to path_1 (Layer 23 features)
    """
    def __init__(self, embed_dim=1024, dpt_dim=256, num_heads=16):
        super().__init__()

        # No ImportancePredictor - use raw attention directly
        self.fg_bg_networks = ForegroundBackgroundNetworks(
            embed_dim=embed_dim, feature_dim=256
        )
        self.modulation_networks = ModulationNetworks(
            feature_dim=256, dpt_dim=dpt_dim
        )
        self.feature_modulator = FeatureModulator()

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(f"Gear3 Metric Head: {trainable_params:,} / {total_params:,} trainable parameters")

    def forward(self, patch_tokens, attention_weights, dpt_features, patch_h, patch_w):
        """
        Args:
            patch_tokens: [B, num_patches, embed_dim] from Layer 23
            attention_weights: [B, num_heads, num_patches+1, num_patches+1] from Layer 23
            dpt_features: List of [B, dpt_dim, H, W] for 4 DPT layers
            patch_h, patch_w: Spatial dimensions

        Returns:
            path_1_modulated: [B, dpt_dim, H, W] modulated path_1 features
            importance_map: [B, 1, patch_h, patch_w] for visualization
            fg_features: [B, 256] foreground features (for ContrastiveFGBGLoss)
            bg_features: [B, 256] background features (for ContrastiveFGBGLoss)
        """
        # 1. Convert raw attention to importance map (no learnable params)
        # Register patches removed, min-max normalized to [0,1]
        importance_map = process_attention_to_importance(attention_weights, patch_h, patch_w)

        # 2. Generate FG/BG features (from Layer 23 patch tokens)
        # Uses PROCESSED importance map (outliers removed) to separate FG vs BG regions
        fg_features, bg_features = self.fg_bg_networks(patch_tokens, attention_weights, importance_map)

        # 3. Get modulation parameters for path_1 (Layer 23 → path_1)
        fg_gamma, fg_beta, bg_gamma, bg_beta = self.modulation_networks(
            fg_features, bg_features
        )

        # 4. Modulate ONLY path_1 (last element, from Layer 23)
        # Other paths (from Layer 4, 11, 17) are NOT modulated - semantic mismatch
        path_1 = dpt_features[-1]
        path_1_modulated = self.feature_modulator(
            path_1, importance_map, fg_gamma, fg_beta, bg_gamma, bg_beta
        )

        return path_1_modulated, importance_map, fg_features, bg_features
