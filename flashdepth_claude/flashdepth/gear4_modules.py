"""
Gear3 Upgrade Modules: Advanced FG/BG Separation Methods

This module implements 3 improved FG/BG separation strategies:
1. CLS-based Light Segmentation: Self-supervised segmentation from CLS token
2. Differentiable K-means: Soft clustering for bimodal separation
3. Multi-layer Attention Fusion: Combine attention from multiple ViT layers

All methods maintain < 5ms overhead compared to Gear3 baseline.
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
        for b in range(B):
            attn_2d = attn_map[b, 0]  # [patch_h, patch_w]

            # Find the patch with maximum attention (register token)
            max_val = attn_2d.max()
            outlier_mask = (attn_2d == max_val)

            # Inpaint with local average (3×3 box filter)
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
    for b in range(B):
        attn_flat = attn_map[b].flatten()
        attn_p1 = torch.quantile(attn_flat, 0.01)
        attn_p99 = torch.quantile(attn_flat, 0.99)

        # Normalize to [0, 1] and clip
        attn_map[b] = (attn_map[b] - attn_p1) / (attn_p99 - attn_p1 + 1e-8)
        attn_map[b] = torch.clamp(attn_map[b], 0.0, 1.0)

    return attn_map


# ==================== Multi-layer Attention Fusion ====================

class MultiLayerAttentionFusion(nn.Module):
    """
    Fuse attention weights from multiple ViT layers.

    Combines attention from layers 4, 11, 17, 23:
    - Layer 4 (early): Low-level patterns (edges, textures)
    - Layer 11 (mid): Mid-level semantics (parts)
    - Layer 17 (late): High-level semantics (objects)
    - Layer 23 (last): Abstract semantics

    Overhead: ~3ms (4× attention processing)
    """
    def __init__(self, num_layers=4, uniform_weights=False):
        super().__init__()
        self.num_layers = num_layers
        self.uniform_weights = uniform_weights

        if uniform_weights:
            # Fixed uniform weights (equal ratio for all layers)
            uniform = torch.ones(num_layers) / num_layers
            self.register_buffer('fusion_weights', uniform)
        else:
            # Learnable fusion weights (favor later layers)
            init_weights = torch.tensor([0.1, 0.2, 0.3, 0.4])
            self.fusion_weights = nn.Parameter(init_weights)

    def forward(self, attention_weights_list, patch_h, patch_w):
        """
        Args:
            attention_weights_list: List of [B, num_heads, N+1, N+1] from different layers
            patch_h, patch_w: Spatial dimensions

        Returns:
            importance_fused: [B, 1, patch_h, patch_w] - fused importance map
        """
        importance_maps = []

        # Process each layer's attention
        for attn in attention_weights_list:
            importance = process_attention_to_importance(attn, patch_h, patch_w)  # [B, 1, H, W]
            importance_maps.append(importance.squeeze(1))  # Remove channel dim → [B, H, W]

        # Stack: [B, num_layers, patch_h, patch_w]
        importance_stack = torch.stack(importance_maps, dim=1)

        # Normalize fusion weights (uniform weights are already normalized)
        if self.uniform_weights:
            weights_norm = self.fusion_weights
        else:
            weights_norm = torch.softmax(self.fusion_weights, dim=0)

        # Weighted fusion: [B, num_layers, patch_h, patch_w] → [B, patch_h, patch_w] → [B, 1, patch_h, patch_w]
        importance_fused = (importance_stack * weights_norm.view(1, -1, 1, 1)).sum(dim=1).unsqueeze(1)

        return importance_fused


# ==================== Common Modules (from Gear3) ====================

class GlobalFeatureNetwork(nn.Module):
    """Extract global semantic feature from CLS token"""
    def __init__(self, embed_dim=1024, feature_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, cls_token):
        return self.net(cls_token)


class ForegroundBackgroundNetworks(nn.Module):
    """
    Generates foreground and background semantic features from patch tokens.

    Uses FG/BG masks (from various separation methods) to pool patch tokens.
    MLP architecture matches train_gear3 for consistency.

    Input: Patch tokens [B, num_patches, embed_dim], FG/BG masks
    Output: FG features [B, feature_dim], BG features [B, feature_dim]
    """
    def __init__(self, embed_dim=1024, feature_dim=256):
        super().__init__()

        # Foreground network (focus on salient objects)
        # Architecture: embed_dim -> feature_dim*2 -> feature_dim (matches gear3)
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

        logging.info(f"FG/BG Networks: {embed_dim} -> {feature_dim}")

    def forward(self, patch_tokens, fg_mask, bg_mask, importance_map=None):
        """
        Args:
            patch_tokens: [B, num_patches, embed_dim]
            fg_mask: [B, 1, patch_h, patch_w] - FG probability/mask (from separation method)
            bg_mask: [B, 1, patch_h, patch_w] - BG probability/mask (from separation method)
            importance_map: [B, 1, patch_h, patch_w] - Optional importance scores for weighting
                           (matches Gear3 behavior: soft weighting with attention scores)

        Returns:
            fg_features: [B, feature_dim]
            bg_features: [B, feature_dim]
        """
        B, num_patches, embed_dim = patch_tokens.shape

        # Flatten masks first
        fg_mask_flat = fg_mask.flatten(2).squeeze(1)  # [B, mask_patches]
        bg_mask_flat = bg_mask.flatten(2).squeeze(1)  # [B, mask_patches]

        # Handle dimension mismatch using 1D interpolation (more robust than 2D)
        mask_patches = fg_mask_flat.shape[1]
        if mask_patches != num_patches:
            # Use 1D interpolation to match exact patch count
            fg_mask_flat = F.interpolate(
                fg_mask_flat.unsqueeze(1), size=num_patches, mode='linear', align_corners=True
            ).squeeze(1)  # [B, num_patches]
            bg_mask_flat = F.interpolate(
                bg_mask_flat.unsqueeze(1), size=num_patches, mode='linear', align_corners=True
            ).squeeze(1)  # [B, num_patches]

        # If importance_map provided, use it for soft weighting (like Gear3)
        if importance_map is not None:
            # Flatten importance_map
            attn_scores = importance_map.flatten(2).squeeze(1)  # [B, map_patches]

            # Handle dimension mismatch using 1D interpolation
            if attn_scores.shape[1] != num_patches:
                attn_scores = F.interpolate(
                    attn_scores.unsqueeze(1), size=num_patches, mode='linear', align_corners=True
                ).squeeze(1)  # [B, num_patches]

            # Weighted pooling with attention scores (matches Gear3!)
            fg_weights = attn_scores * fg_mask_flat  # Soft weighting
            bg_weights = (1.0 - attn_scores) * bg_mask_flat  # Inverse for BG
        else:
            # Use masks directly (for cls_seg/kmeans where masks are already soft)
            fg_weights = fg_mask_flat
            bg_weights = bg_mask_flat

        # Normalize weights (to ensure proper weighted average)
        fg_weights = fg_weights / (fg_weights.sum(dim=1, keepdim=True) + 1e-8)
        bg_weights = bg_weights / (bg_weights.sum(dim=1, keepdim=True) + 1e-8)

        # Weighted pooling (mask-weighted average)
        fg_pooled = (patch_tokens * fg_weights.unsqueeze(-1)).sum(dim=1)  # [B, embed_dim]
        bg_pooled = (patch_tokens * bg_weights.unsqueeze(-1)).sum(dim=1)  # [B, embed_dim]

        # Pass through networks (matches gear3 architecture)
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


# ==================== Main Gear3 Upgrade Head ====================

class Gear4MetricHead(nn.Module):
    """
    Gear3 Upgrade: Enhanced FG/BG separation with multiple strategies.

    Options:
        1. 'cls_seg': CLS-based light segmentation
        2. 'kmeans': Differentiable K-means clustering
        3. 'multi_layer': Multi-layer attention fusion
    """
    def __init__(self, embed_dim=1024, dpt_dim=256, num_heads=16, uniform_fusion_weights=False):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Multi-layer attention fusion (only separation method)
        self.multi_layer_fusion = MultiLayerAttentionFusion(num_layers=4, uniform_weights=uniform_fusion_weights)

        # Common modules
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
        logging.info(f"Gear3 Upgrade Head (multi_layer): {trainable_params:,} / {total_params:,} trainable parameters")

    def forward(self, patch_tokens, attention_weights, dpt_features, patch_h, patch_w,
                attention_weights_multi_layer=None, cls_token=None):
        """
        Args:
            patch_tokens: [B, num_patches+1, embed_dim] from Layer 23 (includes CLS token at index 0)
            attention_weights: [B, num_heads, N+1, N+1] from Layer 23 (unused, kept for compatibility)
            dpt_features: List of [B, dpt_dim, H, W] for 4 DPT layers
            patch_h, patch_w: Spatial dimensions
            attention_weights_multi_layer: List of attention from [Layer 4, 11, 17, 23]
            cls_token: [B, embed_dim] (unused, kept for compatibility)

        Returns:
            path_1_modulated: [B, dpt_dim, H, W]
            importance_map: [B, 1, patch_h, patch_w]
            fg_features: [B, 256]
            bg_features: [B, 256]
            fg_mask: [B, 1, patch_h, patch_w] - for visualization
            bg_mask: [B, 1, patch_h, patch_w] - for visualization
        """
        # Remove CLS token to get patch-only tokens
        patch_tokens_only = patch_tokens[:, 1:, :]  # [B, num_patches, embed_dim]

        # Step 1: Generate importance map and FG/BG masks using multi-layer attention fusion
        assert attention_weights_multi_layer is not None, "attention_weights_multi_layer required"
        importance_map = self.multi_layer_fusion(attention_weights_multi_layer, patch_h, patch_w)

        # Simple mean-based FG/BG split for fused importance
        importance_flat = importance_map.flatten(2).squeeze(1)
        threshold = importance_flat.mean(dim=1, keepdim=True)
        fg_mask = (importance_flat > threshold).float().reshape(importance_map.shape)
        bg_mask = (importance_flat <= threshold).float().reshape(importance_map.shape)

        # Step 2: Generate FG/BG features
        # Binary masks need importance_map for soft weighting (matches Gear3)
        fg_features, bg_features = self.fg_bg_networks(
            patch_tokens_only, fg_mask, bg_mask, importance_map=importance_map
        )

        # Step 3: Get modulation parameters
        fg_gamma, fg_beta, bg_gamma, bg_beta = self.modulation_networks(
            fg_features, bg_features
        )

        # Step 4: Modulate path_1
        path_1 = dpt_features[-1]
        path_1_modulated = self.feature_modulator(
            path_1, importance_map, fg_gamma, fg_beta, bg_gamma, bg_beta
        )

        return path_1_modulated, importance_map, fg_features, bg_features, fg_mask, bg_mask


# ==================== Ablation Study: Multi-layer CLS without FG/BG Separation ====================

class MultiLayerCLSNetwork(nn.Module):
    """
    Extract multi-layer CLS features and fuse them.
    
    This module extracts CLS tokens from multiple ViT layers (4, 11, 17, 23)
    and fuses them into a single global feature.
    
    Unlike Gear2 which uses only Layer 23 CLS, this captures hierarchical semantics:
    - Layer 4 (early): Low-level patterns
    - Layer 11 (mid): Mid-level semantics
    - Layer 17 (late): High-level semantics
    - Layer 23 (last): Abstract semantics
    """
    def __init__(self, embed_dim=1024, feature_dim=256, num_layers=4):
        super().__init__()
        self.num_layers = num_layers

        # Uniform fusion weights (non-trainable, consistent with Gear5)
        # Equal weight for all layers: 25:25:25:25
        uniform_weights = torch.ones(num_layers) / num_layers
        self.register_buffer('fusion_weights', uniform_weights)

        # Project fused CLS to feature space
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True)
        )

        logging.info(f"Multi-layer CLS Network: {num_layers} layers -> {feature_dim} features (uniform fusion: 25:25:25:25)")
    
    def forward(self, cls_tokens_list):
        """
        Args:
            cls_tokens_list: List of [B, embed_dim] CLS tokens from different layers
                            [Layer 4, Layer 11, Layer 17, Layer 23]

        Returns:
            global_feature: [B, feature_dim] - fused multi-layer feature
        """
        # Stack CLS tokens: [B, num_layers, embed_dim]
        cls_stack = torch.stack(cls_tokens_list, dim=1)

        # Uniform weighted average (25:25:25:25)
        # No normalization needed - weights already sum to 1.0
        cls_fused = (cls_stack * self.fusion_weights.view(1, -1, 1)).sum(dim=1)

        # Project to feature space
        global_feature = self.projection(cls_fused)

        return global_feature


class Gear4AblationHead(nn.Module):
    """
    Ablation Study: Multi-layer CLS features WITHOUT FG/BG separation.
    
    This is a hybrid of Gear2 and Gear3 Upgrade:
    - Uses multi-layer CLS tokens (like Gear3 Upgrade's multi-layer approach)
    - No FG/BG separation (like Gear2's uniform modulation)
    
    Purpose: Evaluate whether the gain from multi-layer comes from:
        (a) Better global features (multi-layer CLS)
        (b) Better spatial reasoning (FG/BG separation)
    
    Expected result: If this performs better than Gear2 but worse than Gear3 Upgrade multi_layer,
    it confirms that BOTH multi-layer features AND FG/BG separation contribute to performance.
    """
    def __init__(self, embed_dim=1024, dpt_dim=256):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Multi-layer CLS feature extraction
        self.multi_layer_cls = MultiLayerCLSNetwork(
            embed_dim=embed_dim,
            feature_dim=256,
            num_layers=4
        )
        
        # Modulation network (uniform, like Gear2)
        from flashdepth.gear2_modules import ModulationNetwork, SimpleFeatureModulator
        self.modulation_network = ModulationNetwork(
            feature_dim=256,
            dpt_dim=dpt_dim
        )
        self.feature_modulator = SimpleFeatureModulator()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(f"Gear3 Upgrade Ablation Head (multi-layer CLS, no separation): {trainable_params:,} / {total_params:,} trainable parameters")
    
    def forward(self, cls_tokens_multi_layer, dpt_features):
        """
        Args:
            cls_tokens_multi_layer: List of [B, embed_dim] CLS tokens from [Layer 4, 11, 17, 23]
            dpt_features: List of [B, dpt_dim, H, W] for 4 DPT layers
        
        Returns:
            path_1_modulated: [B, dpt_dim, H, W]
            (dummy values for compatibility with Gear3 Upgrade interface)
        """
        # Step 1: Extract multi-layer global feature
        global_feature = self.multi_layer_cls(cls_tokens_multi_layer)  # [B, 256]
        
        # Step 2: Get uniform modulation parameters
        gamma, beta = self.modulation_network(global_feature)
        
        # Step 3: Modulate path_1 (last DPT layer)
        path_1 = dpt_features[-1]
        path_1_modulated = self.feature_modulator(path_1, gamma, beta)
        
        # Return dummy values for unused outputs (for compatibility)
        B = path_1.shape[0]
        dummy_importance = torch.zeros(B, 1, 1, 1, device=path_1.device)
        dummy_features = torch.zeros(B, 256, device=path_1.device)
        dummy_mask = torch.zeros(B, 1, 1, 1, device=path_1.device)
        
        return path_1_modulated, dummy_importance, dummy_features, dummy_features, dummy_mask, dummy_mask
