"""
Gear3 Modules: Feature-level Metric Injection via FiLM-style Modulation

This module implements the following architecture:
1. ImportancePredictor: Predicts spatial importance map from attention weights
2. ForegroundBackgroundNetworks: Generates FG/BG semantic features
3. ModulationNetworks: Generates gamma and beta for FiLM-style modulation
4. FeatureModulator: Applies hierarchical modulation to DPT features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import logging


class ImportancePredictor(nn.Module):
    """
    Predicts spatial importance map (0~1) from DINOv2 attention weights.

    Input: Attention weights from DINOv2 multi-head attention [B, num_heads, num_patches+1, num_patches+1]
    Output: Importance map [B, 1, H, W] (0~1, higher = foreground/important)
    """
    def __init__(self, num_heads=16, hidden_dim=128):
        super().__init__()
        self.num_heads = num_heads

        # Process attention weights to spatial importance
        self.conv1 = nn.Conv2d(num_heads, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim // 2)

        self.conv3 = nn.Conv2d(hidden_dim // 2, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        logging.info(f"ImportancePredictor initialized with {num_heads} heads")

    def forward(self, attention_weights, patch_h, patch_w):
        """
        Args:
            attention_weights: [B, num_heads, num_patches+1, num_patches+1]
            patch_h, patch_w: Spatial dimensions of patches

        Returns:
            importance_map: [B, 1, patch_h, patch_w] in range [0, 1]
        """
        B = attention_weights.shape[0]

        # Extract patch-to-patch attention (exclude CLS token)
        # Average over CLS attention to patches: [B, num_heads, num_patches]
        cls_to_patches = attention_weights[:, :, 0, 1:]  # [B, num_heads, num_patches]

        # Reshape to spatial: [B, num_heads, patch_h, patch_w]
        attn_spatial = cls_to_patches.reshape(B, self.num_heads, patch_h, patch_w)

        # Process through conv layers
        x = self.conv1(attn_spatial)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        importance_map = self.sigmoid(x)  # [B, 1, patch_h, patch_w]

        return importance_map


class ForegroundBackgroundNetworks(nn.Module):
    """
    Generates foreground and background semantic features from patch tokens.

    Input: Patch tokens [B, num_patches, embed_dim]
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

        logging.info(f"FG/BG Networks initialized: {embed_dim} -> {feature_dim}")

    def forward(self, patch_tokens):
        """
        Args:
            patch_tokens: [B, num_patches, embed_dim]

        Returns:
            fg_features: [B, feature_dim]
            bg_features: [B, feature_dim]
        """
        # Global average pooling over patches
        global_features = patch_tokens.mean(dim=1)  # [B, embed_dim]

        fg_features = self.fg_net(global_features)  # [B, feature_dim]
        bg_features = self.bg_net(global_features)  # [B, feature_dim]

        return fg_features, bg_features


class ModulationNetworks(nn.Module):
    """
    Generates gamma and beta for FiLM-style modulation for each DPT layer.

    Input: FG/BG features [B, feature_dim]
    Output: Gamma [B, dpt_dim], Beta [B, dpt_dim] for FG and BG separately
    """
    def __init__(self, feature_dim=256, dpt_dim=256, num_dpt_layers=4):
        super().__init__()
        self.num_dpt_layers = num_dpt_layers
        self.dpt_dim = dpt_dim

        # Separate modulation networks for each DPT layer
        self.fg_modulation = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, dpt_dim * 2),
                nn.ReLU(inplace=True),
                nn.Linear(dpt_dim * 2, dpt_dim * 2)  # First half: gamma, second half: beta
            ) for _ in range(num_dpt_layers)
        ])

        self.bg_modulation = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, dpt_dim * 2),
                nn.ReLU(inplace=True),
                nn.Linear(dpt_dim * 2, dpt_dim * 2)  # First half: gamma, second half: beta
            ) for _ in range(num_dpt_layers)
        ])

        logging.info(f"Modulation Networks initialized for {num_dpt_layers} DPT layers")

    def forward(self, fg_features, bg_features, layer_idx):
        """
        Args:
            fg_features: [B, feature_dim]
            bg_features: [B, feature_dim]
            layer_idx: Which DPT layer (0-3)

        Returns:
            fg_gamma: [B, dpt_dim]
            fg_beta: [B, dpt_dim]
            bg_gamma: [B, dpt_dim]
            bg_beta: [B, dpt_dim]
        """
        # FG modulation
        fg_params = self.fg_modulation[layer_idx](fg_features)  # [B, dpt_dim * 2]
        fg_gamma = fg_params[:, :self.dpt_dim]
        fg_beta = fg_params[:, self.dpt_dim:]

        # BG modulation
        bg_params = self.bg_modulation[layer_idx](bg_features)  # [B, dpt_dim * 2]
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

        # Spatially-varying modulation parameters
        gamma = importance_map * fg_gamma + (1 - importance_map) * bg_gamma  # [B, C, H, W]
        beta = importance_map * fg_beta + (1 - importance_map) * bg_beta  # [B, C, H, W]

        # Apply FiLM modulation
        modulated_features = gamma * features + beta

        return modulated_features


class Gear3MetricHead(nn.Module):
    """
    Complete Gear3 metric depth head combining all modules.

    Architecture:
        1. ImportancePredictor: attention -> importance map
        2. FG/BG Networks: patch tokens -> FG/BG features
        3. Modulation Networks: FG/BG features -> gamma/beta for each layer
        4. Feature Modulator: Apply modulation to DPT layer features
    """
    def __init__(self, embed_dim=1024, dpt_dim=256, num_heads=16, num_dpt_layers=4):
        super().__init__()

        self.importance_predictor = ImportancePredictor(num_heads=num_heads)
        self.fg_bg_networks = ForegroundBackgroundNetworks(
            embed_dim=embed_dim, feature_dim=256
        )
        self.modulation_networks = ModulationNetworks(
            feature_dim=256, dpt_dim=dpt_dim, num_dpt_layers=num_dpt_layers
        )
        self.feature_modulator = FeatureModulator()

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(f"Gear3 Metric Head: {trainable_params:,} / {total_params:,} trainable parameters")

    def forward(self, patch_tokens, attention_weights, dpt_features, patch_h, patch_w):
        """
        Args:
            patch_tokens: [B, num_patches, embed_dim]
            attention_weights: [B, num_heads, num_patches+1, num_patches+1]
            dpt_features: List of [B, dpt_dim, H, W] for 4 DPT layers
            patch_h, patch_w: Spatial dimensions

        Returns:
            modulated_dpt_features: List of modulated DPT features
            importance_map: [B, 1, patch_h, patch_w] for visualization
        """
        # 1. Predict importance map
        importance_map = self.importance_predictor(attention_weights, patch_h, patch_w)

        # 2. Generate FG/BG features
        fg_features, bg_features = self.fg_bg_networks(patch_tokens)

        # 3. Modulate each DPT layer
        modulated_dpt_features = []
        for layer_idx, features in enumerate(dpt_features):
            # Get modulation parameters for this layer
            fg_gamma, fg_beta, bg_gamma, bg_beta = self.modulation_networks(
                fg_features, bg_features, layer_idx
            )

            # Apply modulation
            modulated = self.feature_modulator(
                features, importance_map, fg_gamma, fg_beta, bg_gamma, bg_beta
            )
            modulated_dpt_features.append(modulated)

        return modulated_dpt_features, importance_map
