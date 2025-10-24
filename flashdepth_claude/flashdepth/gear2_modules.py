"""
Gear2 Modules: Ablation version of Gear3 without FG/BG separation

Key differences from Gear3:
- No importance map computation
- No FG/BG separation
- Uses CLS token for global feature extraction
- Single modulation parameters (gamma/beta) applied uniformly to all pixels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


class GlobalFeatureNetwork(nn.Module):
    """
    Extracts global semantic feature from CLS token.

    Input: CLS token [B, embed_dim]
    Output: Global feature [B, feature_dim]
    """
    def __init__(self, embed_dim=1024, feature_dim=256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embed_dim, feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True)
        )

        logging.info(f"Global Feature Network: {embed_dim} -> {feature_dim}")

    def forward(self, cls_token):
        """
        Args:
            cls_token: [B, embed_dim] from DINOv2

        Returns:
            global_feature: [B, feature_dim]
        """
        return self.network(cls_token)


class ModulationNetwork(nn.Module):
    """
    Generates gamma and beta for uniform FiLM-style modulation.

    Input: Global feature [B, feature_dim]
    Output: Gamma [B, dpt_dim], Beta [B, dpt_dim]
    """
    def __init__(self, feature_dim=256, dpt_dim=256):
        super().__init__()
        self.dpt_dim = dpt_dim

        # Single modulation network for path_1 (Layer 23)
        self.modulation = nn.Sequential(
            nn.Linear(feature_dim, dpt_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(dpt_dim * 2, dpt_dim * 2)  # First half: gamma, second half: beta
        )

        logging.info("Modulation Network initialized for uniform modulation")

    def forward(self, global_feature):
        """
        Args:
            global_feature: [B, feature_dim]

        Returns:
            gamma: [B, dpt_dim]
            beta: [B, dpt_dim]
        """
        params = self.modulation(global_feature)  # [B, dpt_dim * 2]
        gamma = params[:, :self.dpt_dim]
        beta = params[:, self.dpt_dim:]

        return gamma, beta


class SimpleFeatureModulator(nn.Module):
    """
    Applies uniform FiLM-style modulation to DPT features.

    Modulation formula:
        modulated[x,y] = gamma ⊙ feature[x,y] + beta

    Unlike Gear3, gamma and beta are constant across all spatial locations.
    """
    def __init__(self):
        super().__init__()

    def forward(self, features, gamma, beta):
        """
        Args:
            features: [B, C, H, W] DPT layer features
            gamma: [B, C] uniform scaling parameter
            beta: [B, C] uniform shift parameter

        Returns:
            modulated_features: [B, C, H, W]
        """
        B, C, H, W = features.shape

        # Expand gamma and beta to spatial dimensions
        gamma = gamma.view(B, C, 1, 1)  # [B, C, 1, 1]
        beta = beta.view(B, C, 1, 1)    # [B, C, 1, 1]

        # Apply uniform FiLM modulation
        modulated_features = gamma * features + beta

        return modulated_features


class Gear2MetricHead(nn.Module):
    """
    Gear2 metric depth head: Ablation version without FG/BG separation.

    Architecture:
        1. CLS token extraction
        2. Global feature network: CLS -> global semantic feature
        3. Modulation network: global feature -> gamma/beta
        4. Uniform feature modulator: Apply same gamma/beta to all pixels

    This serves as a baseline to evaluate the contribution of:
    - Importance map computation
    - FG/BG separation
    - Spatial-varying modulation
    """
    def __init__(self, embed_dim=1024, dpt_dim=256):
        super().__init__()

        self.global_feature_net = GlobalFeatureNetwork(
            embed_dim=embed_dim, feature_dim=256
        )
        self.modulation_net = ModulationNetwork(
            feature_dim=256, dpt_dim=dpt_dim
        )
        self.feature_modulator = SimpleFeatureModulator()

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(f"Gear2 Metric Head: {trainable_params:,} / {total_params:,} trainable parameters")

    def forward(self, patch_tokens, attention_weights, dpt_features, patch_h, patch_w):
        """
        Args:
            patch_tokens: [B, num_patches+1, embed_dim] from Layer 23
                         Note: First token (index 0) is CLS token
            attention_weights: [B, num_heads, num_patches+1, num_patches+1] (not used in Gear2)
            dpt_features: List of [B, dpt_dim, H, W] for 4 DPT layers
            patch_h, patch_w: Spatial dimensions (not used in Gear2)

        Returns:
            path_1_modulated: [B, dpt_dim, H, W] modulated path_1 features
            importance_map: None (not computed in Gear2)
            fg_features: None (not computed in Gear2)
            bg_features: None (not computed in Gear2)
        """
        # 1. Extract CLS token (first token in sequence)
        cls_token = patch_tokens[:, 0]  # [B, embed_dim]

        # 2. Generate global semantic feature
        global_feature = self.global_feature_net(cls_token)  # [B, 256]

        # 3. Get modulation parameters (uniform for all pixels)
        gamma, beta = self.modulation_net(global_feature)  # [B, dpt_dim] each

        # 4. Modulate ONLY path_1 (last element, from Layer 23)
        path_1 = dpt_features[-1]  # [B, dpt_dim, H, W]
        path_1_modulated = self.feature_modulator(path_1, gamma, beta)

        # Return None for removed components (for compatibility with Gear3 interface)
        return path_1_modulated, None, None, None
