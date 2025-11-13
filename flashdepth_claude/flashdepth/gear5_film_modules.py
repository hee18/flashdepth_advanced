"""
Gear5 FiLM Modules: Temporal FiLM-style modulation for DPT features

Key features:
- Multi-layer CLS token extraction (Layers 11, 23 from Gear5)
- Temporal processing: handles video sequences [B, T, ...]
- Channel-wise FiLM modulation (Gear2 style)
- Modulates DPT features BEFORE Mamba temporal modeling
- Importance map generation from attention weights (for loss weighting)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


# ==================== Importance Map Generator ====================

class ImportanceMapGenerator(nn.Module):
    """
    Generate importance map from multi-layer CLS attention weights.
    (Same as Gear5 - used for importance-weighted loss)

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
        logging.info(f"[Gear5 FiLM] ImportanceMapGenerator: Averaging {num_layers} attention layers")

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
        for b in range(B):
            attn_flat = importance_map[b].flatten()
            attn_p1 = torch.quantile(attn_flat, 0.01)   # 1st percentile
            attn_p99 = torch.quantile(attn_flat, 0.99)  # 99th percentile

            # Normalize to [0, 1] and clip
            importance_map[b] = (importance_map[b] - attn_p1) / (attn_p99 - attn_p1 + 1e-8)
            importance_map[b] = torch.clamp(importance_map[b], 0.0, 1.0)

        return importance_map


# ==================== FiLM Modulation Networks ====================

class GlobalFeatureNetwork(nn.Module):
    """
    Extracts global semantic feature from CLS token across temporal dimension.

    Processes each frame independently, then handles temporal dimension.

    Input: CLS token [B, T, embed_dim]
    Output: Global feature [B, T, feature_dim]
    """
    def __init__(self, embed_dim=1024, feature_dim=256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embed_dim, feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True)
        )

        logging.info(f"[Gear5 FiLM] Global Feature Network: {embed_dim} -> {feature_dim}")

    def forward(self, cls_token):
        """
        Args:
            cls_token: [B, T, embed_dim] from DINOv2

        Returns:
            global_feature: [B, T, feature_dim]
        """
        B, T, C = cls_token.shape

        # Reshape to process all frames at once
        cls_token_flat = cls_token.reshape(B * T, C)  # [B*T, embed_dim]

        # Extract global features
        global_feature_flat = self.network(cls_token_flat)  # [B*T, feature_dim]

        # Reshape back to temporal
        global_feature = global_feature_flat.reshape(B, T, -1)  # [B, T, feature_dim]

        return global_feature


class ModulationNetwork(nn.Module):
    """
    Generates gamma and beta for channel-wise FiLM-style modulation.

    Processes temporal sequences to generate modulation parameters for each frame.

    Input: Global feature [B, T, feature_dim]
    Output: Gamma [B, T, dpt_dim], Beta [B, T, dpt_dim]
    """
    def __init__(self, feature_dim=256, dpt_dim=256):
        super().__init__()
        self.dpt_dim = dpt_dim

        # Modulation network for path_1 (Layer 23 DPT features)
        self.modulation = nn.Sequential(
            nn.Linear(feature_dim, dpt_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(dpt_dim * 2, dpt_dim * 2)  # First half: gamma, second half: beta
        )

        logging.info("[Gear5 FiLM] Modulation Network initialized for channel-wise modulation")

    def forward(self, global_feature):
        """
        Args:
            global_feature: [B, T, feature_dim]

        Returns:
            gamma: [B, T, dpt_dim]
            beta: [B, T, dpt_dim]
        """
        B, T, C = global_feature.shape

        # Reshape to process all frames
        global_feature_flat = global_feature.reshape(B * T, C)  # [B*T, feature_dim]

        # Generate modulation parameters
        params = self.modulation(global_feature_flat)  # [B*T, dpt_dim * 2]
        gamma_flat = params[:, :self.dpt_dim]
        beta_flat = params[:, self.dpt_dim:]

        # Reshape back to temporal
        gamma = gamma_flat.reshape(B, T, self.dpt_dim)  # [B, T, dpt_dim]
        beta = beta_flat.reshape(B, T, self.dpt_dim)    # [B, T, dpt_dim]

        return gamma, beta


class SimpleFeatureModulator(nn.Module):
    """
    Applies channel-wise FiLM-style modulation to DPT features.

    Modulation formula:
        modulated[b,t,c,x,y] = gamma[b,t,c] ⊙ feature[b,t,c,x,y] + beta[b,t,c]

    Each channel gets its own gamma and beta, but all spatial locations
    within a channel share the same modulation parameters.
    """
    def __init__(self):
        super().__init__()

    def forward(self, features, gamma, beta):
        """
        Args:
            features: [B*T, C, H, W] DPT layer features (flattened temporal)
            gamma: [B, T, C] channel-wise scaling parameter
            beta: [B, T, C] channel-wise shift parameter

        Returns:
            modulated_features: [B*T, C, H, W]
        """
        BT, C, H, W = features.shape
        B, T, _ = gamma.shape

        # Reshape gamma and beta to [B*T, C]
        gamma_flat = gamma.reshape(B * T, C)  # [B*T, C]
        beta_flat = beta.reshape(B * T, C)    # [B*T, C]

        # Expand to spatial dimensions
        gamma_expanded = gamma_flat.view(B * T, C, 1, 1)  # [B*T, C, 1, 1]
        beta_expanded = beta_flat.view(B * T, C, 1, 1)    # [B*T, C, 1, 1]

        # Apply channel-wise FiLM modulation
        modulated_features = gamma_expanded * features + beta_expanded

        return modulated_features


class Gear5FilmHead(nn.Module):
    """
    Gear5 FiLM metric depth head: Temporal FiLM-style modulation + Importance Map.

    Architecture:
        1. Multi-layer CLS token extraction (Layers 11, 23)
        2. Global feature network: CLS -> global semantic feature
        3. Modulation network: global feature -> gamma/beta
        4. Channel-wise feature modulator: Apply gamma/beta per channel
        5. Importance map generator: attention weights -> importance map (for loss)

    Key differences from Gear2:
    - Handles temporal sequences [B, T, ...]
    - Uses multi-layer CLS tokens (Gear5 style)
    - Modulates features BEFORE Mamba temporal modeling

    Key differences from Gear5 (original):
    - Uses FiLM modulation instead of GRU-based scale/shift prediction
    - Modulates intermediate DPT features, not final depth output
    - Still generates importance map for loss weighting (like Gear5)
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

        # Importance map generator (for loss weighting, like Gear5)
        self.importance_map_generator = ImportanceMapGenerator(num_layers=2)

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(f"[Gear5 FiLM] Metric Head: {trainable_params:,} / {total_params:,} trainable parameters")

    def forward(self, cls_tokens_multi_layer, attention_weights_list, dpt_features, patch_h, patch_w):
        """
        Args:
            cls_tokens_multi_layer: List of [B, T, embed_dim] from Layers 11, 23
                                   Note: Each tensor contains only CLS tokens (no patch tokens)
            attention_weights_list: List of [B*T, num_heads, N+1, N+1] from 2 layers (for importance map)
            dpt_features: List of [B*T, dpt_dim, H, W] for 4 DPT layers
                         Features are already flattened in temporal dimension
            patch_h, patch_w: Spatial patch dimensions (e.g., 37×37 for 518×518)

        Returns:
            dict with:
                - path_1_modulated: [B*T, dpt_dim, H, W] modulated path_1 features
                - gamma: [B, T, dpt_dim] modulation scaling parameters
                - beta: [B, T, dpt_dim] modulation shift parameters
                - importance_map: [B, T, patch_h, patch_w] importance map for loss
        """
        # 1. Use the last layer's CLS tokens (Layer 23, following Gear5 pattern)
        cls_token = cls_tokens_multi_layer[-1]  # [B, T, embed_dim]
        B, T = cls_token.shape[:2]

        # 2. Generate global semantic feature
        global_feature = self.global_feature_net(cls_token)  # [B, T, 256]

        # 3. Get modulation parameters (channel-wise)
        gamma, beta = self.modulation_net(global_feature)  # [B, T, dpt_dim] each

        # 4. Modulate ONLY path_1 (last element, from Layer 23)
        path_1 = dpt_features[-1]  # [B*T, dpt_dim, H, W]
        path_1_modulated = self.feature_modulator(path_1, gamma, beta)

        # 5. Generate importance map (for loss weighting)
        importance_map = self.importance_map_generator(
            attention_weights_list, patch_h, patch_w
        )  # [B*T, 1, patch_h, patch_w]

        # Reshape importance map to [B, T, patch_h, patch_w]
        importance_map = importance_map.view(B, T, patch_h, patch_w)

        return {
            'path_1_modulated': path_1_modulated,
            'gamma': gamma,
            'beta': beta,
            'importance_map': importance_map
        }
