"""
Onepiece V3 Core Modules for FlashDepth-Metric.

Architecture (Dual-Stream):
    DPT features → SpatialMamba (downsample→Mamba→final_layer→upsample+add)
        → Relative stream: post_mamba [B*T, 256, h, w] → final_head → relative depth
        → Metric stream: mamba_raw [B*T, 256, h', w'] → ConvMetricHead → scale, shift

Components:
    1. SpatialMamba: FlashDepth-style spatial Mamba on 1/10 downsampled DPT features
    2. ConvMetricHead: Conv-based scale/shift prediction from low-res Mamba output
    3. SceneCutDetector: CLS cosine distance for scene cut detection (inference only)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
import logging
from einops import rearrange

from .mamba import MambaBlock, InferenceParams

logger = logging.getLogger(__name__)


# ============================================================================
# V1/V2 modules — commented out for reference
# ============================================================================

# class UnifiedGlobalMamba(nn.Module):
#     """
#     V1: Unified Global Mamba on CLS + GAP concatenated tokens (1280-dim).
#     Replaced by SpatialMamba in V3.
#     """
#     def __init__(self, d_input, num_layers=2, d_state=64, d_conv=4,
#                  expand=2, headdim=64, max_batch_size=8):
#         super().__init__()
#         d_model = self._find_valid_mamba_dim(d_input, expand, headdim)
#         self.d_input = d_input
#         self.d_model = d_model
#         self.num_layers = num_layers
#         if d_input != d_model:
#             self.input_proj = nn.Linear(d_input, d_model)
#             self.output_proj = nn.Linear(d_model, d_input)
#         else:
#             self.input_proj = None
#             self.output_proj = None
#         nheads = d_model * expand // headdim
#         assert nheads % 8 == 0
#         self.blocks = nn.ModuleList([
#             MambaBlock(d_model=d_model, layer_idx=i, expand=expand,
#                        d_state=d_state, d_conv=d_conv, headdim=headdim, use_hydra=False)
#             for i in range(num_layers)
#         ])
#         self.max_seqlen = 60000
#         self.max_batch_size = max_batch_size
#         self.inference_params = None
#
#     @staticmethod
#     def _find_valid_mamba_dim(d_input, expand, headdim):
#         unit = headdim * 8 // expand
#         d_model = ((d_input + unit - 1) // unit) * unit
#         return d_model
#
#     def start_new_sequence(self):
#         self.inference_params = InferenceParams(
#             max_seqlen=self.max_seqlen, max_batch_size=self.max_batch_size)
#
#     def forward(self, x):
#         B, T, D = x.shape
#         if self.input_proj is not None:
#             x = self.input_proj(x)
#         for block in self.blocks:
#             x = block(x, inference_params=None)
#         if self.output_proj is not None:
#             x = self.output_proj(x)
#         return x
#
#     def forward_single_frame(self, x):
#         if self.inference_params is None:
#             self.start_new_sequence()
#         if self.input_proj is not None:
#             x = self.input_proj(x)
#         for block in self.blocks:
#             x = block(x, inference_params=self.inference_params)
#         if self.output_proj is not None:
#             x = self.output_proj(x)
#         self.inference_params.seqlen_offset += x.shape[1]
#         return x

# class OnepieceMetricHead(nn.Module):
#     """
#     V1: MLP-based scale/shift + FiLM from refined global token (1280-dim).
#     Replaced by ConvMetricHead in V3.
#     """
#     def __init__(self, input_dim=1280, dpt_dim=256, hidden_dim=512):
#         super().__init__()
#         self.input_dim = input_dim
#         self.dpt_dim = dpt_dim
#         self.scale_shift_mlp = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2))
#         self.film_generator = nn.Sequential(
#             nn.Linear(dpt_dim, dpt_dim), nn.ReLU(), nn.Linear(dpt_dim, dpt_dim * 2))
#
#     def forward(self, refined_global):
#         scale_shift = self.scale_shift_mlp(refined_global)
#         raw_scale, raw_shift = scale_shift[:, 0:1], scale_shift[:, 1:2]
#         scale = F.softplus(raw_scale)
#         shift = 0.1 * torch.sigmoid(raw_shift)
#         refined_gap = refined_global[:, -self.dpt_dim:]
#         film_out = self.film_generator(refined_gap)
#         gamma_raw, beta = film_out.chunk(2, dim=-1)
#         gamma = 1.0 + gamma_raw
#         return scale, shift, gamma, beta


# ============================================================================
# V3 modules
# ============================================================================

class SpatialMamba(nn.Module):
    """
    Spatial Mamba for temporal processing of DPT features.
    Reuses FlashDepth's architecture: downsample → Mamba blocks → final_layer → upsample + add.

    Input: DPT features [B*T, dpt_dim, h, w]
    Output: post_mamba [B*T, dpt_dim, h, w] = DPT + upsample(final_layer(mamba_out))
            mamba_raw [B*T, dpt_dim, h', w'] = low-res Mamba output (for MetricHead)
    """

    def __init__(self, dpt_dim=256, num_layers=4, d_state=256, d_conv=4,
                 expand=2, headdim=64, downsample_factor=0.1, max_batch_size=8):
        super().__init__()

        # ViT-S override (matching FlashDepth MambaModel)
        if dpt_dim == 64:
            expand = 4
            headdim = 32

        self.dpt_dim = dpt_dim
        self.downsample_factor = downsample_factor

        # Mamba blocks (flat list, single DPT layer insertion point)
        self.blocks = nn.ModuleList([
            MambaBlock(
                d_model=dpt_dim,
                layer_idx=i,
                expand=expand,
                d_state=d_state,
                d_conv=d_conv,
                headdim=headdim,
                use_hydra=False
            )
            for i in range(num_layers)
        ])

        # final_layer: GELU + Linear with ZERO INIT (identity at init)
        self.final_layer = nn.Sequential(
            nn.GELU(),
            nn.Linear(dpt_dim, dpt_dim)
        )
        nn.init.zeros_(self.final_layer[1].weight)
        nn.init.zeros_(self.final_layer[1].bias)

        # Inference state
        self.max_seqlen = 60000
        self.max_batch_size = max_batch_size
        self.inference_params = None

        logger.info(
            f"SpatialMamba: dpt_dim={dpt_dim}, layers={num_layers}, "
            f"d_state={d_state}, d_conv={d_conv}, expand={expand}, headdim={headdim}, "
            f"downsample={downsample_factor}"
        )

    def start_new_sequence(self):
        """Reset Mamba hidden state for new video sequence."""
        self.inference_params = InferenceParams(
            max_seqlen=self.max_seqlen,
            max_batch_size=self.max_batch_size
        )

    def forward(self, dpt_features, B, T):
        """
        Batch training mode (per-frame loop with hidden state, matching FlashDepth).

        Args:
            dpt_features: [B*T, dpt_dim, h, w] DPT path_1 features
            B: batch size
            T: number of frames

        Returns:
            post_mamba: [B*T, dpt_dim, h, w] = DPT + upsample(final_layer(mamba_out))
            mamba_raw_spatial: [B*T, dpt_dim, h', w'] = low-res Mamba output
        """
        BT, c, h, w = dpt_features.shape
        assert BT == B * T, f"Expected {B*T}, got {BT}"

        # Save original for residual add
        original = rearrange(dpt_features, '(b t) c h w -> b t c h w', b=B, t=T)

        # Downsample
        h_down = max(int(h * self.downsample_factor), 1)
        w_down = max(int(w * self.downsample_factor), 1)
        down = F.adaptive_avg_pool2d(dpt_features, (h_down, w_down))  # [B*T, c, h', w']

        # Reshape for per-frame processing: [B, T, h'*w', c]
        down_seq = rearrange(down, '(b t) c h w -> b t (h w) c', b=B, t=T)

        # Initialize fresh inference params for training
        self.start_new_sequence()

        # Per-frame Mamba processing (matching FlashDepth pattern)
        mamba_outs = []
        for i in range(T):
            x = down_seq[:, i]  # [B, h'*w', c]
            for block in self.blocks:
                x = block(x, inference_params=self.inference_params)
            self.inference_params.seqlen_offset += x.shape[1]
            mamba_outs.append(x)

        mamba_out = torch.stack(mamba_outs, dim=1)  # [B, T, h'*w', c]

        # Raw spatial output for MetricHead (before final_layer)
        mamba_raw_spatial = rearrange(
            mamba_out, 'b t (h w) c -> (b t) c h w', h=h_down, w=w_down
        )

        # Upsample + final_layer + residual add (matching FlashDepth dpt_features_to_mamba)
        post_mamba_list = []
        for i in range(T):
            # Upsample low-res mamba output to original DPT resolution
            frame_raw = rearrange(
                mamba_out[:, i], 'b (h w) c -> b c h w', h=h_down, w=w_down
            )
            upsampled = F.interpolate(frame_raw, (h, w), mode='bilinear', align_corners=True)

            # Apply final_layer (zero-init → identity at start)
            up_flat = rearrange(upsampled, 'b c h w -> b (h w) c')
            final_out = self.final_layer(up_flat)
            final_spatial = rearrange(final_out, 'b (h w) c -> b c h w', h=h, w=w)

            # Residual add with original DPT features
            post = final_spatial + original[:, i]
            post_mamba_list.append(post)

        post_mamba = torch.stack(post_mamba_list, dim=1)  # [B, T, c, h, w]
        post_mamba = rearrange(post_mamba, 'b t c h w -> (b t) c h w')

        return post_mamba, mamba_raw_spatial

    def forward_single_frame(self, dpt_features_single):
        """
        Streaming inference mode. Single frame with hidden state.

        Args:
            dpt_features_single: [B, dpt_dim, h, w]

        Returns:
            post_mamba: [B, dpt_dim, h, w]
            mamba_raw_spatial: [B, dpt_dim, h', w']
        """
        if self.inference_params is None:
            self.start_new_sequence()

        B, c, h, w = dpt_features_single.shape

        # Save original for residual
        original = dpt_features_single

        # Downsample
        h_down = max(int(h * self.downsample_factor), 1)
        w_down = max(int(w * self.downsample_factor), 1)
        down = F.adaptive_avg_pool2d(dpt_features_single, (h_down, w_down))

        # Reshape: [B, h'*w', c]
        x = rearrange(down, 'b c h w -> b (h w) c')

        # Mamba blocks with hidden state
        for block in self.blocks:
            x = block(x, inference_params=self.inference_params)
        self.inference_params.seqlen_offset += x.shape[1]

        # Raw spatial output for MetricHead
        mamba_raw_spatial = rearrange(x, 'b (h w) c -> b c h w', h=h_down, w=w_down)

        # Upsample + final_layer + residual
        upsampled = F.interpolate(mamba_raw_spatial, (h, w), mode='bilinear', align_corners=True)
        up_flat = rearrange(upsampled, 'b c h w -> b (h w) c')
        final_out = self.final_layer(up_flat)
        final_spatial = rearrange(final_out, 'b (h w) c -> b c h w', h=h, w=w)

        post_mamba = final_spatial + original

        return post_mamba, mamba_raw_spatial


class ConvMetricHead(nn.Module):
    """
    Conv-based scale/shift prediction from low-res Mamba output.

    Input: [B*T, dpt_dim, h', w'] (Mamba raw output, NOT upsampled)
    Output: scale [B*T, 1], shift [B*T, 1]

    Pipeline: Conv1x1(dpt_dim→hidden) → ReLU → Conv1x1(hidden→2) → GAP → scale/shift
    """

    def __init__(self, dpt_dim=256, hidden_dim=64, train_mode="metric"):
        super().__init__()
        self.train_mode = train_mode

        self.conv = nn.Sequential(
            nn.Conv2d(dpt_dim, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 2, 1)
        )

        self._initialize_weights()

        logger.info(f"ConvMetricHead: dpt_dim={dpt_dim}, hidden={hidden_dim}, mode={train_mode}")

    def _initialize_weights(self):
        """Initialize for stable training start (matching V1 OnepieceMetricHead)."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Scale/Shift initialization on final conv
        with torch.no_grad():
            # Scale bias: softplus(0.5413) ≈ 1.0
            self.conv[-1].bias.data[0] = 0.5413
            # Shift bias: 0.1 * sigmoid(-5) ≈ 0
            self.conv[-1].bias.data[1] = -5.0

        logger.info("ConvMetricHead initialized: scale≈1.0, shift≈0.0")

    def forward(self, mamba_raw):
        """
        Args:
            mamba_raw: [B*T, dpt_dim, h', w'] low-res Mamba output

        Returns:
            scale: [B*T, 1] positive scale values
            shift: [B*T, 1] shift values
        """
        out = self.conv(mamba_raw)  # [B*T, 2, h', w']

        # Global average pooling
        out = F.adaptive_avg_pool2d(out, 1).squeeze(-1).squeeze(-1)  # [B*T, 2]
        raw_scale, raw_shift = out[:, 0:1], out[:, 1:2]

        scale = F.softplus(raw_scale)  # Always positive

        if self.train_mode == "metric":
            shift = 0.1 * torch.sigmoid(raw_shift)  # Range [0, 0.1]
        else:  # inverse mode
            shift = raw_shift  # Unconstrained

        return scale, shift


class SceneCutDetector(nn.Module):
    """
    Scene Cut Detector using CLS token cosine distance.
    Used at inference only (not in training forward pass for V3).

    D_cls = 1 - cos_sim(CLS_t, CLS_{t-1})
    W_temporal = 1 - sigmoid(k * (D_cls - tau))

    When D_cls < tau: W_temporal ≈ 1.0 (same scene, keep temporal state)
    When D_cls > tau: W_temporal ≈ 0.0 (scene cut, reset Mamba state)
    """

    def __init__(self, tau=0.05, k=80):
        super().__init__()
        self.tau = tau
        self.k = k

    @torch.no_grad()
    def forward(self, cls_tokens):
        """
        Args:
            cls_tokens: [B, T, embed_dim] CLS tokens per frame

        Returns:
            temporal_weights: [B, T-1] weights (1.0=keep, 0.0=cut)
            d_cls: [B, T-1] cosine distances (for logging)
        """
        B, T, D = cls_tokens.shape

        if T < 2:
            return (
                torch.ones(B, 0, device=cls_tokens.device),
                torch.zeros(B, 0, device=cls_tokens.device)
            )

        # Compute cosine similarity between consecutive frames
        cls_t = F.normalize(cls_tokens[:, 1:], dim=-1)      # [B, T-1, D]
        cls_t_prev = F.normalize(cls_tokens[:, :-1], dim=-1)  # [B, T-1, D]

        cos_sim = (cls_t * cls_t_prev).sum(dim=-1)  # [B, T-1]
        d_cls = 1.0 - cos_sim  # Cosine distance [0, 2]

        # Soft thresholding
        temporal_weights = 1.0 - torch.sigmoid(self.k * (d_cls - self.tau))

        return temporal_weights, d_cls
