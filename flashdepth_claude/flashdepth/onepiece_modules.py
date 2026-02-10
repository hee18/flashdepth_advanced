"""
Onepiece Core Modules for FlashDepth-Metric.

Architecture:
    CLS(1024) + DPT GAP(256) → Unified Global Mamba(1280) → Metric Head + FiLM

Components:
    1. UnifiedGlobalMamba: 2-layer Mamba2 on concatenated CLS+GAP tokens (1280-dim)
    2. OnepieceMetricHead: Scale/Shift prediction + FiLM spatial guidance
    3. SceneCutDetector: CLS cosine distance for scene cut detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
import logging

from .mamba import MambaBlock, InferenceParams

logger = logging.getLogger(__name__)


class UnifiedGlobalMamba(nn.Module):
    """
    Unified Global Mamba: Temporal processing on CLS + GAP concatenated tokens.

    ViT-L: CLS(1024) + GAP(256) = 1280-dim → d_model=1280 (valid for Mamba2)
    ViT-S: CLS(384) + GAP(64) = 448-dim → projected to d_model=512 (448 is not Mamba2-valid)

    Uses 2 MambaBlock layers with expand=2, headdim=64.
    Constraint: d_model * expand / headdim must be multiple of 8.

    When d_input != d_model, input/output projection layers are added automatically.

    Two forward modes:
        - forward(x): Batch mode for training (parallel scan on full [B, T, D])
        - forward_single_frame(x): Streaming mode for inference (per-frame with hidden state)

    NOTE - Architectural option (not yet implemented):
        Current ViT-L uses d_model=1280 (46.36M params) for 514 output values (2 scale/shift + 512 FiLM).
        Alternative: project CLS 1024→256 before concat → d_input=512, d_model=512 (~5.4M per 2 layers).
        This would reduce Mamba params by ~6x while keeping the same output dimensionality.
        See Onepiece.md "Architectural Option: CLS Dimension Reduction" for details.
    """

    def __init__(self, d_input, num_layers=2, d_state=64, d_conv=4,
                 expand=2, headdim=64, max_batch_size=8):
        super().__init__()

        # Find valid Mamba2 dimension >= d_input
        d_model = self._find_valid_mamba_dim(d_input, expand, headdim)

        self.d_input = d_input
        self.d_model = d_model
        self.num_layers = num_layers

        # Add projection layers if raw concat dim is not Mamba2-valid
        if d_input != d_model:
            self.input_proj = nn.Linear(d_input, d_model)
            self.output_proj = nn.Linear(d_model, d_input)
            logger.info(f"UnifiedGlobalMamba: projection {d_input} → {d_model} → {d_input}")
        else:
            self.input_proj = None
            self.output_proj = None

        # Validate dimensions
        nheads = d_model * expand // headdim
        assert nheads % 8 == 0, (
            f"d_model({d_model}) * expand({expand}) / headdim({headdim}) = {nheads}, "
            f"must be multiple of 8"
        )

        # Mamba blocks
        self.blocks = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                layer_idx=i,
                expand=expand,
                d_state=d_state,
                d_conv=d_conv,
                headdim=headdim,
                use_hydra=False
            )
            for i in range(num_layers)
        ])

        # Inference state
        self.max_seqlen = 60000
        self.max_batch_size = max_batch_size
        self.inference_params = None

        logger.info(
            f"UnifiedGlobalMamba: d_input={d_input}, d_model={d_model}, layers={num_layers}, "
            f"d_state={d_state}, d_conv={d_conv}, expand={expand}, headdim={headdim}, "
            f"nheads={nheads}"
        )

    @staticmethod
    def _find_valid_mamba_dim(d_input, expand, headdim):
        """Find nearest d_model >= d_input satisfying Mamba2 nheads constraint."""
        # Constraint: (d_model * expand / headdim) % 8 == 0
        # → d_model must be multiple of (headdim * 8 / expand)
        unit = headdim * 8 // expand
        d_model = ((d_input + unit - 1) // unit) * unit
        return d_model

    def start_new_sequence(self):
        """Reset hidden state for new video sequence (inference mode)."""
        self.inference_params = InferenceParams(
            max_seqlen=self.max_seqlen,
            max_batch_size=self.max_batch_size
        )

    def forward(self, x):
        """
        Batch forward for training.

        Args:
            x: [B, T, d_input] concatenated CLS + GAP tokens

        Returns:
            out: [B, T, d_input] temporally refined tokens
        """
        B, T, D = x.shape
        assert D == self.d_input, f"Expected d_input={self.d_input}, got {D}"

        # Project to Mamba2-valid dimension if needed
        if self.input_proj is not None:
            x = self.input_proj(x)

        for block in self.blocks:
            x = block(x, inference_params=None)

        # Project back to original dimension
        if self.output_proj is not None:
            x = self.output_proj(x)

        return x

    def forward_single_frame(self, x):
        """
        Single-frame forward for streaming inference.

        Args:
            x: [B, 1, d_input] single frame token

        Returns:
            out: [B, 1, d_input] refined token
        """
        if self.inference_params is None:
            self.start_new_sequence()

        # Project to Mamba2-valid dimension if needed
        if self.input_proj is not None:
            x = self.input_proj(x)

        for block in self.blocks:
            x = block(x, inference_params=self.inference_params)

        # Project back to original dimension
        if self.output_proj is not None:
            x = self.output_proj(x)

        self.inference_params.seqlen_offset += x.shape[1]
        return x


class OnepieceMetricHead(nn.Module):
    """
    Metric Head with dual output paths:
        Path A: Scale/Shift prediction from refined global token (1280-dim)
        Path B: FiLM spatial guidance from refined GAP position (256-dim)

    Scale: softplus(raw_scale) → always positive
    Shift: 0.1 * sigmoid(raw_shift) → range [0, 0.1]
    FiLM: gamma = 1 + gamma_raw (residual), beta = beta_raw
    """

    def __init__(self, input_dim=1280, dpt_dim=256, hidden_dim=512):
        super().__init__()

        self.input_dim = input_dim
        self.dpt_dim = dpt_dim

        # Path A: Scale/Shift prediction
        self.scale_shift_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # [raw_scale, raw_shift]
        )

        # Path B: FiLM generator (from GAP-position slice of refined global)
        self.film_generator = nn.Sequential(
            nn.Linear(dpt_dim, dpt_dim),
            nn.ReLU(),
            nn.Linear(dpt_dim, dpt_dim * 2)  # [gamma, beta] each dpt_dim
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize for stable training start."""
        # Xavier init for all linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Scale/Shift initialization
        with torch.no_grad():
            # Scale bias: softplus(x) ≈ 1.0 when x ≈ 0.5413
            # Start with scale ≈ 1.0
            self.scale_shift_mlp[-1].bias.data[0] = 0.5413  # softplus(0.5413) ≈ 1.0

            # Shift bias: 0.1 * sigmoid(x) = 0 when x → -inf, but we want ~0
            # sigmoid(-5) ≈ 0.0067, so 0.1 * 0.0067 ≈ 0.0007 ≈ 0
            self.scale_shift_mlp[-1].bias.data[1] = -5.0

        # FiLM zero-init: gamma=0 → 1+0=1 (identity), beta=0
        with torch.no_grad():
            nn.init.zeros_(self.film_generator[-1].weight)
            nn.init.zeros_(self.film_generator[-1].bias)

        logger.info("OnepieceMetricHead initialized: scale≈1.0, shift≈0.0, FiLM=identity")

    def forward(self, refined_global):
        """
        Args:
            refined_global: [B*T, 1280] refined global tokens from UnifiedGlobalMamba

        Returns:
            scale: [B*T, 1] positive scale values
            shift: [B*T, 1] shift values in [0, 0.1]
            gamma: [B*T, dpt_dim] FiLM multiplicative factor (centered at 1)
            beta: [B*T, dpt_dim] FiLM additive factor
        """
        # Path A: Scale/Shift
        scale_shift = self.scale_shift_mlp(refined_global)  # [B*T, 2]
        raw_scale, raw_shift = scale_shift[:, 0:1], scale_shift[:, 1:2]

        scale = F.softplus(raw_scale)  # Always positive
        shift = 0.1 * torch.sigmoid(raw_shift)  # Range [0, 0.1]

        # Path B: FiLM from GAP-position slice (last 256 dims of 1280)
        refined_gap = refined_global[:, -self.dpt_dim:]  # [B*T, 256]
        film_out = self.film_generator(refined_gap)  # [B*T, 512]
        gamma_raw, beta = film_out.chunk(2, dim=-1)  # Each [B*T, 256]

        gamma = 1.0 + gamma_raw  # Residual: identity when gamma_raw=0

        return scale, shift, gamma, beta


class SceneCutDetector(nn.Module):
    """
    Scene Cut Detector using CLS token cosine distance.

    D_cls = 1 - cos_sim(CLS_t, CLS_{t-1})
    W_temporal = 1 - sigmoid(k * (D_cls - tau))

    When D_cls < tau: W_temporal ≈ 1.0 (same scene, keep temporal loss)
    When D_cls > tau: W_temporal ≈ 0.0 (scene cut, suppress temporal loss)

    Applied to TGM loss and Feature Consistency loss (NOT Log L1, which is per-frame).
    """

    def __init__(self, tau=0.05, k=80):
        super().__init__()
        self.tau = tau
        self.k = k

    @torch.no_grad()
    def forward(self, cls_tokens):
        """
        Args:
            cls_tokens: [B, T, 1024] CLS tokens per frame

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
        cls_t = F.normalize(cls_tokens[:, 1:], dim=-1)    # [B, T-1, D]
        cls_t_prev = F.normalize(cls_tokens[:, :-1], dim=-1)  # [B, T-1, D]

        cos_sim = (cls_t * cls_t_prev).sum(dim=-1)  # [B, T-1]
        d_cls = 1.0 - cos_sim  # Cosine distance [0, 2]

        # Soft thresholding
        temporal_weights = 1.0 - torch.sigmoid(self.k * (d_cls - self.tau))  # [B, T-1]

        return temporal_weights, d_cls
