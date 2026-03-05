"""
Onepiece V2 Core Modules for FlashDepth-Metric.

Architecture:
    DPT features → GAP(256) + GStdP(256) → Unified Global Mamba(512) →
    FiLMGenerator → gamma, beta → modulate DPT features
    MetricHead (Conv) → scale, shift from modulated features

Components:
    1. UnifiedGlobalMamba: 4-layer Mamba2 on GAP+GStdP tokens (512-dim)
    2. OnepieceFiLMGenerator: FiLM parameter generator from Mamba output
    3. OnepieceMetricHead: Conv-based scale/shift prediction from spatial features
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
    Unified Global Mamba: Temporal processing on GAP + GStdP concatenated tokens.

    ViT-L: GAP(256) + GStdP(256) = 512-dim → d_model=512 (valid for Mamba2)

    Uses 4 MambaBlock layers with expand=2, headdim=64.
    Constraint: d_model * expand / headdim must be multiple of 8.

    When d_input != d_model, input/output projection layers are added automatically.

    Two forward modes:
        - forward(x): Batch mode for training (parallel scan on full [B, T, D])
        - forward_single_frame(x): Streaming mode for inference (per-frame with hidden state)
    """

    def __init__(self, d_input, num_layers=4, d_state=64, d_conv=4,
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
            x: [B, T, d_input] concatenated GAP + GStdP tokens

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


class OnepieceFiLMGenerator(nn.Module):
    """
    FiLM parameter generator from Mamba output.

    Linear(mamba_dim, dpt_dim) → ReLU → Linear(dpt_dim, dpt_dim*2)
    Zero-init last layer for identity start (gamma=1, beta=0).
    """

    def __init__(self, mamba_dim=512, dpt_dim=256):
        super().__init__()
        self.mamba_dim = mamba_dim
        self.dpt_dim = dpt_dim

        self.net = nn.Sequential(
            nn.Linear(mamba_dim, dpt_dim),
            nn.ReLU(),
            nn.Linear(dpt_dim, dpt_dim * 2)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Zero-init last layer: gamma_raw=0 → gamma=1+0=1, beta=0."""
        with torch.no_grad():
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        logger.info("OnepieceFiLMGenerator initialized: FiLM=identity (gamma=1, beta=0)")

    def forward(self, mamba_output):
        """
        Args:
            mamba_output: [B*T, mamba_dim] refined global tokens

        Returns:
            gamma: [B*T, dpt_dim] multiplicative factor (centered at 1)
            beta: [B*T, dpt_dim] additive factor
        """
        film_out = self.net(mamba_output)  # [B*T, dpt_dim*2]
        gamma_raw, beta = film_out.chunk(2, dim=-1)  # Each [B*T, dpt_dim]
        gamma = 1.0 + gamma_raw  # Residual: identity when gamma_raw=0
        return gamma, beta


class OnepieceMetricHead(nn.Module):
    """
    Conv-based scale/shift prediction from spatial features.

    Conv2d(dpt_dim, hidden_dim, 1) → ReLU → Conv2d(hidden_dim, 2, 1)
    → spatial mean → softplus(scale), shift

    Modes:
        - "metric": scale≈100, shift=sigmoid(raw)*1.0 ∈ [0, 1.0]
        - "inverse": scale≈1.0, shift=raw (unconstrained, gear5 style)
    """

    def __init__(self, dpt_dim=256, hidden_dim=64, train_mode="metric"):
        super().__init__()
        self.dpt_dim = dpt_dim
        self.train_mode = train_mode

        self.conv = nn.Sequential(
            nn.Conv2d(dpt_dim, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 2, 1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize for stable training start."""
        # Xavier init for all conv layers
        for module in self.conv.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Mode-specific initialization on last conv
        with torch.no_grad():
            if self.train_mode == "inverse":
                # Channel 0 = scale: softplus(0.5) ≈ 1.0
                self.conv[-1].bias.data[0] = 0.5
                # Channel 1 = shift: raw=0.0 (unconstrained)
                self.conv[-1].bias.data[1] = 0.0
                logger.info("OnepieceMetricHead initialized (inverse): scale≈1.0, shift≈0.0")
            else:
                # Channel 0 = scale: softplus(100.0) ≈ 100.0 (softplus(x)≈x for x>>1)
                self.conv[-1].bias.data[0] = 100.0
                # Channel 1 = shift: 1.0 * sigmoid(-5) ≈ 0.007 ≈ 0
                self.conv[-1].bias.data[1] = -5.0
                logger.info("OnepieceMetricHead initialized (metric): scale≈100.0, shift≈0.0")

    def forward(self, modulated_features):
        """
        Args:
            modulated_features: [B*T, dpt_dim, h, w] FiLM-modulated DPT features

        Returns:
            scale: [B*T, 1] positive scale values
            shift: [B*T, 1] shift values
        """
        out = self.conv(modulated_features)  # [B*T, 2, h, w]
        out = out.mean(dim=(-2, -1))  # [B*T, 2] spatial mean
        raw_scale, raw_shift = out[:, 0:1], out[:, 1:2]

        scale = F.softplus(raw_scale).clamp(max=1000.0)  # Positive, capped at 1000

        if self.train_mode == "inverse":
            shift = raw_shift  # Unconstrained (gear5 style)
        else:
            shift = 1.0 * torch.sigmoid(raw_shift)  # Range [0, 1.0]

        return scale, shift
