import torch
import torch.nn as nn


class GlobalScalePredictor(nn.Module):
    """Predicts global scale and shift factors from ViT CLS embeddings."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, eps: float = 1e-6) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )
        self.softplus = nn.Softplus()
        self.eps = eps

    def forward(self, cls_embedding: torch.Tensor) -> torch.Tensor:
        """Returns a tensor of shape (B, 2) containing (scale, shift)."""
        scale_shift = self.mlp(cls_embedding)
        scale, shift = scale_shift.split(1, dim=-1)
        scale = self.softplus(scale) + self.eps
        return torch.cat((scale, shift), dim=-1)
