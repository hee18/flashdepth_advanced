"""
Loss functions for Gear training.

This module provides unified loss classes used across train_gear2.py, train_gear3.py,
and train_gear3_upgrade.py to reduce code duplication.

Loss classes:
    - LogL1Loss: Base inverse depth loss (used by all Gear variants)
    - DepthVariancePseudoLabelLoss: Importance map loss (Gear2 only)
    - EdgeAwareLoss: Edge alignment loss (Gear2 only)
    - ContrastiveFGBGLoss: FG/BG feature contrastive loss (Gear2 only)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogL1Loss(nn.Module):
    """
    Log L1 loss for inverse depth learning.

    Loss = L1(log(pred_inverse), log(gt_inverse))
    Works directly with inverse depth values (100/m)
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred_inverse, gt_inverse, valid_mask=None):
        """
        Args:
            pred_inverse: [B, 1, H, W] predicted inverse depth (100/m)
            gt_inverse: [B, 1, H, W] ground truth inverse depth (100/m)
            valid_mask: [B, 1, H, W] valid pixels (optional)

        Returns:
            loss: scalar
        """
        # Apply valid mask BEFORE log to avoid log(negative values)
        if valid_mask is not None:
            # Only compute loss on valid pixels
            pred_valid = pred_inverse[valid_mask.bool()]
            gt_valid = gt_inverse[valid_mask.bool()]

            if len(pred_valid) == 0:
                return torch.tensor(0.0, device=pred_inverse.device)

            # Clamp to positive values to prevent NaN from log(negative) or log(0)
            # Critical: shift in Gear5 can be negative, making predictions negative
            epsilon = 1e-8
            pred_valid = torch.clamp(pred_valid, min=epsilon)
            gt_valid = torch.clamp(gt_valid, min=epsilon)

            # Log L1 loss on valid pixels only
            loss = F.l1_loss(
                torch.log(pred_valid + epsilon),
                torch.log(gt_valid + epsilon),
                reduction='mean'
            )
        else:
            # Fallback: compute on all pixels
            epsilon = 1e-8
            pred_clamped = torch.clamp(pred_inverse, min=epsilon)
            gt_clamped = torch.clamp(gt_inverse, min=epsilon)
            loss = F.l1_loss(
                torch.log(pred_clamped + epsilon),
                torch.log(gt_clamped + epsilon),
                reduction='mean'
            )

        return loss


class DepthVariancePseudoLabelLoss(nn.Module):
    """
    Depth Variance Pseudo-Label Loss for importance maps.

    Uses local depth variance as pseudo-label (supervision) for importance map.

    High variance regions (complex geometry) → High importance
    Low variance regions (flat surfaces) → Low importance

    This encourages importance map to have spatial diversity (high std).

    **CRITICAL**: GT depth variance is computed with torch.no_grad() to prevent gradient flow.
    """
    def __init__(self, kernel_size=15, sigma=3.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

        # Create Gaussian kernel for weighted variance computation
        self.register_buffer('gaussian_kernel', self._create_gaussian_kernel(kernel_size, sigma))

    def _create_gaussian_kernel(self, kernel_size, sigma):
        """Create 2D Gaussian kernel for weighted variance"""
        # Create 1D Gaussian
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))

        # Normalize to sum to 1
        kernel = kernel / kernel.sum()

        # Reshape for conv2d: [1, 1, kernel_size, kernel_size]
        return kernel.view(1, 1, kernel_size, kernel_size)

    def compute_local_variance(self, depth_map):
        """
        Compute local variance using Gaussian-weighted window.

        Variance = E[x²] - E[x]²

        Args:
            depth_map: [B, 1, H, W] depth values (inverse depth in 100/m scale)

        Returns:
            variance: [B, 1, H, W] local variance map
        """
        # Match dtype and device
        kernel = self.gaussian_kernel.to(dtype=depth_map.dtype, device=depth_map.device)
        padding = self.kernel_size // 2

        # E[x] (local mean)
        local_mean = F.conv2d(depth_map, kernel, padding=padding)

        # E[x²] (local mean of squares)
        local_mean_sq = F.conv2d(depth_map**2, kernel, padding=padding)

        # Variance = E[x²] - E[x]²
        variance = local_mean_sq - local_mean**2

        # Clamp to avoid negative values due to numerical errors
        return variance.clamp(min=0)

    def forward(self, importance_map, depth_map, valid_mask=None):
        """
        Args:
            importance_map: [B, 1, H, W] predicted importance in range [0, 1]
            depth_map: [B, 1, H, W] GT depth (inverse depth in 100/m scale)
            valid_mask: [B, 1, H, W] valid pixels (optional)

        Returns:
            loss: scalar L1 distance between importance and normalized variance
        """
        # CRITICAL: Compute variance WITHOUT gradient to GT depth
        with torch.no_grad():
            # Compute local variance from GT depth
            variance = self.compute_local_variance(depth_map)  # [B, 1, H, W]

            # Normalize to [0, 1] range (min-max normalization)
            if valid_mask is not None:
                # Only consider valid pixels for normalization
                variance_valid = variance[valid_mask.bool()]
                if len(variance_valid) > 0:
                    var_min = variance_valid.min()
                    var_max = variance_valid.max()
                else:
                    var_min = variance.min()
                    var_max = variance.max()
            else:
                var_min = variance.min()
                var_max = variance.max()

            # Avoid division by zero
            variance_range = var_max - var_min + 1e-8
            variance_norm = (variance - var_min) / variance_range  # [0, 1]

        # L1 loss: importance_map (trainable) vs variance_norm (pseudo-label)
        if valid_mask is not None:
            loss = F.l1_loss(
                importance_map[valid_mask.bool()],
                variance_norm[valid_mask.bool()]
            )
        else:
            loss = F.l1_loss(importance_map, variance_norm)

        return loss


class EdgeAwareLoss(nn.Module):
    """
    Edge-aware loss for importance maps.

    Aligns importance map edges with depth edges, ensuring that:
    - FG/BG boundaries coincide with depth discontinuities
    - Interior regions remain smooth
    - Prevents noisy importance maps

    Uses Sobel filter to compute gradients.

    Reference: "Edge-Guided Depth Estimation" (CVPR 2024)
    """
    def __init__(self):
        super().__init__()

        # Sobel kernels for edge detection (fixed, non-trainable)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def compute_edges(self, tensor):
        """
        Compute edge magnitude using Sobel filter.

        Args:
            tensor: [B, 1, H, W]

        Returns:
            edges: [B, 1, H, W] edge magnitude
        """
        # Match dtype and device of input tensor (handles BFloat16)
        sobel_x = self.sobel_x.to(dtype=tensor.dtype, device=tensor.device)
        sobel_y = self.sobel_y.to(dtype=tensor.dtype, device=tensor.device)

        grad_x = F.conv2d(tensor, sobel_x, padding=1)
        grad_y = F.conv2d(tensor, sobel_y, padding=1)

        # Edge magnitude
        edges = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        return edges

    def forward(self, importance_map, depth_map):
        """
        Args:
            importance_map: [B, 1, H, W] importance values in range [0, 1]
            depth_map: [B, 1, H, W] depth values (inverse depth in 100/m scale)

        Returns:
            loss: scalar (L1 distance between edges)
        """
        # Compute edges
        importance_edges = self.compute_edges(importance_map)
        depth_edges = self.compute_edges(depth_map)

        # Normalize edges to [0, 1] for fair comparison
        importance_edges = importance_edges / (importance_edges.max() + 1e-8)
        depth_edges = depth_edges / (depth_edges.max() + 1e-8)

        # L1 loss between edge maps
        return F.l1_loss(importance_edges, depth_edges)


class ContrastiveFGBGLoss(nn.Module):
    """
    Contrastive loss for FG/BG features.

    Encourages FG and BG features to be different in embedding space.
    Based on InfoNCE loss: maximize distance between FG and BG features.

    This ensures that modulation parameters (γ_fg, β_fg, γ_bg, β_bg) are distinct,
    leading to effective spatial modulation.

    Reference: "Foreground-Aware Feature Contrast (FAC++)" (CVPR 2024)
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, fg_features, bg_features):
        """
        Args:
            fg_features: [B, feature_dim] foreground features
            bg_features: [B, feature_dim] background features

        Returns:
            loss: scalar (negative cosine similarity, to maximize distance)
        """
        B = fg_features.shape[0]

        # Normalize features to unit sphere
        fg_norm = F.normalize(fg_features, dim=1)  # [B, feature_dim]
        bg_norm = F.normalize(bg_features, dim=1)  # [B, feature_dim]

        # Compute cosine similarity for same batch indices
        # We want FG[i] and BG[i] to be DIFFERENT (low similarity)
        similarity = (fg_norm * bg_norm).sum(dim=1)  # [B]

        # Average over batch
        avg_similarity = similarity.mean()

        # Maximize distance = minimize similarity
        # Add temperature scaling for numerical stability
        return avg_similarity / self.temperature
