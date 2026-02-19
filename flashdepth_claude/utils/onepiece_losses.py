"""
Onepiece V2 Loss Functions.

4-loss system with gradient isolation:
    - L_log_l1: MetricHead gradient only (via detached modulated features)
    - L_tgm: Full graph (Mamba + FiLM + both heads)
    - L_wfc: FiLM→Mamba gradient (Phase 2 only)
    - L_ssil: RelativeHead gradient only (Phase 2 only, via detached modulated features)

Phase 1: L_total = L_log_l1 + L_tgm
Phase 2: L_total = L_log_l1 + L_tgm + L_wfc + L_ssil
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from .gear_losses import LogL1Loss, TGMTemporalLoss
from flashdepth.util.loss import ScaleAndShiftInvariantLoss

logger = logging.getLogger(__name__)


class WarpFeatureConsistencyLoss(nn.Module):
    """
    Warp-based Feature Consistency Loss.

    Uses optical flow to warp modulated features from frame t-1 to frame t,
    then computes confidence-weighted L2 distance.

    L_feat = mean(confidence * ||feat_t - warp(feat_{t-1}, flow)||^2)

    Features: modulated DPT features [B*T, 256, h, w] at full resolution.
    Flow: Computed on original images, resized to match feature resolution.
    Flow confidence is used internally only (not passed to other losses).
    """

    def __init__(self):
        super().__init__()

    def forward(self, modulated_features, images, flow_estimator):
        """
        Args:
            modulated_features: [B, T, 256, h, w] FiLM-modulated DPT features
            images: [B, T, 3, H, W] original video frames (0-1 normalized)
            flow_estimator: FlowEstimator instance (frozen Sea-RAFT)

        Returns:
            loss: scalar feature consistency loss
        """
        B, T, C, h, w = modulated_features.shape

        if T < 2:
            return torch.tensor(0.0, device=modulated_features.device)

        feat_h, feat_w = h, w

        # Compute optical flow on original images
        # flow: [B, T-1, 2, H, W], confidence: [B, T-1, 1, H, W]
        flows, confidences = flow_estimator.estimate_flow_batch(images)

        # Resize flow to feature resolution
        _, T_minus_1, _, H_img, W_img = flows.shape
        flows_resized = F.interpolate(
            flows.view(B * T_minus_1, 2, H_img, W_img),
            size=(feat_h, feat_w),
            mode='bilinear',
            align_corners=True
        )
        # Scale flow values to match new resolution
        flows_resized[:, 0] *= feat_w / W_img
        flows_resized[:, 1] *= feat_h / H_img
        flows_resized = flows_resized.view(B, T_minus_1, 2, feat_h, feat_w)

        # Resize confidence
        confidences_resized = F.interpolate(
            confidences.view(B * T_minus_1, 1, H_img, W_img),
            size=(feat_h, feat_w),
            mode='bilinear',
            align_corners=True
        )
        confidences_resized = confidences_resized.view(B, T_minus_1, 1, feat_h, feat_w)

        total_loss = 0.0
        num_pairs = 0

        for t in range(T - 1):
            # Features: current and previous
            feat_t = modulated_features[:, t + 1]    # [B, C, feat_h, feat_w]
            feat_prev = modulated_features[:, t]      # [B, C, feat_h, feat_w]

            # Flow from t to t+1 (warp previous to current)
            flow = flows_resized[:, t]         # [B, 2, feat_h, feat_w]
            conf = confidences_resized[:, t]   # [B, 1, feat_h, feat_w]

            # Create sampling grid: grid + flow
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(-1, 1, feat_h, device=flow.device),
                torch.linspace(-1, 1, feat_w, device=flow.device),
                indexing='ij'
            )
            grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

            # Convert flow to normalized coordinates
            flow_norm = torch.zeros_like(flow)
            flow_norm[:, 0] = flow[:, 0] / (feat_w / 2.0)
            flow_norm[:, 1] = flow[:, 1] / (feat_h / 2.0)
            flow_norm = flow_norm.permute(0, 2, 3, 1)

            # Warp previous features using flow
            warp_grid = grid + flow_norm
            warped_feat = F.grid_sample(
                feat_prev, warp_grid,
                mode='bilinear', padding_mode='border', align_corners=True
            )

            # Confidence-weighted L2 loss
            diff = (feat_t - warped_feat) ** 2
            weighted_diff = conf * diff.mean(dim=1, keepdim=True)

            pair_loss = weighted_diff.mean()
            total_loss = total_loss + pair_loss
            num_pairs += 1

        if num_pairs == 0:
            return torch.tensor(0.0, device=modulated_features.device)

        return total_loss / num_pairs


class OnepieceCombinedLoss(nn.Module):
    """
    Combined loss for Onepiece V2 training.

    Phase 1: L_total = w1 * L_log_l1 + w2 * L_tgm
    Phase 2: L_total = w1 * L_log_l1 + w2 * L_tgm + w3 * L_wfc + w4 * L_ssil

    Gradient isolation is handled by the forward pass providing separate outputs:
        - metric_depth_isolated: gradient flows through MetricHead only
        - relative_depth_isolated: gradient flows through RelativeHead only
    """

    def __init__(self, log_l1_weight=1.0, tgm_weight=1.0,
                 wfc_weight=0.01, ssil_weight=1.0, use_log_space=True):
        super().__init__()

        self.log_l1_weight = log_l1_weight
        self.tgm_weight = tgm_weight
        self.wfc_weight = wfc_weight
        self.ssil_weight = ssil_weight

        self.log_l1_loss = LogL1Loss(use_log_space=use_log_space)
        self.tgm_loss = TGMTemporalLoss(use_log_space=use_log_space)
        self.wfc_loss = WarpFeatureConsistencyLoss()
        self.ssil_loss = ScaleAndShiftInvariantLoss()

        logger.info(
            f"OnepieceCombinedLoss V2: log_l1={log_l1_weight}, "
            f"tgm={tgm_weight}, wfc={wfc_weight}, ssil={ssil_weight}"
        )

    def forward(self,
                # Full-graph outputs (for TGM)
                metric_depth, gt_depth, valid_mask,
                # Isolated outputs (for LogL1, SSIL)
                metric_depth_isolated,
                relative_depth_isolated=None,
                gt_depth_for_ssil=None,
                # WFC inputs
                modulated_features=None, images=None, flow_estimator=None,
                phase=1, return_components=True):
        """
        Compute combined loss with gradient isolation.

        Args:
            metric_depth: [B, T, H, W] full-graph metric depth (for TGM)
            gt_depth: [B, T, H, W] ground truth (inverse depth * 100)
            valid_mask: [B, T, H, W] validity mask
            metric_depth_isolated: [B, T, H, W] MetricHead-only graph (for LogL1)
            relative_depth_isolated: [B, T, H, W] RelativeHead-only graph (for SSIL, Phase 2)
            gt_depth_for_ssil: [B, T, H, W] GT inverse depth for SSIL
            modulated_features: [B, T, 256, h, w] for WFC (Phase 2)
            images: [B, T, 3, H, W] original images for flow estimation
            flow_estimator: FlowEstimator instance
            phase: 1 or 2
            return_components: If True, return individual loss components

        Returns:
            total_loss: Combined loss scalar
            components: dict of individual losses (if return_components=True)
        """
        # 1. Log L1 (always, MetricHead gradient only)
        l_logl1 = self.log_l1_loss(metric_depth_isolated, gt_depth, valid_mask)

        # 2. TGM (always, full gradient)
        if self.tgm_weight > 0 and metric_depth.shape[1] >= 2:
            l_tgm = self.tgm_loss(metric_depth, gt_depth, valid_mask)
        else:
            l_tgm = torch.tensor(0.0, device=metric_depth.device)

        # 3. WFC (Phase 2 only, FiLM→Mamba gradient)
        if (phase == 2 and self.wfc_weight > 0 and
                modulated_features is not None and
                images is not None and
                flow_estimator is not None and
                metric_depth.shape[1] >= 2):
            l_wfc = self.wfc_loss(modulated_features, images, flow_estimator)
        else:
            l_wfc = torch.tensor(0.0, device=metric_depth.device)

        # 4. SSIL (Phase 2 only, RelativeHead gradient only)
        if (phase == 2 and self.ssil_weight > 0 and
                relative_depth_isolated is not None and
                gt_depth_for_ssil is not None):
            # Reshape [B, T, H, W] → [B*T, H, W] for SSIL (expects 3D input)
            # Also convert mask to bool (SSIL uses boolean indexing internally)
            rel_ssil = relative_depth_isolated.reshape(-1, *relative_depth_isolated.shape[2:])
            gt_ssil = gt_depth_for_ssil.reshape(-1, *gt_depth_for_ssil.shape[2:])
            mask_ssil = valid_mask.reshape(-1, *valid_mask.shape[2:]).bool() if valid_mask is not None else None
            l_ssil = self.ssil_loss(rel_ssil, gt_ssil, mask=mask_ssil)
        else:
            l_ssil = torch.tensor(0.0, device=metric_depth.device)

        # Combined loss
        total_loss = (
            self.log_l1_weight * l_logl1 +
            self.tgm_weight * l_tgm
        )
        if phase == 2:
            total_loss = total_loss + self.wfc_weight * l_wfc + self.ssil_weight * l_ssil

        if return_components:
            components = {
                'log_l1_loss': l_logl1.item(),
                'tgm_loss': l_tgm.item(),
                'wfc_loss': l_wfc.item() if torch.is_tensor(l_wfc) else l_wfc,
                'ssil_loss': l_ssil.item() if torch.is_tensor(l_ssil) else l_ssil,
            }
            return total_loss, components

        return total_loss
