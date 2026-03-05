"""
Onepiece V3 Loss Functions.

Combined loss: L_total = L_log_l1 + L_tgm + L_ofc (1:1:0.01 default)

Components:
    - L_log_l1: Reuses LogL1Loss from gear_losses (metric depth space)
    - L_tgm: Reuses TGMTemporalLoss from gear_losses
    - L_ofc: OpticalFlowConsistencyLoss on post-Mamba features (DPT + mamba residual)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from .gear_losses import LogL1Loss, TGMTemporalLoss

logger = logging.getLogger(__name__)


class OpticalFlowConsistencyLoss(nn.Module):
    """
    Optical Flow Consistency Loss (OFC).

    Uses optical flow to warp post-Mamba features from frame t-1 to frame t,
    then computes confidence-weighted L2 distance.

    L_ofc = mean(confidence * ||feat_t - warp(feat_{t-1}, flow)||^2)

    Features: post_mamba [B*T, 256, h, w] downsampled to [B*T, 256, h/4, w/4] for efficiency.
    Flow: Computed on original images, resized to match feature resolution.

    Gradient flow: OFC on post_mamba_features = DPT + upsample(final_layer(mamba_out))
        → Gradient → DPT (trainable in Phase 2)
        → Gradient → final_layer → Mamba blocks
    """

    def __init__(self, feature_downsample=4):
        """
        Args:
            feature_downsample: Factor to downsample features for efficiency (default: 4)
        """
        super().__init__()
        self.feature_downsample = feature_downsample

    def forward(self, post_mamba_features, images, flow_estimator, scene_cut_weights=None):
        """
        Args:
            post_mamba_features: [B, T, 256, h, w] post-Mamba features (DPT + mamba residual)
            images: [B, T, 3, H, W] original video frames (0-1 normalized)
            flow_estimator: FlowEstimator instance (frozen Sea-RAFT)
            scene_cut_weights: [B, T-1] temporal weights (optional, unused in V3 training)

        Returns:
            loss: scalar optical flow consistency loss
        """
        B, T, C, h, w = post_mamba_features.shape

        if T < 2:
            return torch.tensor(0.0, device=post_mamba_features.device)

        # Downsample features for efficiency
        feat_h = h // self.feature_downsample
        feat_w = w // self.feature_downsample

        # Reshape for pooling: [B*T, C, h, w]
        feats_flat = post_mamba_features.view(B * T, C, h, w)
        feats_down = F.adaptive_avg_pool2d(feats_flat, (feat_h, feat_w))
        feats_down = feats_down.view(B, T, C, feat_h, feat_w)

        # Compute optical flow on original images
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
            feat_t = feats_down[:, t + 1]    # [B, C, feat_h, feat_w]
            feat_prev = feats_down[:, t]      # [B, C, feat_h, feat_w]

            # Flow from t to t+1 (warp previous to current)
            flow = flows_resized[:, t]         # [B, 2, feat_h, feat_w]
            conf = confidences_resized[:, t]   # [B, 1, feat_h, feat_w]

            # Create sampling grid
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

            # Apply scene cut weight if provided
            if scene_cut_weights is not None:
                weight = scene_cut_weights[:, t].mean()
                pair_loss = pair_loss * weight

            total_loss = total_loss + pair_loss
            num_pairs += 1

        if num_pairs == 0:
            return torch.tensor(0.0, device=post_mamba_features.device)

        return total_loss / num_pairs


class OnepieceCombinedLoss(nn.Module):
    """
    Combined loss for Onepiece V3 training.

    L_total = w1 * L_log_l1 + w2 * L_tgm + w3 * L_ofc

    Default weights: 1:1:0.01
    """

    def __init__(self, log_l1_weight=1.0, tgm_weight=1.0, ofc_weight=1.0,
                 use_log_space=True):
        super().__init__()

        self.log_l1_weight = log_l1_weight
        self.tgm_weight = tgm_weight
        self.ofc_weight = ofc_weight

        self.log_l1_loss = LogL1Loss(use_log_space=use_log_space)
        self.tgm_loss = TGMTemporalLoss(use_log_space=use_log_space)
        self.ofc_loss = OpticalFlowConsistencyLoss()

        logger.info(
            f"OnepieceCombinedLoss: log_l1={log_l1_weight}, "
            f"tgm={tgm_weight}, ofc={ofc_weight}"
        )

    def forward(self, pred_depth, gt_depth, valid_mask=None,
                post_mamba_features=None, images=None, flow_estimator=None,
                scene_cut_weights=None, return_components=True):
        """
        Compute combined loss.

        Args:
            pred_depth: [B, T, H, W] predicted depth (inverse, 100/m)
            gt_depth: [B, T, H, W] ground truth depth (inverse, 100/m)
            valid_mask: [B, T, H, W] validity mask
            post_mamba_features: [B, T, 256, h, w] post-Mamba features (for OFC)
            images: [B, T, 3, H, W] original images (for flow estimation)
            flow_estimator: FlowEstimator instance (for OFC)
            scene_cut_weights: [B, T-1] temporal weights (optional)
            return_components: If True, return individual loss components

        Returns:
            total_loss: Combined loss scalar
            components: dict of individual losses (if return_components=True)
        """
        # 1. Log L1 Loss (per-frame, no scene cut weighting)
        l_log_l1 = self.log_l1_loss(pred_depth, gt_depth, valid_mask)

        # 2. TGM Loss (temporal)
        if self.tgm_weight > 0 and pred_depth.shape[1] >= 2:
            l_tgm = self.tgm_loss(pred_depth, gt_depth, valid_mask,
                                   scene_cut_weights=scene_cut_weights)
        else:
            l_tgm = torch.tensor(0.0, device=pred_depth.device)

        # 3. OFC Loss (needs flow estimator and post-Mamba features)
        if (self.ofc_weight > 0 and
                post_mamba_features is not None and
                images is not None and
                flow_estimator is not None and
                pred_depth.shape[1] >= 2):
            l_ofc = self.ofc_loss(
                post_mamba_features, images, flow_estimator, scene_cut_weights
            )
        else:
            l_ofc = torch.tensor(0.0, device=pred_depth.device)

        # Combined loss
        total_loss = (
            self.log_l1_weight * l_log_l1 +
            self.tgm_weight * l_tgm +
            self.ofc_weight * l_ofc
        )

        if return_components:
            components = {
                'log_l1_loss': l_log_l1.item(),
                'tgm_loss': l_tgm.item(),
                'ofc_loss': l_ofc.item() if torch.is_tensor(l_ofc) else l_ofc,
            }
            return total_loss, components

        return total_loss
