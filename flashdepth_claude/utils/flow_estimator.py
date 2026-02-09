"""
Sea-RAFT Optical Flow Estimator Wrapper for Onepiece training.

Wraps Sea-RAFT (https://github.com/princeton-vl/SEA-RAFT) for optical flow estimation.
Used ONLY during training for Feature Consistency Loss computation.
NOT needed during inference/testing.

Requirements:
    - Sea-RAFT must be cloned to third_party/SEA-RAFT/
    - Pretrained weights must be available
    - NO fallback: ImportError is raised if Sea-RAFT is unavailable
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class FlowEstimator(nn.Module):
    """
    Frozen Sea-RAFT wrapper for optical flow estimation.

    Always runs in eval mode with torch.no_grad().
    Raises ImportError if Sea-RAFT is not installed.
    """

    def __init__(self, checkpoint_path, device='cuda'):
        super().__init__()

        # Add Sea-RAFT to path
        sea_raft_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      'third_party', 'SEA-RAFT')
        if not os.path.exists(sea_raft_path):
            raise ImportError(
                f"Sea-RAFT not found at {sea_raft_path}. "
                f"Please clone it: git clone https://github.com/princeton-vl/SEA-RAFT.git {sea_raft_path}"
            )

        # Sea-RAFT's core/ modules have been patched to use package imports
        # (e.g., "from core.utils.utils import ..." instead of "from utils.utils import ...")
        # so we add the SEA-RAFT root (not core/) to sys.path.
        if sea_raft_path not in sys.path:
            sys.path.insert(0, sea_raft_path)

        try:
            from core.raft import RAFT
        except ImportError as e:
            raise ImportError(
                f"Failed to import Sea-RAFT modules. "
                f"Ensure Sea-RAFT is properly installed at {sea_raft_path}. "
                f"Original error: {e}"
            )

        # Verify checkpoint exists
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Sea-RAFT checkpoint not found at {checkpoint_path}. "
                f"Download pretrained weights to this path."
            )

        # Sea-RAFT configuration (spring-M config)
        # Use argparse.Namespace instead of easydict to avoid extra dependency
        from argparse import Namespace
        args = Namespace(
            use_var=True,
            var_min=0,
            var_max=10,
            pretrain='resnet34',
            initial_dim=64,
            block_dims=[64, 128, 256],
            radius=4,
            dim=128,
            num_blocks=2,
            iters=4,
        )

        # Load model
        self.model = RAFT(args)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if present
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(device)
        self.model.eval()

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.device = device
        logger.info(f"Sea-RAFT loaded from {checkpoint_path} (frozen, eval mode)")

    @torch.no_grad()
    def estimate_flow(self, frame_t, frame_t1):
        """
        Estimate optical flow from frame_t to frame_t1.

        Args:
            frame_t: [B, 3, H, W] current frame (RGB, 0-1 normalized)
            frame_t1: [B, 3, H, W] next frame (RGB, 0-1 normalized)

        Returns:
            flow: [B, 2, H, W] optical flow (u, v)
            confidence: [B, 1, H, W] confidence/uncertainty map (higher = more confident)
        """
        # Sea-RAFT expects 0-255 range (internally normalizes to [-1, 1])
        image1 = (frame_t * 255.0).to(self.device)
        image2 = (frame_t1 * 255.0).to(self.device)

        # Sea-RAFT handles padding internally via InputPadder
        result = self.model(image1, image2, iters=4, test_mode=True)

        # result is a dict: {'final': flow, 'flow': [...], 'info': [...], 'nf': None}
        flow = result['final']  # [B, 2, H, W]

        # Extract confidence from info predictions
        # info contains weight(2) + log_b(2) channels
        info = result['info'][-1]  # [B, 4, H, W]
        if info is not None and info.shape[1] >= 2:
            # Use weight channels as confidence proxy (softmax over 2 components)
            weight = info[:, :2]  # [B, 2, H, W]
            # Higher logsumexp of weights = more confident prediction
            confidence = torch.logsumexp(weight, dim=1, keepdim=True)  # [B, 1, H, W]
            # Normalize to [0, 1] via sigmoid
            confidence = torch.sigmoid(confidence)
        else:
            confidence = torch.ones(flow.shape[0], 1, flow.shape[2], flow.shape[3],
                                    device=flow.device)

        return flow, confidence

    @torch.no_grad()
    def estimate_flow_batch(self, frames):
        """
        Estimate optical flow for consecutive frame pairs in a batch.

        Args:
            frames: [B, T, 3, H, W] video frames

        Returns:
            flows: [B, T-1, 2, H, W] optical flows
            confidences: [B, T-1, 1, H, W] confidence maps
        """
        B, T, C, H, W = frames.shape

        flows = []
        confidences = []

        for t in range(T - 1):
            flow, conf = self.estimate_flow(frames[:, t], frames[:, t + 1])
            flows.append(flow)
            confidences.append(conf)

        flows = torch.stack(flows, dim=1)  # [B, T-1, 2, H, W]
        confidences = torch.stack(confidences, dim=1)  # [B, T-1, 1, H, W]

        return flows, confidences
