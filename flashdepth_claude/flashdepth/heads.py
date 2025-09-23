import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalScalePredictor(nn.Module):
    """
    Global Scale Predictor (GSP) Head for converting relative depth to metric depth.

    This head takes the [CLS] token feature vector from DINOv2 encoder and predicts
    global scale and shift parameters to transform relative depth to metric depth.
    FlashDepth outputs are in the range of 100/gt_depth, so scale should be positive.

    Architecture: Linear(1024 -> 256) -> ReLU -> Linear(256 -> 2)
    Output: [scale, shift] where scale > 0 (enforced by Softplus activation), shift can be negative
    """

    def __init__(self, input_dim=1024, hidden_dim=256):
        super(GlobalScalePredictor, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # MLP architecture
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # Output: [scale, shift]
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for stable training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform initialization
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # Initialize the final layer to output reasonable scale and shift
        # Scale should be around 1.0, shift around 0.0 initially
        with torch.no_grad():
            self.mlp[-1].weight.data.fill_(0.1)
            self.mlp[-1].bias.data[0] = 1.0  # scale
            self.mlp[-1].bias.data[1] = 0.0  # shift

    def forward(self, cls_token):
        """
        Forward pass of GSP head

        Args:
            cls_token: [CLS] token feature vector from DINOv2 encoder
                      Shape: (batch_size, input_dim)

        Returns:
            scale: Global scale parameter (positive, ensured by Softplus activation)
                   Shape: (batch_size, 1)
            shift: Global shift parameter (can be negative)
                   Shape: (batch_size, 1)
        """
        # Ensure input has correct shape
        if cls_token.dim() == 1:
            cls_token = cls_token.unsqueeze(0)

        assert cls_token.shape[-1] == self.input_dim, \
            f"Expected input dimension {self.input_dim}, got {cls_token.shape[-1]}"

        # Forward through MLP
        output = self.mlp(cls_token)  # Shape: (batch_size, 2)

        # Split scale and shift
        scale_raw, shift = output[:, 0:1], output[:, 1:2]  # Each: (batch_size, 1)

        # Apply Softplus to ensure scale is positive (depth must be positive)
        # Since we're converting from 100/gt_depth to metric depth, scale should be positive
        scale = F.softplus(scale_raw) + 1e-8  # Add small epsilon to avoid exactly zero

        return scale, shift

    def predict_metric_depth(self, relative_depth, scale, shift):
        """
        Convert relative depth to metric depth using predicted scale and shift

        Args:
            relative_depth: Relative depth map from FlashDepth
                           Shape: (batch_size, height, width)
            scale: Global scale parameter
                   Shape: (batch_size, 1) or (batch_size, 1, 1)
            shift: Global shift parameter
                   Shape: (batch_size, 1) or (batch_size, 1, 1)

        Returns:
            metric_depth: Metric depth map in meters
                         Shape: (batch_size, height, width)
        """
        # Ensure scale and shift have correct dimensions for broadcasting
        if scale.dim() == 2:
            scale = scale.unsqueeze(-1)  # (batch_size, 1, 1)
        if shift.dim() == 2:
            shift = shift.unsqueeze(-1)  # (batch_size, 1, 1)

        # FlashDepth relative depth is trained as inverse_gt_depth * 100
        # So we need to: relative_depth / 100 -> 1/value -> apply scale/shift
        # D_metric = scale * (1 / (relative_depth / 100)) + shift
        # But we need to handle division by zero
        inverse_depth = relative_depth / 100.0
        depth_from_relative = 1.0 / (inverse_depth + 1e-8)
        metric_depth = scale * depth_from_relative + shift

        return metric_depth


class MetricDepthLoss(nn.Module):
    """
    Loss function for training the Global Scale Predictor
    """

    def __init__(self, loss_type='log_l1'):
        super(MetricDepthLoss, self).__init__()
        self.loss_type = loss_type

        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'log_l1':
            self.loss_fn = self._log_l1_loss
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def _log_l1_loss(self, pred, gt):
        """
        Log L1 loss function for depth estimation

        Args:
            pred: Predicted depth values
            gt: Ground truth depth values

        Returns:
            Log L1 loss
        """
        return F.l1_loss(torch.log(pred + 1e-8), torch.log(gt + 1e-8))

    def forward(self, pred_metric_depth, gt_metric_depth, valid_mask=None):
        """
        Compute loss between predicted and ground truth metric depth

        Args:
            pred_metric_depth: Predicted metric depth
                              Shape: (batch_size, height, width)
            gt_metric_depth: Ground truth metric depth
                            Shape: (batch_size, height, width)
            valid_mask: Mask of valid depth values (optional)
                       Shape: (batch_size, height, width)

        Returns:
            loss: Scalar loss value
        """
        # Use valid mask if provided, otherwise use all pixels
        if valid_mask is not None:
            valid_pred = pred_metric_depth[valid_mask]
            valid_gt = gt_metric_depth[valid_mask]
        else:
            valid_pred = pred_metric_depth.flatten()
            valid_gt = gt_metric_depth.flatten()

        if valid_pred.numel() == 0:
            return torch.tensor(0.0, device=pred_metric_depth.device, requires_grad=True)

        loss = self.loss_fn(valid_pred, valid_gt)
        return loss