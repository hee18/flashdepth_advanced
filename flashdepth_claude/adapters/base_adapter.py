"""
Base adapter class for depth estimation methods
"""

import torch
from abc import ABC, abstractmethod


class MethodAdapter(ABC):
    """
    Base adapter interface for depth estimation methods
    """

    def __init__(self):
        self.model = None
        self.device = None
        self.processing_resolution = None  # Will be set on first inference: (H, W)

    @abstractmethod
    def load_model(self, checkpoint_path=None):
        """
        Load the depth estimation model

        Args:
            checkpoint_path: str - Path to model checkpoint

        Returns:
            model: torch.nn.Module - Loaded model
        """
        pass

    @abstractmethod
    def inference(self, image, intrinsics=None):
        """
        Run inference on a single image

        Args:
            image: torch.Tensor [1, 3, H, W] - Input image
            intrinsics: torch.Tensor or float - Optional camera intrinsics

        Returns:
            depth: torch.Tensor [1, H, W] - Predicted depth in meters
        """
        pass

    @abstractmethod
    def get_required_env(self):
        """
        Get required conda environment name

        Returns:
            str: Conda environment name
        """
        pass

    def preprocess(self, image):
        """
        Optional preprocessing (normalization, resizing, etc.)

        Args:
            image: torch.Tensor [1, 3, H, W]

        Returns:
            processed: torch.Tensor
        """
        return image

    def postprocess(self, depth, target_shape=None):
        """
        Optional postprocessing (resizing, clipping, etc.)

        Args:
            depth: torch.Tensor [1, H, W]
            target_shape: tuple - Optional target (H, W)

        Returns:
            processed: torch.Tensor
        """
        if target_shape is not None and depth.shape[-2:] != target_shape:
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1), size=target_shape,
                mode='bilinear', align_corners=True
            ).squeeze(1)
        return depth
