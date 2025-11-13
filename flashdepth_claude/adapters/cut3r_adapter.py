"""
Adapter for CUT3R (3D Reconstruction)
Reference: refer_test/CUT3R/

WARNING: CUT3R is designed for multi-view 3D reconstruction, not single-image
depth estimation. Single-frame depth inference may not produce reliable results.
Consider using other methods for monocular depth estimation.
"""

import sys
from pathlib import Path
from .base_adapter import MethodAdapter


class CUT3RAdapter(MethodAdapter):
    """
    Adapter for CUT3R

    Note: CUT3R requires multiple views for accurate 3D reconstruction.
    Single-view depth estimation is not the primary use case.
    """

    def __init__(self, size=512):
        super().__init__()
        self.size = size  # Input image size (default: 512)

        cut3r_path = Path(__file__).parent.parent / 'refer_test' / 'CUT3R'
        if str(cut3r_path) not in sys.path:
            sys.path.insert(0, str(cut3r_path))

    def load_model(self, checkpoint_path=None):
        """
        Load CUT3R model

        CUT3R is designed for multi-view 3D reconstruction.
        For comparison purposes, we implement single-view inference,
        but results may not be optimal.
        """
        # TODO: Implement CUT3R loading
        # This requires:
        # 1. Loading ARCroco3DStereo model
        # 2. Setting up image preprocessing
        # 3. Configuring for single-view mode (if possible)
        raise NotImplementedError(
            "CUT3R adapter not fully implemented. "
            "CUT3R is designed for multi-view 3D reconstruction, "
            "not single-image depth estimation. "
            "See refer_test/CUT3R/demo.py for reference implementation."
        )

    def inference(self, image, intrinsics=None):
        """
        Run CUT3R inference (single view)

        Args:
            image: torch.Tensor [1, 3, H, W] - Input image
            intrinsics: Optional camera intrinsics

        Returns:
            depth: torch.Tensor [1, H, W] - Depth map

        Note: CUT3R requires multiple views for best results.
        """
        # TODO: Implement single-view depth extraction
        # This is not CUT3R's primary use case
        raise NotImplementedError(
            "CUT3R single-view depth estimation not implemented. "
            "CUT3R requires multiple views for accurate depth. "
            "Consider using DepthPro, UniDepth, or Metric3D for single-view depth."
        )

    def get_required_env(self):
        return "cut3r"
