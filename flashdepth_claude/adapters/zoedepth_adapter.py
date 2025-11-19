"""
Adapter for ZoeDepth

Reference: refer_test/ZoeDepth/
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from .base_adapter import MethodAdapter


class ZoeDepthAdapter(MethodAdapter):
    """Adapter for ZoeDepth (ZoeD_NK variant)"""

    def __init__(self, variant='NK'):
        super().__init__()
        self.variant = variant  # N (NYU), K (KITTI), NK (NYU+KITTI)

        # Add ZoeDepth path to sys.path
        zoe_path = Path(__file__).parent.parent / 'refer_test' / 'ZoeDepth'
        if str(zoe_path) not in sys.path:
            sys.path.insert(0, str(zoe_path))

    def load_model(self, checkpoint_path=None):
        """
        Load ZoeDepth model using torch.hub.load

        ZoeDepth variants:
        - ZoeD_N: Trained on NYU Depth V2 (indoor)
        - ZoeD_K: Trained on KITTI (outdoor driving)
        - ZoeD_NK: Trained on NYU + KITTI (mixed)
        """
        # Use torch.hub.load to download and load pretrained model
        repo = "isl-org/ZoeDepth"
        model_name = f"ZoeD_{self.variant}"

        print(f"Loading ZoeDepth model: {model_name} from {repo}")

        # Load with PyTorch 1.13.1 (official ZoeDepth requirement)
        # This should work without compatibility issues
        self.model = torch.hub.load(repo, model_name, pretrained=True, trust_repo=True)

        print(f"ZoeDepth model loaded successfully")
        return self.model

    def inference(self, image, intrinsics=None):
        """
        Run ZoeDepth inference

        Args:
            image: torch.Tensor [1, 3, H, W] - Input image (0-1 normalized, RGB)
            intrinsics: Optional camera intrinsics (not used)

        Returns:
            depth: torch.Tensor [1, H, W] - Metric depth in meters
        """
        # ZoeDepth expects input in [0, 1] range, RGB format
        # Input shape: [1, 3, H, W]
        orig_H, orig_W = image.shape[2:]

        if self.device is not None:
            image = image.to(self.device)

        # Run inference
        with torch.no_grad():
            # ZoeDepth's infer method returns [1, 1, H, W] or [1, H, W]
            depth = self.model.infer(image)

        # Ensure output is [1, H, W]
        if depth.dim() == 4:
            depth = depth.squeeze(1)  # [1, 1, H, W] -> [1, H, W]

        # Record processing resolution on first inference (ZoeDepth internal)
        if self.processing_resolution is None:
            # Try to extract internal processing resolution from model
            try:
                # Debug: print model structure
                print(f"[DEBUG] Model type: {type(self.model)}")
                print(f"[DEBUG] Model attributes: {[attr for attr in dir(self.model) if not attr.startswith('_')][:20]}")

                # Try different paths to find img_size
                found = False

                # Path 1: model.core.prep.img_size
                if hasattr(self.model, 'core') and hasattr(self.model.core, 'prep'):
                    prep = self.model.core.prep
                    print(f"[DEBUG] prep type: {type(prep)}")
                    print(f"[DEBUG] prep attributes: {[attr for attr in dir(prep) if not attr.startswith('_')][:20]}")
                    if hasattr(prep, 'img_size'):
                        h, w = prep.img_size
                        self.processing_resolution = (h, w)
                        print(f"[ZoeDepth] Processing resolution: {h}×{w} (model-internal, extracted)")
                        found = True

                # Path 2: model.img_size
                if not found and hasattr(self.model, 'img_size'):
                    h, w = self.model.img_size
                    self.processing_resolution = (h, w)
                    print(f"[ZoeDepth] Processing resolution: {h}×{w} (model-internal, extracted)")
                    found = True

                if not found:
                    self.processing_resolution = "model-internal"
                    print(f"[ZoeDepth] Processing resolution: model-internal (unknown)")
            except Exception as e:
                self.processing_resolution = "model-internal"
                print(f"[ZoeDepth] Processing resolution: model-internal (extraction failed: {e})")

        # Verify and resize if needed
        # Note: ZoeDepth may internally resize, so we ensure output matches input
        if depth.shape[1] != orig_H or depth.shape[2] != orig_W:
            import torch.nn.functional as F
            depth = F.interpolate(
                depth.unsqueeze(1),  # [1, H, W] -> [1, 1, H, W]
                size=(orig_H, orig_W),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)  # [1, 1, H, W] -> [1, H, W]

        return depth

    def get_required_env(self):
        return "zoedepth"
