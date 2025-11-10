"""
Adapter for Video-Depth-Anything
Reference: refer_test/Video-Depth-Anything/
"""

import sys
from pathlib import Path
import torch
from .base_adapter import MethodAdapter

class VideoDepthAnythingAdapter(MethodAdapter):
    """Adapter for Video-Depth-Anything (VDA)"""
    
    def __init__(self):
        super().__init__()
        # Add VDA path to sys.path
        vda_path = Path(__file__).parent.parent / 'refer_test' / 'Video-Depth-Anything'
        if str(vda_path) not in sys.path:
            sys.path.insert(0, str(vda_path))
    
    def load_model(self, checkpoint_path=None):
        """Load Video-Depth-Anything model"""
        # TODO: Implement actual model loading
        # from video_depth_anything import DepthAnythingV2
        # self.model = DepthAnythingV2.from_pretrained(checkpoint_path)
        raise NotImplementedError("VDA adapter needs implementation. See refer_test/Video-Depth-Anything/")
    
    def inference(self, image, intrinsics=None):
        """Run VDA inference"""
        # TODO: Implement actual inference
        # with torch.no_grad():
        #     depth = self.model(image)
        # return depth
        raise NotImplementedError("VDA inference needs implementation")
    
    def get_required_env(self):
        return "vda"
