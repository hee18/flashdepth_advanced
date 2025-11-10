"""
Adapter for DepthCrafter
Reference: refer_test/DepthCrafter/
"""

import sys
from pathlib import Path
from .base_adapter import MethodAdapter

class DepthCrafterAdapter(MethodAdapter):
    """Adapter for DepthCrafter"""
    
    def __init__(self):
        super().__init__()
        dc_path = Path(__file__).parent.parent / 'refer_test' / 'DepthCrafter'
        if str(dc_path) not in sys.path:
            sys.path.insert(0, str(dc_path))
    
    def load_model(self, checkpoint_path=None):
        """Load DepthCrafter model"""
        # TODO: Implement
        raise NotImplementedError("DepthCrafter adapter needs implementation. See refer_test/DepthCrafter/")
    
    def inference(self, image, intrinsics=None):
        """Run DepthCrafter inference"""
        # TODO: Implement
        raise NotImplementedError("DepthCrafter inference needs implementation")
    
    def get_required_env(self):
        return "depthcrafter"
