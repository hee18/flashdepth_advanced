"""
Adapter for DepthPro (Apple ML)
Reference: refer_test/ml-depth-pro/
"""

import sys
from pathlib import Path
from .base_adapter import MethodAdapter

class DepthProAdapter(MethodAdapter):
    """Adapter for DepthPro"""
    
    def __init__(self):
        super().__init__()
        dp_path = Path(__file__).parent.parent / 'refer_test' / 'ml-depth-pro'
        if str(dp_path) not in sys.path:
            sys.path.insert(0, str(dp_path))
    
    def load_model(self, checkpoint_path=None):
        """Load DepthPro model"""
        # TODO: Implement
        raise NotImplementedError("DepthPro adapter needs implementation. See refer_test/ml-depth-pro/")
    
    def inference(self, image, intrinsics=None):
        """Run DepthPro inference"""
        # TODO: Implement
        raise NotImplementedError("DepthPro inference needs implementation")
    
    def get_required_env(self):
        return "depthpro"
