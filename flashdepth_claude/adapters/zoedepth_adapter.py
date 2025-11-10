"""
Adapter for ZoeDepth
Reference: refer_test/ZoeDepth/
"""

import sys
from pathlib import Path
from .base_adapter import MethodAdapter

class ZoeDepthAdapter(MethodAdapter):
    """Adapter for ZoeDepth"""
    
    def __init__(self):
        super().__init__()
        zoe_path = Path(__file__).parent.parent / 'refer_test' / 'ZoeDepth'
        if str(zoe_path) not in sys.path:
            sys.path.insert(0, str(zoe_path))
    
    def load_model(self, checkpoint_path=None):
        """Load ZoeDepth model"""
        # TODO: Implement
        raise NotImplementedError("ZoeDepth adapter needs implementation. See refer_test/ZoeDepth/")
    
    def inference(self, image, intrinsics=None):
        """Run ZoeDepth inference"""
        # TODO: Implement
        raise NotImplementedError("ZoeDepth inference needs implementation")
    
    def get_required_env(self):
        return "zoedepth"
