"""
Adapter for UniDepth v1/v2
Reference: refer_test/unidepth/
"""

import sys
from pathlib import Path
from .base_adapter import MethodAdapter

class UniDepthAdapter(MethodAdapter):
    """Adapter for UniDepth v1/v2"""
    
    def __init__(self, version='v2'):
        super().__init__()
        self.version = version
        ud_path = Path(__file__).parent.parent / 'refer_test' / 'unidepth'
        if str(ud_path) not in sys.path:
            sys.path.insert(0, str(ud_path))
    
    def load_model(self, checkpoint_path=None):
        """Load UniDepth model"""
        # TODO: Implement
        raise NotImplementedError(f"UniDepth {self.version} adapter needs implementation. See refer_test/unidepth/")
    
    def inference(self, image, intrinsics=None):
        """Run UniDepth inference"""
        # TODO: Implement
        raise NotImplementedError("UniDepth inference needs implementation")
    
    def get_required_env(self):
        return "unidepth"
