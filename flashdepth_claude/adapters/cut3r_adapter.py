"""
Adapter for CUT3R
Reference: refer_test/CUT3R/
"""

import sys
from pathlib import Path
from .base_adapter import MethodAdapter

class CUT3RAdapter(MethodAdapter):
    """Adapter for CUT3R"""
    
    def __init__(self):
        super().__init__()
        cut3r_path = Path(__file__).parent.parent / 'refer_test' / 'CUT3R'
        if str(cut3r_path) not in sys.path:
            sys.path.insert(0, str(cut3r_path))
    
    def load_model(self, checkpoint_path=None):
        """Load CUT3R model"""
        # TODO: Implement
        raise NotImplementedError("CUT3R adapter needs implementation. See refer_test/CUT3R/")
    
    def inference(self, image, intrinsics=None):
        """Run CUT3R inference"""
        # TODO: Implement
        raise NotImplementedError("CUT3R inference needs implementation")
    
    def get_required_env(self):
        return "cut3r"
