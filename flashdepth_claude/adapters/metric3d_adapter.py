"""
Adapter for Metric3D v1/v2
Reference: refer_test/Metric3D/
"""

import sys
from pathlib import Path
from .base_adapter import MethodAdapter

class Metric3DAdapter(MethodAdapter):
    """Adapter for Metric3D v1/v2"""
    
    def __init__(self, version='v2'):
        super().__init__()
        self.version = version
        m3d_path = Path(__file__).parent.parent / 'refer_test' / 'Metric3D'
        if str(m3d_path) not in sys.path:
            sys.path.insert(0, str(m3d_path))
    
    def load_model(self, checkpoint_path=None):
        """Load Metric3D model"""
        # TODO: Implement
        # if self.version == 'v1':
        #     from metric3d.v1 import Metric3D
        # else:
        #     from metric3d.v2 import Metric3Dv2
        raise NotImplementedError(f"Metric3D {self.version} adapter needs implementation. See refer_test/Metric3D/")
    
    def inference(self, image, intrinsics=None):
        """Run Metric3D inference"""
        # TODO: Implement
        raise NotImplementedError("Metric3D inference needs implementation")
    
    def get_required_env(self):
        return "metric3d"
