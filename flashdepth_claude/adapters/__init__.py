"""
Method adapters for comparison depth estimation methods

Each adapter implements a unified interface for:
- Model loading
- Inference
- Environment requirements
"""

from .base_adapter import MethodAdapter

__all__ = ['MethodAdapter']
