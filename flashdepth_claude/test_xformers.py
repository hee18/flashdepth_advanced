#!/usr/bin/env python3
"""
Test script to verify xFormers memory efficient attention is working
"""
import torch
import sys
import logging

def test_xformers():
    try:
        from xformers.ops import memory_efficient_attention, unbind
        print(f"✓ xFormers imported successfully")

        # Test with CUDA if available
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            dtype = torch.bfloat16
            print(f"✓ Testing on CUDA with bfloat16")
        else:
            device = torch.device('cpu')
            dtype = torch.float32
            print(f"✓ Testing on CPU with float32")

        # Test parameters similar to FlashDepth
        B, N, H, D = 2, 10, 8, 32
        query = torch.randn(B, N, H, D, dtype=dtype, device=device)
        key = torch.randn(B, N, H, D, dtype=dtype, device=device)
        value = torch.randn(B, N, H, D, dtype=dtype, device=device)

        print(f"Input shapes: Q={query.shape}, K={key.shape}, V={value.shape}")

        # Run memory efficient attention
        result = memory_efficient_attention(query, key, value)
        print(f"✓ memory_efficient_attention succeeded!")
        print(f"Output shape: {result.shape}")
        print(f"Output dtype: {result.dtype}")

        return True

    except ImportError as e:
        print(f"✗ xFormers import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ xFormers test failed: {e}")
        return False

def test_attention_import():
    """Test the actual attention module from FlashDepth"""
    try:
        # Add current directory to path
        sys.path.insert(0, '/app')
        from flashdepth.dinov2_layers.attention import XFORMERS_AVAILABLE, MemEffAttention

        print(f"XFORMERS_AVAILABLE: {XFORMERS_AVAILABLE}")

        # Test MemEffAttention initialization
        attn = MemEffAttention(dim=512, num_heads=8)
        print(f"✓ MemEffAttention initialized successfully")

        # Test forward pass
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            attn = attn.to(device)
            x = torch.randn(2, 100, 512, device=device, dtype=torch.bfloat16)
        else:
            device = torch.device('cpu')
            x = torch.randn(2, 100, 512, device=device)

        print(f"Testing MemEffAttention forward pass...")
        with torch.no_grad():
            output = attn(x)
        print(f"✓ MemEffAttention forward pass succeeded!")
        print(f"Input shape: {x.shape}, Output shape: {output.shape}")

        return True

    except Exception as e:
        print(f"✗ MemEffAttention test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Testing xFormers Memory Efficient Attention")
    print("=" * 50)

    success1 = test_xformers()
    print()
    success2 = test_attention_import()

    print("\n" + "=" * 50)
    if success1 and success2:
        print("✓ All tests PASSED! MemEffAttention should work correctly.")
    else:
        print("✗ Some tests FAILED! Check the errors above.")
        sys.exit(1)