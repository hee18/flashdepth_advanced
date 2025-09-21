#!/usr/bin/env python3
"""
Simple test script to verify Docker environment setup
This script tests basic functionality without requiring GPU or Mamba2
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

def test_environment():
    """Test basic environment setup"""
    print("=" * 50)
    print("FlashDepth Docker Environment Test")
    print("=" * 50)

    # Test Python version
    print(f"Python version: {sys.version}")

    # Test PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    # Test directory structure
    print(f"Current working directory: {os.getcwd()}")

    # Check if datasets are mounted
    datasets_path = "/data/datasets"
    if os.path.exists(datasets_path):
        print(f"Datasets directory exists: {datasets_path}")
        if os.path.exists(os.path.join(datasets_path, "tartanair")):
            print("  ✓ TartanAir dataset found")
        else:
            print("  ✗ TartanAir dataset not found")

        # List available datasets
        try:
            datasets = [d for d in os.listdir(datasets_path) if os.path.isdir(os.path.join(datasets_path, d))]
            print(f"  Available datasets: {datasets}")
        except Exception as e:
            print(f"  Error listing datasets: {e}")
    else:
        print(f"Datasets directory not found: {datasets_path}")

    # Check results directories
    results_dirs = ["/app/train_results", "/app/checkpoints"]
    for dir_path in results_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path} exists")
        else:
            print(f"✗ {dir_path} not found")
            try:
                os.makedirs(dir_path, exist_ok=True)
                print(f"  Created {dir_path}")
            except Exception as e:
                print(f"  Error creating {dir_path}: {e}")

    # Test basic imports
    print("\nTesting basic imports...")
    try:
        import hydra
        print("✓ Hydra imported successfully")
    except ImportError as e:
        print(f"✗ Error importing Hydra: {e}")

    try:
        import cv2
        print("✓ OpenCV imported successfully")
    except ImportError as e:
        print(f"✗ Error importing OpenCV: {e}")

    try:
        import matplotlib
        print("✓ Matplotlib imported successfully")
    except ImportError as e:
        print(f"✗ Error importing Matplotlib: {e}")

    try:
        from einops import rearrange
        print("✓ Einops imported successfully")
    except ImportError as e:
        print(f"✗ Error importing Einops: {e}")

    # Test torch functionality
    print("\nTesting PyTorch functionality...")
    try:
        x = torch.randn(2, 3, 4)
        y = torch.nn.functional.relu(x)
        print(f"✓ Basic tensor operations work")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {y.shape}")
    except Exception as e:
        print(f"✗ Error with tensor operations: {e}")

    # Test project imports (without Mamba)
    print("\nTesting project imports...")
    sys.path.append('/app')

    try:
        import utils.helpers
        print("✓ Utils helpers imported successfully")
    except Exception as e:
        print(f"✗ Error importing utils: {e}")

    try:
        from dataloaders.combined_dataset import CombinedDataset
        print("✓ CombinedDataset imported successfully")
    except Exception as e:
        print(f"✗ Error importing CombinedDataset: {e}")

    print("\n" + "=" * 50)
    print("Environment test completed!")
    print("=" * 50)

if __name__ == "__main__":
    test_environment()