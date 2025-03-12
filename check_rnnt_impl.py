#!/usr/bin/env python3
"""
Check which RNN-T loss implementation is being used.
This script helps diagnose which implementation will be used for training.
"""

import sys
import os
import logging
import torch

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def check_implementation():
    """Check which RNN-T loss implementation will be used."""
    print("Checking available RNN-T loss implementations...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Python version: {sys.version}")
    
    # Device information
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA available: Yes (version {torch.version.cuda})")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS (Apple Silicon) available: Yes")
    else:
        device = torch.device("cpu")
        print("No GPU acceleration available. Using CPU.")
    
    print(f"Using device: {device}")
    
    # Check for warp-rnnt (NVIDIA CUDA implementation)
    try:
        import warp_rnnt
        print("✅ warp-rnnt available (CUDA-optimized implementation)")
        has_warp_rnnt = True
    except ImportError:
        print("❌ warp-rnnt not available")
        has_warp_rnnt = False
    
    # Check for torchaudio.functional.rnnt_loss
    try:
        import torchaudio.functional as F_audio
        if hasattr(F_audio, 'rnnt_loss'):
            print("✅ torchaudio.functional.rnnt_loss available")
            has_torchaudio_rnnt = True
        else:
            print("❌ torchaudio.functional.rnnt_loss not available (requires newer torchaudio)")
            has_torchaudio_rnnt = False
    except ImportError:
        print("❌ torchaudio not available")
        has_torchaudio_rnnt = False
    
    # Check for fast-rnnt
    try:
        import fast_rnnt
        print("✅ fast-rnnt available (efficient CPU/CUDA implementation)")
        has_fast_rnnt = True
    except ImportError:
        print("❌ fast-rnnt not available")
        has_fast_rnnt = False
    
    # Determine which implementation will be used
    print("\nBased on your environment, the following implementation will be used:")
    
    if device.type == "cuda" and has_warp_rnnt:
        print("=> warp-rnnt (CUDA) - FASTEST for NVIDIA GPUs")
        print("   Expected speedup: 10-100x faster than CPU")
    elif (device.type == "cuda" or device.type == "mps") and has_torchaudio_rnnt:
        print(f"=> torchaudio.functional.rnnt_loss ({device.type.upper()})")
        if device.type == "cuda":
            print("   Expected speedup: 5-20x faster than CPU")
        else:  # mps
            print("   Expected speedup: 2-5x faster than CPU")
    elif (device.type == "cuda" or device.type == "cpu") and has_fast_rnnt:
        print(f"=> fast-rnnt ({device.type.upper()})")
        if device.type == "cuda":
            print("   Expected speedup: 5-20x faster than CPU")
        else:
            print("   Expected speedup: 2-3x faster than regular CPU implementation")
    else:
        print("=> Pure PyTorch CPU implementation (SLOWEST)")
        print("   Consider installing one of the accelerated implementations for better performance")
    
    # Recommendations
    print("\nRecommendations:")
    if device.type == "cuda" and not has_warp_rnnt:
        print("- Install warp-rnnt for maximum performance: pip install warp-rnnt")
    elif device.type == "mps" and not has_torchaudio_rnnt:
        print("- Update torchaudio to the latest version for MPS support: pip install --upgrade torchaudio")
    elif device.type == "cpu" and not has_fast_rnnt:
        print("- Install fast-rnnt for better CPU performance: pip install fast-rnnt")
    
    # Run setup_gpu.py script
    print("\nFor automatic setup, run: python setup_gpu.py")

if __name__ == "__main__":
    check_implementation() 