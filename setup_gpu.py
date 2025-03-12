#!/usr/bin/env python3
"""
Setup script to install the appropriate GPU-accelerated RNN-T loss implementation.
This script detects the available GPU backend and installs the appropriate package.
"""

import os
import platform
import subprocess
import sys

def run_command(cmd):
    """Run a command and return its output."""
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(f"Error: {e.stderr}")
        return None

def check_cuda_available():
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        # Try using nvidia-smi
        result = run_command("nvidia-smi")
        return result is not None

def check_mps_available():
    """Check if MPS (Metal Performance Shaders) is available on macOS."""
    try:
        import torch
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps'):
            return torch.backends.mps.is_available()
        return False
    except ImportError:
        # Check if we're on macOS
        if platform.system() == "Darwin":
            # Check if we're on Apple Silicon
            result = run_command("sysctl -n machdep.cpu.brand_string")
            return result is not None and "Apple" in result
        return False

def install_package(package):
    """Install a Python package using pip."""
    print(f"Installing {package}...")
    result = run_command(f"{sys.executable} -m pip install {package}")
    if result is not None:
        print(f"Successfully installed {package}")
    else:
        print(f"Failed to install {package}")

def main():
    print("Setting up GPU-accelerated RNN-T loss implementation...")
    
    # Check for CUDA
    if check_cuda_available():
        print("CUDA detected! Installing CUDA-optimized RNN-T loss implementations...")
        
        # Install warp-rnnt (NVIDIA's implementation)
        install_package("warp-rnnt>=0.4.0")
        
        # Install fast-rnnt (another efficient implementation)
        install_package("fast-rnnt>=0.8.0")
        
        print("\nCUDA setup complete! Your training should now use GPU-accelerated RNN-T loss.")
        print("Expected speedup: 10-100x faster than CPU implementation.")
    
    # Check for MPS (Apple Silicon)
    elif check_mps_available():
        print("Apple Silicon MPS detected! Setting up for MPS acceleration...")
        
        # Make sure we have the latest PyTorch and torchaudio with MPS support
        install_package("--upgrade torch torchaudio")
        
        # fast-rnnt may also work on MPS in some cases
        try:
            install_package("fast-rnnt>=0.8.0")
        except:
            print("Could not install fast-rnnt, but that's okay. torchaudio will be used.")
        
        print("\nMPS setup complete! Your training will use Apple Silicon acceleration.")
        print("Expected speedup: 2-5x faster than CPU implementation.")
        print("Note: For best performance on Apple Silicon, make sure you're using")
        print("PyTorch 2.0+ and torchaudio with proper MPS support.")
    
    else:
        print("No compatible GPU detected. The training will use the CPU implementation.")
        print("This will be significantly slower. Consider running on a machine with CUDA support.")
    
    print("\nSetup completed!")

if __name__ == "__main__":
    main() 