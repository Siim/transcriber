#!/usr/bin/env python3

"""
Wrapper script to run multi-stage training for XLSR-Transducer.
This handles Python path issues better than running scripts directly.
"""

import os
import sys

# Ensure the src directory is in the Python path
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

def main():
    # Add scripts directory to path
    scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    
    # Import and run the training script (sys.argv already contains the arguments)
    from train_by_stages import main as train_main
    train_main()

if __name__ == "__main__":
    main() 