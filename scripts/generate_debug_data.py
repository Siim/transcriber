#!/usr/bin/env python3
"""
Script to generate placeholder data files for testing the XLSR-Transducer model.
Creates fake manifest files with sample entries for training, validation, and testing.
"""

import os
import argparse


def generate_debug_manifest(output_path, num_samples=100):
    """
    Generate a placeholder manifest file for testing.
    
    Args:
        output_path: Path to save the manifest file
        num_samples: Number of sample entries to generate
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create fake audio paths and transcriptions
    lines = []
    for i in range(num_samples):
        # Format: /path/to/audio.wav|transcription|speaker_id
        line = f"samples/sample_{i:04d}.wav|See test järjekordne näidis lause number {i}.|speaker_{i % 10:02d}\n"
        lines.append(line)
    
    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    
    print(f"Generated {num_samples} sample entries in {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate placeholder data files for testing")
    parser.add_argument("--train", type=int, default=100, help="Number of training samples")
    parser.add_argument("--val", type=int, default=50, help="Number of validation samples")
    parser.add_argument("--test", type=int, default=20, help="Number of test samples")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory")
    args = parser.parse_args()
    
    # Generate manifest files
    generate_debug_manifest(os.path.join(args.output_dir, "train_list.txt"), args.train)
    generate_debug_manifest(os.path.join(args.output_dir, "val_list.txt"), args.val)
    generate_debug_manifest(os.path.join(args.output_dir, "test_list.txt"), args.test)
    
    # Create samples directory if it doesn't exist
    os.makedirs(os.path.join(args.output_dir, "samples"), exist_ok=True)
    
    print("Debug data generation complete")


if __name__ == "__main__":
    main() 