#!/usr/bin/env python3
"""
Script to generate placeholder data files for testing the XLSR-Transducer model.
Creates fake manifest files with sample entries for training, validation, and testing.
"""

import os
import argparse
import random
import string


def generate_random_text(length=50):
    """Generate random Estonian text for debugging."""
    # Basic Estonian characters
    chars = list("abcdefghijklmnopqrsšzžtuvwõäöüxy") + [" "] * 10
    return ''.join(random.choice(chars) for _ in range(length))


def generate_debug_manifest(manifest_path, num_samples=100):
    """Generate a debug manifest file with fake data."""
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    
    with open(manifest_path, 'w', encoding='utf-8') as f:
        for i in range(num_samples):
            # Generate fake paths and texts
            sample_id = f"{i:04d}"
            audio_path = f"samples/sample_{sample_id}.wav"
            text = generate_random_text(random.randint(10, 100))
            speaker_id = f"speaker_{i % 10:02d}"
            
            # Write to manifest
            f.write(f"{audio_path}|{text}|{speaker_id}\n")
    
    print(f"Debug manifest generated at {manifest_path} with {num_samples} samples")


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
    # Generate debug manifests for train, valid, and test sets
    data_dir = "data/manifests"
    os.makedirs(data_dir, exist_ok=True)
    
    generate_debug_manifest(os.path.join(data_dir, "train_debug.lst"), 100)
    generate_debug_manifest(os.path.join(data_dir, "valid_debug.lst"), 20)
    generate_debug_manifest(os.path.join(data_dir, "test_debug.lst"), 10)
    
    main() 