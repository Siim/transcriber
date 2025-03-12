#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.processor import AudioPreprocessor, XLSRTransducerProcessor
from src.utils.audio import load_audio, plot_waveform


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test audio preprocessing")
    
    parser.add_argument(
        "--audio_file", type=str, required=True, help="Path to audio file"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Path to output directory"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load audio
    print(f"Loading audio file: {args.audio_file}")
    waveform, sample_rate = load_audio(args.audio_file, target_sample_rate=16000)
    
    # Plot original waveform
    plt.figure(figsize=(12, 4))
    plot_waveform(waveform, sample_rate, title="Original Waveform")
    plt.savefig(os.path.join(args.output_dir, "original_waveform.png"))
    
    # Print audio stats
    print(f"Audio shape: {waveform.shape}")
    print(f"Sample rate: {sample_rate}")
    print(f"Duration: {waveform.shape[1] / sample_rate:.2f}s")
    print(f"Min value: {waveform.min().item():.4f}")
    print(f"Max value: {waveform.max().item():.4f}")
    print(f"Mean value: {waveform.mean().item():.4f}")
    print(f"Std value: {waveform.std().item():.4f}")
    
    # Create audio preprocessor
    print("Creating audio preprocessor...")
    preprocessor = AudioPreprocessor(sample_rate=16000)
    
    # Process audio
    print("Processing audio...")
    inputs = preprocessor(args.audio_file)
    
    # Print feature stats
    input_values = inputs["input_values"]
    print(f"Feature shape: {input_values.shape}")
    print(f"Feature min value: {input_values.min().item():.4f}")
    print(f"Feature max value: {input_values.max().item():.4f}")
    print(f"Feature mean value: {input_values.mean().item():.4f}")
    print(f"Feature std value: {input_values.std().item():.4f}")
    
    # Check for NaN values
    if torch.isnan(input_values).any():
        print("WARNING: NaN values detected in features!")
    else:
        print("No NaN values detected in features.")
    
    # Plot feature values
    plt.figure(figsize=(12, 4))
    plt.plot(input_values[0, 0, :100].numpy())
    plt.title("First 100 Feature Values")
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, "feature_values.png"))
    
    # Create full processor
    print("Creating full processor...")
    processor = XLSRTransducerProcessor()
    
    # Process audio with text
    print("Processing audio with text...")
    sample_text = "this is a test"
    processed = processor(args.audio_file, sample_text)
    
    # Print processed stats
    print(f"Processed keys: {processed.keys()}")
    print(f"Input values shape: {processed['input_values'].shape}")
    if "labels" in processed:
        print(f"Labels shape: {processed['labels'].shape}")
        print(f"Labels: {processed['labels'].tolist()}")
    
    print("Audio preprocessing test completed successfully!")


if __name__ == "__main__":
    main() 