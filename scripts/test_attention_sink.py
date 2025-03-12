#!/usr/bin/env python3

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the src directory to the path
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.append(src_dir)

from model.attention import AttentionSink


def parse_args():
    parser = argparse.ArgumentParser(description="Test and visualize attention sink mechanism")
    parser.add_argument(
        "--sink_size", type=int, default=4, help="Number of sink tokens at the beginning"
    )
    parser.add_argument(
        "--left_context", type=int, default=25, help="Number of left context tokens"
    )
    parser.add_argument(
        "--right_context", type=int, default=0, help="Number of right context tokens"
    )
    parser.add_argument(
        "--seq_len", type=int, default=100, help="Sequence length for visualization"
    )
    parser.add_argument(
        "--output_dir", type=str, default="attention_sink_plots", help="Output directory for plots"
    )
    return parser.parse_args()


def plot_attention_mask(mask_np, title, output_path):
    """Plot an attention mask."""
    # Convert -inf to a small value for visualization
    mask_viz = mask_np.copy()
    mask_viz[mask_viz == float("-inf")] = -10
    
    plt.figure(figsize=(10, 8))
    plt.imshow(mask_viz, cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.savefig(output_path)
    plt.close()
    
    print(f"Plot saved to {output_path}")


def simulate_streaming(sink, seq_len, output_dir):
    """Simulate streaming inference with attention sink."""
    # Create a directory for streaming simulation plots
    stream_dir = os.path.join(output_dir, "streaming")
    os.makedirs(stream_dir, exist_ok=True)
    
    # Simulate frames arriving one by one
    for frame in range(0, seq_len, 10):  # Step by 10 frames
        current_len = min(frame + 50, seq_len)  # Current sequence length
        
        # Create mask for current sequence
        mask = sink.create_attention_mask(current_len, torch.device("cpu"))
        mask_np = mask.cpu().numpy()
        
        # Plot the current state
        title = f"Streaming Frame {frame}/{seq_len}"
        output_path = os.path.join(stream_dir, f"frame_{frame:03d}.png")
        plot_attention_mask(mask_np, title, output_path)
    
    print(f"Streaming simulation plots saved to {stream_dir}")


def compare_with_standard(sink, seq_len, output_dir):
    """Compare attention sink with standard attention masks."""
    # Create directory for comparison plots
    compare_dir = os.path.join(output_dir, "comparison")
    os.makedirs(compare_dir, exist_ok=True)
    
    # Create attention sink mask
    sink_mask = sink.create_attention_mask(seq_len, torch.device("cpu"))
    sink_mask_np = sink_mask.cpu().numpy()
    
    # Create standard full attention mask (causal)
    full_mask = torch.full((seq_len, seq_len), float("-inf"), device="cpu")
    for i in range(seq_len):
        full_mask[i, :i+1] = 0.0
    full_mask_np = full_mask.cpu().numpy()
    
    # Create chunk attention mask (no sink)
    chunk_size = 20
    chunk_mask = torch.full((seq_len, seq_len), float("-inf"), device="cpu")
    for i in range(seq_len):
        # Current chunk
        chunk_idx = i // chunk_size
        chunk_start = chunk_idx * chunk_size
        chunk_end = min((chunk_idx + 1) * chunk_size, seq_len)
        
        # Left context
        left_start = max(0, i - sink.left_context)
        
        # Allow attention within the chunk and to left context
        chunk_mask[i, chunk_start:chunk_end] = 0.0
        chunk_mask[i, left_start:chunk_start] = 0.0
    chunk_mask_np = chunk_mask.cpu().numpy()
    
    # Plot masks
    plot_attention_mask(
        full_mask_np, "Full Causal Attention", os.path.join(compare_dir, "full_attention.png")
    )
    plot_attention_mask(
        chunk_mask_np, "Chunk Attention", os.path.join(compare_dir, "chunk_attention.png")
    )
    plot_attention_mask(
        sink_mask_np, "Attention Sink", os.path.join(compare_dir, "attention_sink.png")
    )
    
    # Compute statistics
    full_connections = (full_mask != float("-inf")).sum().item()
    chunk_connections = (chunk_mask != float("-inf")).sum().item()
    sink_connections = (sink_mask != float("-inf")).sum().item()
    
    total_connections = seq_len * seq_len
    
    print("\nAttention Pattern Comparison:")
    print(f"Full Attention: {full_connections} connections ({full_connections/total_connections:.2%})")
    print(f"Chunk Attention: {chunk_connections} connections ({chunk_connections/total_connections:.2%})")
    print(f"Attention Sink: {sink_connections} connections ({sink_connections/total_connections:.2%})")
    
    with open(os.path.join(compare_dir, "stats.txt"), "w") as f:
        f.write("Attention Pattern Comparison:\n")
        f.write(f"Sequence Length: {seq_len}\n")
        f.write(f"Full Attention: {full_connections} connections ({full_connections/total_connections:.2%})\n")
        f.write(f"Chunk Attention: {chunk_connections} connections ({chunk_connections/total_connections:.2%})\n")
        f.write(f"Attention Sink: {sink_connections} connections ({sink_connections/total_connections:.2%})\n")
        f.write(f"Left Context: {sink.left_context}, Right Context: {sink.right_context}, Sink Size: {sink.sink_size}\n")


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create an attention sink
    sink = AttentionSink(
        sink_size=args.sink_size,
        left_context=args.left_context,
        right_context=args.right_context,
    )
    
    print(f"Testing Attention Sink with:")
    print(f"  - Sink size: {args.sink_size}")
    print(f"  - Left context: {args.left_context}")
    print(f"  - Right context: {args.right_context}")
    print(f"  - Sequence length: {args.seq_len}")
    
    # Create and plot the mask
    mask = sink.create_attention_mask(args.seq_len, torch.device("cpu"))
    mask_np = mask.cpu().numpy()
    
    # Plot the full mask
    plot_attention_mask(
        mask_np, "Attention Sink Mask", os.path.join(args.output_dir, "attention_sink.png")
    )
    
    # Print statistics
    print(f"\nAttention mask shape: {mask.shape}")
    print(f"Number of valid attention connections: {(mask != float('-inf')).sum().item()}")
    print(f"Sparsity: {(mask == float('-inf')).sum().item() / (mask.shape[0] * mask.shape[1]):.2%}")
    
    # Compare with other attention patterns
    compare_with_standard(sink, args.seq_len, args.output_dir)
    
    # Simulate streaming inference
    simulate_streaming(sink, args.seq_len, args.output_dir)


if __name__ == "__main__":
    main() 