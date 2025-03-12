import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union


class AttentionSink(nn.Module):
    """
    Attention Sink mechanism for efficient streaming inference with Transformer models.
    
    This is based on the paper "Attention Sinks: Skip Connections in Transformers for Long Context".
    It allows efficient linear-complexity processing of long sequences by using two key mechanisms:
    1. A fixed number of "sink" tokens at the beginning of the sequence that attend to all positions
    2. A sliding window attention mechanism where each token can only attend to recent tokens
    
    This results in much more efficient processing while maintaining accuracy.
    """
    
    def __init__(
        self,
        sink_size: int = 4,
        left_context: int = 25,
        right_context: int = 0,
    ):
        super().__init__()
        self.sink_size = sink_size
        self.left_context = left_context
        self.right_context = right_context
    
    def create_attention_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create an attention mask with attention sinks for streaming inference.
        
        Args:
            seq_len: Length of the sequence
            device: Device to create the mask on
            
        Returns:
            Attention mask tensor of shape (seq_len, seq_len)
        """
        # Initialize with all -inf (no attention)
        mask = torch.full(
            (seq_len, seq_len), float("-inf"), device=device
        )
        
        # For each position, allow attention to:
        # 1. The first attention_sink_size frames (attention sink)
        # 2. Positions within the left context
        # 3. Positions within the right context
        for i in range(seq_len):
            # Attention sink (first few frames)
            mask[i, :self.sink_size] = 0.0
            
            # Left context (including current position)
            left_start = max(0, i - self.left_context)
            mask[i, left_start:i+1] = 0.0
            
            # Right context
            right_end = min(seq_len, i + self.right_context + 1)
            if right_end > i + 1:
                mask[i, i+1:right_end] = 0.0
        
        return mask
    
    def apply_mask(
        self,
        attention_weights: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """
        Apply the attention sink mask to attention weights.
        
        Args:
            attention_weights: Attention weights of shape (batch_size, num_heads, seq_len, seq_len)
            seq_len: Length of the sequence
            
        Returns:
            Masked attention weights
        """
        # Create the mask on the same device as the attention weights
        mask = self.create_attention_mask(seq_len, attention_weights.device)
        
        # Apply the mask
        masked_attention = attention_weights + mask.unsqueeze(0).unsqueeze(0)
        
        return masked_attention


def test_attention_sink():
    """Test function to visualize the attention sink mask."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create an attention sink with 4 sink positions and 10 context positions
    sink = AttentionSink(sink_size=4, left_context=10, right_context=2)
    
    # Create a mask for a sequence of length 50
    mask = sink.create_attention_mask(50, torch.device("cpu"))
    
    # Convert to numpy for visualization
    mask_np = mask.cpu().numpy()
    
    # Convert -inf to a small value for visualization
    mask_np[mask_np == float("-inf")] = -10
    
    # Plot the mask
    plt.figure(figsize=(10, 8))
    plt.imshow(mask_np, cmap="viridis")
    plt.colorbar()
    plt.title("Attention Sink Mask")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.savefig("attention_sink_mask.png")
    
    # Print some statistics
    print(f"Attention mask shape: {mask.shape}")
    print(f"Number of valid attention connections: {(mask != float('-inf')).sum().item()}")
    print(f"Sparsity: {(mask == float('-inf')).sum().item() / (mask.shape[0] * mask.shape[1]):.2%}")


if __name__ == "__main__":
    test_attention_sink() 