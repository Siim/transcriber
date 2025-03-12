import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union


class TransducerPredictor(nn.Module):
    """
    Predictor network for the transducer model.
    Takes previous non-blank labels and predicts the next label distribution.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 640,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Output projection
        self.output_dim = hidden_dim
        
        # Initialize weights with smaller values to prevent exploding gradients
        nn.init.xavier_uniform_(self.embedding.weight, gain=0.1)
    
    def forward(
        self,
        labels: torch.Tensor,
        label_lengths: Optional[torch.Tensor] = None,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the predictor network with robust error handling.
        
        Args:
            labels: Input labels of shape (batch_size, max_label_length)
            label_lengths: Optional lengths of labels of shape (batch_size,)
            hidden: Optional initial hidden state for LSTM
            
        Returns:
            Dictionary containing:
                - outputs: Output tensor of shape (batch_size, max_label_length, hidden_dim)
                - hidden: Tuple of hidden states for LSTM
        """
        batch_size, max_label_length = labels.size()
        device = labels.device
        
        # Check for NaN values in input
        if torch.isnan(labels.float()).any():
            print(f"WARNING: NaN values detected in predictor input! Shape: {labels.shape}")
            labels = torch.nan_to_num(labels.float(), nan=0.0).long()
        
        # Validate label_lengths - with detailed diagnostics
        if label_lengths is not None:
            print(f"DEBUG: label_lengths min: {label_lengths.min().item()}, max: {label_lengths.max().item()}, shape: {label_lengths.shape}")
            print(f"DEBUG: labels shape: {labels.shape}")
            
            # Ensure label_lengths is at least 1 for each batch item
            if (label_lengths <= 0).any():
                print("WARNING: Some label lengths are zero or negative. Setting them to 1.")
                label_lengths = torch.clamp(label_lengths, min=1)
                
            # Ensure label_lengths doesn't exceed max_label_length
            if (label_lengths > max_label_length).any():
                print(f"WARNING: Some label lengths exceed max_label_length ({max_label_length}). Clamping.")
                label_lengths = torch.clamp(label_lengths, max=max_label_length)
        
        # Safety check for label values that might be out of range for the embedding layer
        if (labels >= self.vocab_size).any():
            print(f"WARNING: Some label values are >= vocab_size ({self.vocab_size}). Clipping to valid range.")
            labels = torch.clamp(labels, max=self.vocab_size-1)
        
        # Embed labels
        embedded = self.embedding(labels)  # (batch_size, max_label_length, embedding_dim)
        embedded = self.dropout(embedded)
        
        # Don't use pack_padded_sequence, directly process with LSTM
        try:
            outputs, hidden = self.lstm(embedded, hidden)
        except RuntimeError as e:
            print(f"WARNING: LSTM forward pass failed: {e}")
            # Attempt fallback to CPU
            try:
                print("Attempting CPU fallback for LSTM...")
                cpu_embedded = embedded.cpu()
                cpu_lstm = nn.LSTM(
                    input_size=self.embedding_dim,
                    hidden_size=self.hidden_dim,
                    num_layers=self.num_layers,
                    dropout=self.dropout.p if self.num_layers > 1 else 0.0,
                    batch_first=True,
                ).to('cpu')
                
                # Copy weights from original LSTM
                for name, param in self.lstm.named_parameters():
                    if hasattr(cpu_lstm, name):
                        getattr(cpu_lstm, name).data.copy_(param.data.cpu())
                
                if hidden is not None:
                    cpu_hidden = (hidden[0].cpu(), hidden[1].cpu())
                else:
                    cpu_hidden = None
                
                outputs, hidden = cpu_lstm(cpu_embedded, cpu_hidden)
                
                # Move back to original device
                outputs = outputs.to(device)
                if hidden is not None:
                    hidden = (hidden[0].to(device), hidden[1].to(device))
                
                print("CPU fallback succeeded")
            except Exception as inner_e:
                print(f"WARNING: CPU fallback also failed: {inner_e}")
                # Last resort - use a dummy output
                print("Using dummy output as last resort")
                outputs = torch.zeros(
                    batch_size, max_label_length, self.hidden_dim, 
                    device=device
                )
                # Keep hidden state as is, or initialize to zeros if None
                if hidden is None:
                    hidden = (
                        torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device),
                        torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
                    )
        
        # Apply layer normalization for stability
        outputs = self.layer_norm(outputs)
        
        # Apply dropout
        outputs = self.dropout(outputs)
        
        # Clip extreme values to prevent NaN propagation
        outputs = torch.clamp(outputs, min=-100.0, max=100.0)
        
        # Check for NaN values in output
        if torch.isnan(outputs).any():
            print("WARNING: NaN values detected in predictor output!")
            outputs = torch.nan_to_num(outputs, nan=0.0)
        
        return {
            "outputs": outputs,
            "hidden": hidden,
        }
    
    def forward_step(
        self,
        label: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward one step for streaming inference.
        
        Args:
            label: Input label of shape (batch_size, 1)
            hidden: Optional initial hidden state for LSTM
            
        Returns:
            output: Output tensor of shape (batch_size, hidden_dim)
            hidden: Tuple of hidden states for LSTM
        """
        # Safety check for label values that might be out of range for the embedding layer
        if (label >= self.vocab_size).any():
            print(f"WARNING: Some label values are >= vocab_size ({self.vocab_size}). Clipping to valid range.")
            label = torch.clamp(label, max=self.vocab_size-1)
            
        # Embed label
        embedded = self.embedding(label)  # (batch_size, 1, embedding_dim)
        embedded = self.dropout(embedded)
        
        # Forward through LSTM
        output, hidden = self.lstm(embedded, hidden)
        
        # Apply layer normalization
        output = self.layer_norm(output)
        
        # Apply dropout
        output = self.dropout(output)
        
        # Clip extreme values
        output = torch.clamp(output, min=-100.0, max=100.0)
        
        return output.squeeze(1), hidden 