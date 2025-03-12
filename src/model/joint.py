import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union


class TransducerJoint(nn.Module):
    """
    Joint network for the transducer model.
    Combines encoder and predictor outputs to predict the next token.
    """
    
    def __init__(
        self,
        encoder_dim: int,
        predictor_dim: int,
        hidden_dim: int,
        vocab_size: int,
        activation: str = "relu",
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.encoder_dim = encoder_dim
        self.predictor_dim = predictor_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        # Projection layers
        self.encoder_proj = nn.Linear(encoder_dim, hidden_dim)
        self.predictor_proj = nn.Linear(predictor_dim, hidden_dim)
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        
        # Initialize weights with smaller values to prevent exploding gradients
        nn.init.xavier_uniform_(self.encoder_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.predictor_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.1)
    
    def forward(
        self,
        encoder_outputs: torch.Tensor,
        predictor_outputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the joint network.
        
        Args:
            encoder_outputs: Encoder outputs of shape (batch_size, time_steps, encoder_dim)
            predictor_outputs: Predictor outputs of shape (batch_size, label_length, predictor_dim)
            
        Returns:
            Joint outputs of shape (batch_size, time_steps, label_length, vocab_size)
        """
        # Check for NaN values in inputs
        if torch.isnan(encoder_outputs).any() or torch.isnan(predictor_outputs).any():
            print("WARNING: NaN values detected in joint network inputs!")
            
        # Project encoder outputs
        encoder_proj = self.encoder_proj(encoder_outputs)  # (batch_size, time_steps, hidden_dim)
        
        # Project predictor outputs
        predictor_proj = self.predictor_proj(predictor_outputs)  # (batch_size, label_length, hidden_dim)
        
        # Add dimensions for broadcasting
        encoder_proj = encoder_proj.unsqueeze(2)  # (batch_size, time_steps, 1, hidden_dim)
        predictor_proj = predictor_proj.unsqueeze(1)  # (batch_size, 1, label_length, hidden_dim)
        
        # Combine outputs
        joint = encoder_proj + predictor_proj  # (batch_size, time_steps, label_length, hidden_dim)
        
        # Apply activation and dropout
        joint = self.activation(joint)
        joint = self.dropout(joint)
        
        # Project to vocabulary
        outputs = self.output_proj(joint)  # (batch_size, time_steps, label_length, vocab_size)
        
        # Scale outputs to prevent extreme values
        outputs = outputs * 0.1
        
        # Check for NaN values in outputs
        if torch.isnan(outputs).any():
            print("WARNING: NaN values detected in joint network outputs!")
            
        return outputs
    
    def forward_step(
        self,
        encoder_output: torch.Tensor,
        predictor_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward one step for streaming inference.
        
        Args:
            encoder_output: Encoder output of shape (batch_size, encoder_dim)
            predictor_output: Predictor output of shape (batch_size, predictor_dim)
            
        Returns:
            Joint output of shape (batch_size, vocab_size)
        """
        # Project encoder output
        encoder_proj = self.encoder_proj(encoder_output)  # (batch_size, hidden_dim)
        
        # Project predictor output
        predictor_proj = self.predictor_proj(predictor_output)  # (batch_size, hidden_dim)
        
        # Combine outputs
        joint = encoder_proj + predictor_proj  # (batch_size, hidden_dim)
        
        # Apply activation and dropout
        joint = self.activation(joint)
        joint = self.dropout(joint)
        
        # Project to vocabulary
        output = self.output_proj(joint)  # (batch_size, vocab_size)
        
        # Scale outputs to prevent extreme values
        output = output * 0.1
        
        return output 