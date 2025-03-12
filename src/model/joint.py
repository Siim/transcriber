import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import logging

logger = logging.getLogger(__name__)

class TransducerJoint(nn.Module):
    """
    Joint network for RNN-T.
    """

    def __init__(
        self,
        encoder_dim: int,
        predictor_dim: int,
        hidden_dim: int,
        vocab_size: int,
        dropout: float = 0.0,
        joint_dropout: float = 0.0,
        use_bias: bool = True,
        activation: str = "relu",
        norm_type: Optional[str] = None,
    ):
        """
        Initialize the joint network.
        
        Args:
            encoder_dim: Dimension of encoder output
            predictor_dim: Dimension of predictor output
            hidden_dim: Dimension of hidden layer
            vocab_size: Size of vocabulary
            dropout: Dropout rate for projection layers
            joint_dropout: Dropout rate for joint output
            use_bias: Whether to use bias in projection layers
            activation: Activation function to use
            norm_type: Type of normalization to use. Either "batch" or "layer" or None.
        """
        super().__init__()
        
        # Dimensions
        self.encoder_dim = encoder_dim
        self.predictor_dim = predictor_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        # Dropout rates
        self.dropout = dropout
        self.joint_dropout = joint_dropout
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        # Projection layers
        self.encoder_proj = nn.Linear(encoder_dim, hidden_dim, bias=use_bias)
        self.predictor_proj = nn.Linear(predictor_dim, hidden_dim, bias=use_bias)
        self.dropout_layer = nn.Dropout(dropout)
        self.joint_dropout_layer = nn.Dropout(joint_dropout)
        
        # Normalization
        self.norm_type = norm_type
        if norm_type == "batch":
            self.norm = nn.BatchNorm1d(hidden_dim)
            self.norm_2d = True
        elif norm_type == "layer":
            self.norm = nn.LayerNorm(hidden_dim)
            self.norm_2d = False
        else:
            self.norm = None
            self.norm_2d = False
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, vocab_size, bias=use_bias)
        
        # Weight initialization - Xavier uniform to prevent exploding gradients
        nn.init.xavier_uniform_(self.encoder_proj.weight)
        nn.init.xavier_uniform_(self.predictor_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
        
        # Bias initialization
        if use_bias:
            nn.init.zeros_(self.encoder_proj.bias)
            nn.init.zeros_(self.predictor_proj.bias)
            nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        predictor_outputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for joint network.
        
        Args:
            encoder_outputs: Output from encoder (B, T, D_enc)
            predictor_outputs: Output from predictor (B, U, D_pred)
            
        Returns:
            Joint output (B, T, U, V)
        """
        # Check for NaN values in inputs
        if torch.isnan(encoder_outputs).any():
            logger.warning("NaN values detected in encoder outputs")
        if torch.isnan(predictor_outputs).any():
            logger.warning("NaN values detected in predictor outputs")
        
        # Check if inputs require gradients - log warning but don't modify tensors
        if not encoder_outputs.requires_grad:
            logger.warning("Encoder outputs don't require gradients - this may cause issues in training")
        if not predictor_outputs.requires_grad:
            logger.warning("Predictor outputs don't require gradients - this may cause issues in training")
        
        # Get dimensions
        batch_size, max_time, _ = encoder_outputs.size()
        _, max_label, _ = predictor_outputs.size()
        
        # Project encoder outputs (B, T, D_enc) -> (B, T, D_h)
        encoder_proj = self.encoder_proj(encoder_outputs)
        encoder_proj = self.dropout_layer(encoder_proj)
        
        # Project predictor outputs (B, U, D_pred) -> (B, U, D_h)
        predictor_proj = self.predictor_proj(predictor_outputs)
        predictor_proj = self.dropout_layer(predictor_proj)
        
        # Expand tensors for broadcasting
        # (B, T, D_h) -> (B, T, 1, D_h)
        encoder_proj = encoder_proj.unsqueeze(2)
        # (B, U, D_h) -> (B, 1, U, D_h)
        predictor_proj = predictor_proj.unsqueeze(1)
        
        # Broadcast and sum
        # (B, T, 1, D_h) + (B, 1, U, D_h) -> (B, T, U, D_h)
        joint = encoder_proj + predictor_proj
        
        # Apply activation
        joint = self.activation(joint)
        
        # Apply normalization if needed
        if self.norm is not None:
            if self.norm_2d:  # BatchNorm1d expects [N, C, *]
                joint_for_norm = joint.view(-1, self.hidden_dim).contiguous()
                joint = self.norm(joint_for_norm).view(batch_size, max_time, max_label, self.hidden_dim)
            else:  # LayerNorm
                joint = self.norm(joint)
        
        # Apply dropout and project to vocabulary dimension
        joint = self.joint_dropout_layer(joint)
        
        # (B, T, U, D_h) -> (B, T, U, V)
        logits = self.output_proj(joint)
        
        return logits

    def forward_step(
        self,
        encoder_output: torch.Tensor,
        predictor_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single step forward pass for streaming inference.
        
        Args:
            encoder_output: Output from encoder (B, D_enc)
            predictor_output: Output from predictor (B, D_pred)
            
        Returns:
            Joint output (B, V)
        """
        # Check for NaN values in inputs
        if torch.isnan(encoder_output).any():
            logger.warning("NaN values detected in encoder output")
        if torch.isnan(predictor_output).any():
            logger.warning("NaN values detected in predictor output")
        
        # Check if inputs require gradients - log warning but don't modify tensors
        if not encoder_output.requires_grad:
            logger.warning("Encoder output doesn't require gradients - this may cause issues in training")
        if not predictor_output.requires_grad:
            logger.warning("Predictor output doesn't require gradients - this may cause issues in training")
        
        # Project encoder output (B, D_enc) -> (B, D_h)
        encoder_proj = self.encoder_proj(encoder_output)
        encoder_proj = self.dropout_layer(encoder_proj)
        
        # Project predictor output (B, D_pred) -> (B, D_h)
        predictor_proj = self.predictor_proj(predictor_output)
        predictor_proj = self.dropout_layer(predictor_proj)
        
        # Sum projections
        # (B, D_h) + (B, D_h) -> (B, D_h)
        joint = encoder_proj + predictor_proj
        
        # Apply activation
        joint = self.activation(joint)
        
        # Apply normalization if needed
        if self.norm is not None:
            if self.norm_2d:  # BatchNorm1d
                joint = self.norm(joint.unsqueeze(1)).squeeze(1)
            else:  # LayerNorm
                joint = self.norm(joint)
        
        # Apply dropout and project to vocabulary dimension
        joint = self.joint_dropout_layer(joint)
        
        # (B, D_h) -> (B, V)
        logits = self.output_proj(joint)
        
        return logits 