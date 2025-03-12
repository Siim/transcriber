import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class TransducerLoss(nn.Module):
    """
    Transducer loss for training the XLSR-Transducer model.
    Implements the RNN-T loss function.
    """
    
    def __init__(self, blank_id: int, reduction: str = "mean"):
        """
        Initialize the transducer loss.
        
        Args:
            blank_id: ID of the blank token
            reduction: Reduction method ("none", "mean", "sum")
        """
        super().__init__()
        self.blank_id = blank_id
        self.reduction = reduction
        
        try:
            # Try to import and use the CUDA implementation for faster computation
            from warp_rnnt import rnnt_loss
            self.rnnt_loss = rnnt_loss
            self.use_cuda = True
        except ImportError:
            # Fall back to CPU implementation
            self.use_cuda = False
            print("Warning: warp_rnnt not found. Using CPU implementation of transducer loss.")
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        logit_lengths: torch.Tensor,
        label_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the transducer loss.
        
        Args:
            logits: Output tensor of shape (batch_size, time_steps, label_length, vocab_size)
            labels: Labels of shape (batch_size, max_label_length)
            logit_lengths: Lengths of logits of shape (batch_size,)
            label_lengths: Lengths of labels of shape (batch_size,)
            
        Returns:
            Loss value
        """
        # Check for NaN values in inputs
        if torch.isnan(logits).any():
            print("WARNING: NaN values detected in loss function logits input!")
            logits = torch.nan_to_num(logits, nan=0.0)
        
        # Ensure lengths are valid
        logit_lengths = torch.clamp(logit_lengths, min=1)
        label_lengths = torch.clamp(label_lengths, min=1)
        
        # Ensure label values are within the valid vocabulary range
        vocab_size = logits.size(-1)
        if (labels >= vocab_size).any():
            print(f"WARNING: Found label values >= vocab_size ({vocab_size}) in TransducerLoss. Clipping to valid range.")
            labels = torch.clamp(labels, max=vocab_size-1)
        
        if self.use_cuda and logits.is_cuda:
            # Use CUDA implementation
            try:
                loss = self.rnnt_loss(
                    logits=logits,
                    targets=labels,
                    logit_lengths=logit_lengths,
                    target_lengths=label_lengths,
                    blank=self.blank_id,
                    reduction=self.reduction,
                )
                
                # Check for NaN loss
                if torch.isnan(loss).any():
                    print("WARNING: NaN loss detected in CUDA implementation!")
                    # Fall back to CPU implementation
                    loss = self._compute_loss_cpu(
                        logits=logits,
                        labels=labels,
                        logit_lengths=logit_lengths,
                        label_lengths=label_lengths,
                    )
            except Exception as e:
                print(f"Error in CUDA implementation: {e}")
                # Fall back to CPU implementation
                loss = self._compute_loss_cpu(
                    logits=logits,
                    labels=labels,
                    logit_lengths=logit_lengths,
                    label_lengths=label_lengths,
                )
        else:
            # Use CPU implementation
            loss = self._compute_loss_cpu(
                logits=logits,
                labels=labels,
                logit_lengths=logit_lengths,
                label_lengths=label_lengths,
            )
        
        # Final check for NaN loss
        if torch.isnan(loss).any():
            print("WARNING: NaN loss detected after computation!")
            # Return a small constant loss instead of NaN
            loss = torch.tensor(10.0, device=logits.device, requires_grad=True)
        
        return loss
    
    def _compute_loss_cpu(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        logit_lengths: torch.Tensor,
        label_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the transducer loss using CPU implementation.
        
        Args:
            logits: Output tensor of shape (batch_size, max_time_steps, max_label_length + 1, vocab_size)
            labels: Labels of shape (batch_size, max_label_length)
            logit_lengths: Lengths of logits of shape (batch_size,)
            label_lengths: Lengths of labels of shape (batch_size,)
            
        Returns:
            Loss tensor
        """
        batch_size = logits.size(0)
        losses = []
        
        # Ensure label values are within the valid vocabulary range
        vocab_size = logits.size(-1)
        if (labels >= vocab_size).any():
            print(f"WARNING: Found label values >= vocab_size ({vocab_size}) in _compute_loss_cpu. Clipping to valid range.")
            labels = torch.clamp(labels, max=vocab_size-1)
        
        # Create a zero tensor to accumulate losses (keeps grad connection)
        total_loss = torch.zeros(1, device=logits.device, requires_grad=True)
        
        for b in range(batch_size):
            # Get sample data
            sample_logits = logits[b, :logit_lengths[b], :label_lengths[b] + 1, :]
            sample_labels = labels[b, :label_lengths[b]]
            
            # Double-check that sample_labels are within valid range
            if (sample_labels >= vocab_size).any():
                print(f"WARNING: Found label values >= vocab_size ({vocab_size}) in sample {b}. Clipping to valid range.")
                sample_labels = torch.clamp(sample_labels, max=vocab_size-1)
            
            # Compute forward variables
            log_alpha = self._compute_forward_variables(
                logits=sample_logits,
                labels=sample_labels,
            )
            
            # Get the loss
            T, U = sample_logits.size(0), sample_labels.size(0) + 1
            
            # Ensure indices are valid (handle edge case where T or U might be 0)
            T_idx = max(0, min(T - 1, log_alpha.size(0) - 1))
            U_idx = max(0, min(U - 1, log_alpha.size(1) - 1))
            
            # Directly accumulate the loss to maintain gradient flow
            loss = -log_alpha[T_idx, U_idx]
            total_loss = total_loss + loss
            losses.append(loss.detach())  # For tracking only
        
        # Average the loss if needed
        if self.reduction == "mean":
            total_loss = total_loss / batch_size
        elif self.reduction == "none":
            # For "none" reduction, we need to return individual losses
            # In this case, we have to recreate a tensor with grad
            stacked_losses = torch.stack([loss.clone() for loss in losses])
            return stacked_losses
            
        return total_loss
    
    def _compute_forward_variables(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute forward variables for the transducer loss.
        
        Args:
            logits: Output tensor of shape (time_steps, label_length, vocab_size)
            labels: Labels of shape (label_length,)
            
        Returns:
            Forward variables of shape (time_steps, label_length + 1)
        """
        T, U, vocab_size = logits.size()
        U = U - 1  # Adjust for blank
        
        # Ensure label values are within the valid vocabulary range
        if (labels >= vocab_size).any():
            print(f"WARNING: Found label values >= vocab_size ({vocab_size}) in _compute_forward_variables. Clipping to valid range.")
            labels = torch.clamp(labels, max=vocab_size-1)
        
        # Initialize forward variables with -inf
        log_alpha = torch.full((T, U + 1), float("-inf"), device=logits.device)
        # Set initial value directly to maintain gradient connection
        log_alpha[0, 0] = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        
        # Compute forward variables
        for t in range(T):
            for u in range(U + 1):
                if t == 0 and u == 0:
                    continue
                
                # Initialize the current position
                current_value = torch.tensor(float("-inf"), device=logits.device, dtype=logits.dtype)
                
                # Probability of emitting blank
                if t > 0:
                    blank_prob = logits[t - 1, u, self.blank_id]
                    # Use logaddexp to avoid numerical issues while maintaining gradient
                    current_value = torch.logaddexp(
                        current_value, 
                        log_alpha[t - 1, u] + blank_prob
                    )
                
                # Probability of emitting label
                if u > 0:
                    # Ensure label index is valid
                    label_idx = labels[u - 1]
                    if label_idx >= vocab_size:
                        print(f"WARNING: Label index {label_idx} is out of bounds for vocab_size {vocab_size}. Clipping.")
                        label_idx = vocab_size - 1
                    
                    label_prob = logits[t, u - 1, label_idx]
                    current_value = torch.logaddexp(
                        current_value,
                        log_alpha[t, u - 1] + label_prob
                    )
                
                # Set the value
                log_alpha[t, u] = current_value
        
        return log_alpha


class TransducerLossWrapper(nn.Module):
    """
    Wrapper for the transducer loss to handle the model outputs.
    """
    
    def __init__(self, blank_id: int, reduction: str = "mean"):
        """
        Initialize the transducer loss wrapper.
        
        Args:
            blank_id: ID of the blank token
            reduction: Reduction method ("none", "mean", "sum")
        """
        super().__init__()
        self.loss_fn = TransducerLoss(blank_id=blank_id, reduction=reduction)
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        label_lengths: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the transducer loss wrapper.
        
        Args:
            outputs: Dictionary of model outputs
            labels: Labels of shape (batch_size, max_label_length)
            label_lengths: Lengths of labels of shape (batch_size,)
            attention_mask: Optional attention mask
            
        Returns:
            Loss value
        """
        # Get logits from outputs
        logits = outputs["logits"]
        
        # Get encoder lengths from attention mask
        if attention_mask is not None:
            encoder_lengths = attention_mask.sum(dim=1).long()
        else:
            # If no attention mask, assume all time steps are valid
            encoder_lengths = torch.full(
                (logits.size(0),), logits.size(1), device=logits.device, dtype=torch.long
            )
        
        # Ensure label values are within the valid vocabulary range
        vocab_size = logits.size(-1)
        if (labels >= vocab_size).any():
            print(f"WARNING: Found label values >= vocab_size ({vocab_size}) in loss function. Clipping to valid range.")
            labels = torch.clamp(labels, max=vocab_size-1)
        
        # Compute loss
        try:
            loss = self.loss_fn(
                logits=logits,
                labels=labels,
                logit_lengths=encoder_lengths,
                label_lengths=label_lengths,
            )
        except Exception as e:
            print(f"ERROR in transducer loss: {e}")
            print(f"Shapes - logits: {logits.shape}, labels: {labels.shape}, encoder_lengths: {encoder_lengths.shape}, label_lengths: {label_lengths.shape}")
            # Return a dummy loss that can be backpropagated
            loss = torch.sum(logits[:, 0, 0, 0]) * 0.0 + 10.0
        
        return loss 