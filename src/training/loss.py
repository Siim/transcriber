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
        if self.use_cuda and logits.is_cuda:
            # Use CUDA implementation
            loss = self.rnnt_loss(
                logits=logits,
                targets=labels,
                logit_lengths=logit_lengths,
                target_lengths=label_lengths,
                blank=self.blank_id,
                reduction=self.reduction,
            )
        else:
            # Use CPU implementation
            loss = self._compute_loss_cpu(
                logits=logits,
                labels=labels,
                logit_lengths=logit_lengths,
                label_lengths=label_lengths,
            )
        
        return loss
    
    def _compute_loss_cpu(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        logit_lengths: torch.Tensor,
        label_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the transducer loss on CPU.
        This is a simplified implementation and may be slow for large inputs.
        
        Args:
            logits: Output tensor of shape (batch_size, time_steps, label_length, vocab_size)
            labels: Labels of shape (batch_size, max_label_length)
            logit_lengths: Lengths of logits of shape (batch_size,)
            label_lengths: Lengths of labels of shape (batch_size,)
            
        Returns:
            Loss value
        """
        batch_size = logits.size(0)
        losses = []
        
        # Compute loss for each sample in the batch
        for b in range(batch_size):
            # Get sample data
            sample_logits = logits[b, :logit_lengths[b], :label_lengths[b] + 1, :]
            sample_labels = labels[b, :label_lengths[b]]
            
            # Compute forward variables
            log_alpha = self._compute_forward_variables(
                logits=sample_logits,
                labels=sample_labels,
            )
            
            # Get the loss
            T, U = sample_logits.size(0), sample_labels.size(0) + 1
            loss = -log_alpha[T - 1, U - 1]
            losses.append(loss)
        
        # Apply reduction
        if self.reduction == "none":
            return torch.tensor(losses, device=logits.device)
        elif self.reduction == "mean":
            return torch.mean(torch.tensor(losses, device=logits.device))
        elif self.reduction == "sum":
            return torch.sum(torch.tensor(losses, device=logits.device))
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")
    
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
        T, U, _ = logits.size()
        U = U - 1  # Adjust for blank
        
        # Initialize forward variables
        log_alpha = torch.full((T, U + 1), float("-inf"), device=logits.device)
        log_alpha[0, 0] = 0.0
        
        # Compute forward variables
        for t in range(T):
            for u in range(U + 1):
                if t == 0 and u == 0:
                    continue
                
                # Probability of emitting blank
                if t > 0:
                    blank_prob = logits[t - 1, u, self.blank_id]
                    log_alpha[t, u] = log_alpha[t - 1, u] + blank_prob
                
                # Probability of emitting label
                if u > 0:
                    label_prob = logits[t, u - 1, labels[u - 1]]
                    log_alpha[t, u] = torch.logaddexp(
                        log_alpha[t, u], log_alpha[t, u - 1] + label_prob
                    )
        
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
            outputs: Model outputs containing logits
            labels: Labels of shape (batch_size, max_label_length)
            label_lengths: Lengths of labels of shape (batch_size,)
            attention_mask: Optional mask of shape (batch_size, seq_len)
            
        Returns:
            Loss value
        """
        logits = outputs["logits"]
        
        # Compute logit lengths from attention mask
        if attention_mask is not None:
            logit_lengths = attention_mask.sum(dim=1).long()
        else:
            logit_lengths = torch.full(
                (logits.size(0),), logits.size(1), dtype=torch.long, device=logits.device
            )
        
        # Compute loss
        loss = self.loss_fn(
            logits=logits,
            labels=labels,
            logit_lengths=logit_lengths,
            label_lengths=label_lengths,
        )
        
        return loss 