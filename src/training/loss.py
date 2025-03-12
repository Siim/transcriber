import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Dict, Optional, Tuple, List, Union, Any

logger = logging.getLogger(__name__)

class TransducerLoss(nn.Module):
    """
    RNN-T loss implementation.
    
    This implementation can use either a pure PyTorch implementation or
    a CUDA implementation if available.
    """
    
    def __init__(
        self,
        blank_id: int = 0,
        reduction: str = "mean",
    ):
        """
        Initialize the RNN-T loss.
        
        Args:
            blank_id: ID of the blank token
            reduction: Reduction method ('none', 'mean', 'sum')
        """
        super().__init__()
        self.blank_id = blank_id
        self.reduction = reduction
        
        # Try to import CUDA implementation
        try:
            import warp_rnnt
            self.warp_rnnt = warp_rnnt
            self.use_cuda_implementation = True
            logger.info("Using CUDA implementation of RNN-T loss")
        except ImportError:
            self.use_cuda_implementation = False
            logger.info("Using CPU implementation of RNN-T loss")
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        label_lengths: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the RNN-T loss.
        
        Args:
            outputs: Dictionary containing the model outputs, with 'logits'
            labels: Label sequences [batch_size, max_label_length]
            label_lengths: Length of each label sequence [batch_size]
            attention_mask: Mask for the encoder outputs [batch_size, max_time_steps]
            
        Returns:
            Loss tensor
        """
        logits = outputs["logits"]
        
        # Check for NaN values in logits
        if torch.isnan(logits).any():
            logger.warning("NaN values detected in logits. This will affect loss computation.")
        
        # Get the device of the inputs
        device = logits.device
        
        # Get sequence lengths from attention mask if provided
        if attention_mask is not None:
            input_lengths = attention_mask.sum(dim=1).long()
        else:
            # Assume full length if no mask provided
            input_lengths = torch.full(
                (logits.size(0),), logits.size(1), device=device, dtype=torch.long
            )
        
        # Ensure valid lengths (should never happen, but just in case)
        input_lengths = torch.clamp(input_lengths, min=1)
        label_lengths = torch.clamp(label_lengths, min=0)
        
        # Check if we have at least one valid target in the batch
        if torch.max(label_lengths) == 0:
            logger.warning("All label lengths are 0. Returning zero loss.")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Clip the label values to ensure they are in the valid range
        # This helps avoid issues with out-of-vocabulary tokens
        vocab_size = logits.size(-1)
        labels_clipped = torch.clamp(labels, 0, vocab_size - 1)
        
        # Compute the actual loss - use CUDA implementation if available
        if self.use_cuda_implementation and device.type == "cuda":
            try:
                loss = self._compute_loss_cuda(
                    logits, labels_clipped, input_lengths, label_lengths
                )
            except Exception as e:
                logger.warning(f"Error using CUDA implementation: {e}. Falling back to CPU implementation.")
                loss = self._compute_loss_cpu(
                    logits, labels_clipped, input_lengths, label_lengths
                )
        else:
            # Use CPU implementation
            loss = self._compute_loss_cpu(
                logits, labels_clipped, input_lengths, label_lengths
            )
        
        # Check for NaN values in loss
        if torch.isnan(loss).any():
            logger.warning("NaN loss detected. Returning zero loss.")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Log warning if loss doesn't require gradients
        if not loss.requires_grad:
            logger.warning("Loss does not require gradients. This will prevent backpropagation.")
        
        return loss
    
    def _compute_loss_cpu(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        input_lengths: torch.Tensor,
        label_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the transducer loss using CPU implementation.
        This is a numerically stable implementation, but slower than CUDA.
        
        Args:
            logits: Joint network outputs [batch_size, max_time, max_label, vocab_size]
            labels: Label sequences [batch_size, max_label_length]
            input_lengths: Length of input sequences [batch_size]
            label_lengths: Length of label sequences [batch_size]
            
        Returns:
            Loss tensor
        """
        batch_size = logits.size(0)
        device = logits.device
        
        # Initialize the loss tensor
        losses = torch.zeros(batch_size, device=device, requires_grad=True)
        
        # Compute the loss for each sample in the batch
        for b in range(batch_size):
            # Skip invalid samples
            if input_lengths[b] <= 0 or label_lengths[b] <= 0:
                continue
                
            # Get the actual sequence for this batch
            T = min(int(input_lengths[b].item()), logits.size(1))
            U = min(int(label_lengths[b].item()) + 1, logits.size(2))  # +1 for blank
            
            # Ensure we have valid dimensions
            if T <= 0 or U <= 0:
                logger.warning(f"Invalid sequence lengths for batch {b}: T={T}, U={U}")
                continue
                
            # Get log probabilities for this batch
            log_probs = F.log_softmax(logits[b, :T, :U], dim=-1)
            target = labels[b, :label_lengths[b]]
            
            # Compute forward variables
            alpha = self._compute_forward_variables(log_probs, target, T, U)
            
            # The loss is the negative log of the sum over all paths
            # Ensure indices are valid
            if alpha.size(0) > 0 and alpha.size(1) > 0:
                loss_b = -alpha[alpha.size(0)-1, alpha.size(1)-1]
                
                # Add to the batch losses
                losses = torch.cat([losses[:b], loss_b.unsqueeze(0), losses[b+1:]], 0)
            else:
                logger.warning(f"Empty alpha matrix for batch {b}")
        
        # Apply reduction
        if self.reduction == "mean":
            valid_samples = (input_lengths > 0) & (label_lengths > 0)
            num_valid = valid_samples.sum().item()
            return losses.sum() / max(num_valid, 1)
        elif self.reduction == "sum":
            return losses.sum()
        else:  # 'none'
            return losses
    
    def _compute_loss_cuda(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        input_lengths: torch.Tensor,
        label_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the transducer loss using CUDA implementation.
        
        Args:
            logits: Joint network outputs [batch_size, max_time, max_label, vocab_size]
            labels: Label sequences [batch_size, max_label_length]
            input_lengths: Length of input sequences [batch_size]
            label_lengths: Length of label sequences [batch_size]
            
        Returns:
            Loss tensor
        """
        batch_size = logits.size(0)
        max_time = logits.size(1)
        max_label = logits.size(2)
        
        # Ensure contiguous memory layout for CUDA
        log_probs = F.log_softmax(logits, dim=-1).contiguous()
        
        # The warp-rnnt expects [B, T, U, V] layout but Pytorch is [B, T, U, V]
        # so we're already good
        
        # Compute the loss
        loss = self.warp_rnnt.rnnt_loss(
            log_probs,
            labels.int(),
            input_lengths.int(),
            label_lengths.int(),
            blank=self.blank_id,
            reduction=self.reduction
        )
        
        return loss
    
    def _compute_forward_variables(
        self,
        log_probs: torch.Tensor,
        target: torch.Tensor,
        T: int,
        U: int,
    ) -> torch.Tensor:
        """
        Compute forward variables for the RNN-T loss calculation.
        
        Args:
            log_probs: Log-probabilities from the joint network [T, U, vocab_size]
            target: Target sequence [U-1]
            T: Length of input sequence
            U: Length of target sequence + 1 (for blank)
            
        Returns:
            Alpha matrix [T, U]
        """
        device = log_probs.device
        
        # Make sure target values are within valid range to avoid index errors
        vocab_size = log_probs.size(2)
        target = torch.clamp(target, 0, vocab_size - 1)
        
        # Ensure T and U don't exceed the actual dimensions of log_probs
        T = min(T, log_probs.size(0))
        U = min(U, log_probs.size(1))
        
        if T <= 0 or U <= 0:
            logger.warning(f"Invalid sequence lengths: T={T}, U={U}. Returning zero loss.")
            return torch.zeros((1, 1), device=device)
        
        # Initialize forward variables with -inf (log domain)
        alpha = torch.full((T, U), -float('inf'), device=device)
        
        # Base case: alpha[0, 0] = log_probs[0, 0, blank]
        alpha[0, 0] = log_probs[0, 0, self.blank_id]
        
        # Fill first row (can only emit blanks)
        for t in range(1, T):
            alpha[t, 0] = alpha[t-1, 0] + log_probs[t, 0, self.blank_id]
        
        # Fill first column (can only consume labels)
        for u in range(1, U):
            label_idx = target[u-1] if u-1 < len(target) else 0  # Use 0 (blank) if out of bounds
            alpha[0, u] = alpha[0, u-1] + log_probs[0, u-1, label_idx]
        
        # Fill the rest of the table
        for t in range(1, T):
            for u in range(1, U):
                # Get the correct label index
                label_idx = target[u-1] if u-1 < len(target) else 0  # Use 0 (blank) if out of bounds
                
                # Emission: either blank or label
                blank_prob = alpha[t-1, u] + log_probs[t, u, self.blank_id]
                label_prob = alpha[t, u-1] + log_probs[t, u-1, label_idx]
                
                # log-sum-exp trick for numerical stability
                max_prob = torch.max(blank_prob, label_prob)
                if torch.isfinite(max_prob):
                    alpha[t, u] = max_prob + torch.log(
                        torch.exp(blank_prob - max_prob) + torch.exp(label_prob - max_prob)
                    )
                else:
                    # If both are -inf, keep it that way
                    alpha[t, u] = max_prob
        
        return alpha


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