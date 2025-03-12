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
    
    This implementation can use different backends:
    1. CUDA implementation (warp-rnnt) - fastest for CUDA GPUs
    2. PyTorch implementation with torchaudio - works with both CUDA and MPS
    3. Pure PyTorch CPU implementation - slowest fallback
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
        
        # Track which implementation is being used
        self.implementation = "cpu"
        
        # Try to import GPU implementations in order of preference
        
        # 1. Try warp-rnnt (CUDA only, fastest)
        try:
            import warp_rnnt
            self.warp_rnnt = warp_rnnt
            self.implementation = "warp-rnnt"
            logger.info("Using warp-rnnt CUDA implementation of RNN-T loss (fastest)")
            return
        except ImportError:
            logger.info("warp-rnnt not available, trying other implementations...")
        
        # 2. Try torchaudio.functional.rnnt_loss (works on CUDA and MPS)
        try:
            import torchaudio.functional as F_audio
            self.F_audio = F_audio
            
            # Check if we have a proper GPU (CUDA or MPS)
            if torch.cuda.is_available() or hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.implementation = "torchaudio"
                device_type = "CUDA" if torch.cuda.is_available() else "MPS"
                logger.info(f"Using torchaudio implementation of RNN-T loss (optimized for {device_type})")
                return
        except (ImportError, AttributeError):
            logger.info("torchaudio.functional.rnnt_loss not available, continuing...")
        
        # 3. Try fast_rnnt (optimized CPU/CUDA implementation)
        try:
            import fast_rnnt
            self.fast_rnnt = fast_rnnt
            self.implementation = "fast-rnnt"
            logger.info("Using fast-rnnt implementation of RNN-T loss")
            return
        except ImportError:
            logger.info("fast-rnnt not available, falling back to CPU implementation")
        
        # 4. Fallback to pure PyTorch CPU implementation
        logger.info("Using pure PyTorch CPU implementation of RNN-T loss (slowest)")
    
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
        
        # Choose implementation based on device and available libraries
        loss = None
        implementation_tried = []
        
        # Attempt to use the implementations in order of preference
        if self.implementation == "warp-rnnt" and device.type == "cuda":
            implementation_tried.append("warp-rnnt")
            try:
                logger.debug("Attempting to use warp-rnnt implementation...")
                loss = self._compute_loss_warp_rnnt(
                    logits, labels_clipped, input_lengths, label_lengths
                )
                logger.debug("warp-rnnt implementation successful")
            except Exception as e:
                logger.warning(f"Error using warp-rnnt implementation: {e}. Trying next implementation.")
        
        if loss is None and self.implementation in ["warp-rnnt", "torchaudio"] and (device.type == "cuda" or device.type == "mps"):
            implementation_tried.append("torchaudio")
            try:
                logger.debug("Attempting to use torchaudio implementation...")
                loss = self._compute_loss_torchaudio(
                    logits, labels_clipped, input_lengths, label_lengths
                )
                logger.debug("torchaudio implementation successful")
            except Exception as e:
                logger.warning(f"Error using torchaudio implementation: {e}. Trying next implementation.")
        
        if loss is None and self.implementation in ["warp-rnnt", "torchaudio", "fast-rnnt"] and (device.type == "cuda" or device.type == "cpu"):
            implementation_tried.append("fast-rnnt")
            try:
                logger.debug("Attempting to use fast-rnnt implementation...")
                loss = self._compute_loss_fast_rnnt(
                    logits, labels_clipped, input_lengths, label_lengths
                )
                logger.debug("fast-rnnt implementation successful")
            except Exception as e:
                logger.warning(f"Error using fast-rnnt implementation: {e}. Falling back to CPU implementation.")
        
        # Fallback to CPU implementation if all else fails
        if loss is None:
            implementation_tried.append("cpu")
            logger.info(f"Using CPU implementation after trying: {', '.join(implementation_tried)}")
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
    
    def _compute_loss_warp_rnnt(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        input_lengths: torch.Tensor,
        label_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the transducer loss using the warp-rnnt CUDA implementation.
        
        Args:
            logits: Joint network outputs [batch_size, max_time, max_label, vocab_size]
            labels: Label sequences [batch_size, max_label_length]
            input_lengths: Length of input sequences [batch_size]
            label_lengths: Length of label sequences [batch_size]
            
        Returns:
            Loss tensor
        """
        try:
            # Ensure contiguous memory layout for CUDA
            log_probs = F.log_softmax(logits, dim=-1).contiguous()
            
            # According to warp-rnnt documentation:
            # - acts: (B, T, U, V) where B is batch size, T is input length, U is target length, and V is vocab size
            # - labels: (B, U-1) where U-1 is target length without blank
            # - act_lens: (B) input lengths
            # - label_lens: (B) label lengths
            # https://github.com/HawkAaron/warp-transducer#c-interface
            
            batch_size = labels.size(0)
            device = labels.device
            
            # Trim labels according to actual lengths
            # warp-rnnt expects labels without blanks, just the real target symbols
            trimmed_labels = []
            max_length = 0
            
            for b in range(batch_size):
                length = min(label_lengths[b].item(), labels.size(1))
                if length > 0:
                    # Only include non-blank labels up to the specified length
                    trimmed_labels.append(labels[b, :length])
                    max_length = max(max_length, length)
                else:
                    # Handle edge case of zero length
                    trimmed_labels.append(torch.tensor([0], device=device))
                    max_length = max(max_length, 1)
            
            # Create properly sized tensor for the trimmed labels
            packed_labels = torch.zeros(batch_size, max_length, dtype=torch.int, device=device)
            for b in range(batch_size):
                actual_length = trimmed_labels[b].size(0)
                packed_labels[b, :actual_length] = trimmed_labels[b]
            
            # Double-check for NaN or inf values in log_probs
            if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
                logger.warning("NaN or inf values detected in log_probs before warp-rnnt. Replacing with zeros.")
                log_probs = torch.nan_to_num(log_probs, nan=0.0, posinf=0.0, neginf=-100.0)
            
            # Make sure all tensors are on the same device and have the right type
            input_lengths_cuda = input_lengths.int().to(device)
            label_lengths_cuda = label_lengths.int().to(device)
            
            # Debug logs
            logger.debug(f"Warp-RNNT input shapes: log_probs={log_probs.shape}, packed_labels={packed_labels.shape}")
            logger.debug(f"Warp-RNNT length shapes: input_lengths={input_lengths_cuda.shape}, label_lengths={label_lengths_cuda.shape}")
            
            # Call warp-rnnt with the properly formatted inputs
            loss = self.warp_rnnt.rnnt_loss(
                log_probs,
                packed_labels,
                input_lengths_cuda,
                label_lengths_cuda,
                blank=self.blank_id,
                reduction=self.reduction
            )
            
            # Check for NaN loss
            if torch.isnan(loss).any():
                logger.warning("NaN loss detected from warp-rnnt. Using a fallback value.")
                return torch.tensor(100.0, device=device, requires_grad=True)
            
            return loss
        
        except Exception as e:
            # Detailed error reporting for debugging
            logger.warning(f"Error in warp-rnnt: {e}")
            logger.warning(f"Input shapes - logits: {logits.shape}, labels: {labels.shape}")
            logger.warning(f"Input lengths - input: {input_lengths}, label: {label_lengths}")
            
            # Raise the exception to let the caller handle fallback
            raise e
    
    def _compute_loss_torchaudio(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        input_lengths: torch.Tensor,
        label_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the transducer loss using torchaudio implementation.
        Works on both CUDA and MPS.
        
        Args:
            logits: Joint network outputs [batch_size, max_time, max_label, vocab_size]
            labels: Label sequences [batch_size, max_label_length]
            input_lengths: Length of input sequences [batch_size]
            label_lengths: Length of label sequences [batch_size]
            
        Returns:
            Loss tensor
        """
        # Ensure contiguous memory layout
        log_probs = F.log_softmax(logits, dim=-1).contiguous()
        
        # torchaudio expects [T, B, U, V] layout but our tensor is [B, T, U, V]
        # so we need to permute
        log_probs = log_probs.permute(1, 0, 2, 3)
        
        # Compute the loss
        neg_log_likelihood = self.F_audio.rnnt_loss(
            log_probs,
            labels,
            input_lengths,
            label_lengths,
            blank=self.blank_id,
            reduction=self.reduction
        )
        
        return neg_log_likelihood
    
    def _compute_loss_fast_rnnt(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        input_lengths: torch.Tensor,
        label_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the transducer loss using fast-rnnt implementation.
        
        Args:
            logits: Joint network outputs [batch_size, max_time, max_label, vocab_size]
            labels: Label sequences [batch_size, max_label_length]
            input_lengths: Length of input sequences [batch_size]
            label_lengths: Length of label sequences [batch_size]
            
        Returns:
            Loss tensor
        """
        # Ensure contiguous memory layout
        log_probs = F.log_softmax(logits, dim=-1).contiguous()
        
        # Compute the loss
        loss = self.fast_rnnt.rnnt_loss_simple(
            log_probs,
            labels,
            input_lengths,
            label_lengths,
            blank=self.blank_id,
            reduction=self.reduction
        )
        
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
            # Make sure parameter names match the TransducerLoss.forward method
            loss = self.loss_fn(
                outputs={"logits": logits},  # TransducerLoss expects a dict with 'logits'
                labels=labels,
                label_lengths=label_lengths,
                attention_mask=attention_mask,
            )
        except Exception as e:
            print(f"ERROR in transducer loss: {e}")
            print(f"Shapes - logits: {logits.shape}, labels: {labels.shape}, encoder_lengths: {encoder_lengths.shape}, label_lengths: {label_lengths.shape}")
            # Return a dummy loss that can be backpropagated
            loss = torch.sum(logits[:, 0, 0, 0]) * 0.0 + 10.0
        
        return loss 