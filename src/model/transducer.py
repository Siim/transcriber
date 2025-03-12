import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass

from model.encoder import XLSREncoder
from model.predictor import TransducerPredictor
from model.joint import TransducerJoint

logger = logging.getLogger(__name__)

@dataclass
class TransducerBeamHypothesis:
    """
    Hypothesis for beam search decoding.
    """
    sequence: List[int]
    score: float
    predictor_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None


class XLSRTransducer(nn.Module):
    """
    XLSR-Transducer model for streaming ASR.
    Combines XLSR encoder, predictor network, and joint network.
    """
    
    def __init__(
        self,
        vocab_size: int,
        blank_id: int = 0,
        encoder_model_path: Optional[str] = None,
        encoder_dim: int = 1024,
        predictor_dim: int = 640,
        predictor_hidden_dim: int = 640,
        joint_dim: int = 640,
        freeze_encoder: bool = True,
        encoder_params: Optional[Dict] = None,
        predictor_params: Optional[Dict] = None,
        joint_params: Optional[Dict] = None,
        debug: bool = False,
    ):
        """
        Initialize the XLSR-Transducer model.
        
        Args:
            vocab_size: Size of vocabulary
            blank_id: ID of blank token (default: 0)
            encoder_model_path: Path to pretrained XLSR model
            encoder_dim: Dimension of encoder output
            predictor_dim: Dimension of predictor output
            predictor_hidden_dim: Dimension of predictor hidden state
            joint_dim: Dimension of joint network
            freeze_encoder: Whether to freeze encoder parameters
            encoder_params: Optional dictionary of parameters for encoder (overrides individual params)
            predictor_params: Optional dictionary of parameters for predictor (overrides individual params)
            joint_params: Optional dictionary of parameters for joint network (overrides individual params)
            debug: Whether to print debug information
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.blank_id = blank_id
        self.debug = debug
        self.logger = logger
        
        # Default parameters
        if encoder_params is None:
            encoder_params = {}
            if encoder_model_path is not None:
                encoder_params["model_path"] = encoder_model_path
            encoder_params["output_dim"] = encoder_dim
            encoder_params["freeze"] = freeze_encoder
        
        if predictor_params is None:
            predictor_params = {}
            predictor_params["embedding_dim"] = predictor_dim
            predictor_params["hidden_dim"] = predictor_hidden_dim
            predictor_params["output_dim"] = predictor_dim
        
        if joint_params is None:
            joint_params = {}
            joint_params["encoder_dim"] = encoder_dim
            joint_params["predictor_dim"] = predictor_dim
            joint_params["hidden_dim"] = joint_dim
        
        # Initialize encoder
        self.encoder = XLSREncoder(**encoder_params)
        
        # Initialize predictor
        predictor_params["vocab_size"] = vocab_size
        self.predictor = TransducerPredictor(**predictor_params)
        
        # Initialize joint network
        joint_params["vocab_size"] = vocab_size
        # Ensure encoder and predictor dimensions match what's in the components
        if "encoder_dim" not in joint_params:
            joint_params["encoder_dim"] = self.encoder.output_dim
        if "predictor_dim" not in joint_params:
            joint_params["predictor_dim"] = predictor_dim
        
        self.joint = TransducerJoint(**joint_params)
    
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        label_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            input_values: Input tensor of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            labels: Labels of shape (batch_size, max_label_length)
            label_lengths: Lengths of labels of shape (batch_size,)
            
        Returns:
            Dictionary containing model outputs
        """
        # Debug info
        if self.debug:
            print(f"Input values shape: {input_values.shape}, device: {input_values.device}")
            if attention_mask is not None:
                print(f"Attention mask shape: {attention_mask.shape}, device: {attention_mask.device}")
            if labels is not None:
                print(f"Labels shape: {labels.shape}, device: {labels.device}")
            if label_lengths is not None:
                print(f"Label lengths shape: {label_lengths.shape}, device: {label_lengths.device}")
        
        try:
            # Encoder forward pass
            encoder_outputs = self.encoder(
                input_values=input_values,
                attention_mask=attention_mask,
            )
            
            # Get encoder outputs
            encoder_hidden_states = encoder_outputs["last_hidden_state"]
            
            # Check for NaN values in encoder outputs
            if torch.isnan(encoder_hidden_states).any():
                self.logger.warning("NaN values detected in encoder hidden states!")
                encoder_hidden_states = torch.nan_to_num(encoder_hidden_states, nan=0.0)
            
            # Ensure encoder outputs require gradients if we're training
            if self.training and not encoder_hidden_states.requires_grad:
                self.logger.warning("Encoder outputs don't require gradients. Creating a differentiable copy.")
                # Create a differentiable copy that maintains the computational graph
                encoder_hidden_states = encoder_hidden_states + 0.0
            
            # Predictor forward pass (if labels are provided)
            if labels is not None and label_lengths is not None:
                # Validate and fix label_lengths
                device = labels.device
                
                # Create a new tensor on CPU first for safety
                fixed_label_lengths = label_lengths.detach().cpu()
                
                # Ensure label_lengths is at least 1
                fixed_label_lengths = torch.clamp(fixed_label_lengths, min=1)
                
                # Ensure label_lengths doesn't exceed labels.size(1)
                fixed_label_lengths = torch.clamp(fixed_label_lengths, max=labels.size(1))
                
                # Move back to the original device
                fixed_label_lengths = fixed_label_lengths.to(device)
                
                if self.debug:
                    print(f"After fixing: Label lengths min: {fixed_label_lengths.min().item()}, max: {fixed_label_lengths.max().item()}")
                
                # Shift labels right by adding blank at the beginning
                blank_tensor = torch.full(
                    (labels.size(0), 1), self.blank_id, dtype=labels.dtype, device=labels.device
                )
                predictor_labels = torch.cat([blank_tensor, labels[:, :-1]], dim=1)
                
                # Forward through predictor - use fixed label lengths
                predictor_outputs = self.predictor(
                    labels=predictor_labels,
                    label_lengths=fixed_label_lengths,
                )
                
                # Get predictor outputs
                predictor_hidden_states = predictor_outputs["outputs"]
                
                # Check for NaN values in predictor outputs
                if torch.isnan(predictor_hidden_states).any():
                    self.logger.warning("NaN values detected in predictor hidden states!")
                    predictor_hidden_states = torch.nan_to_num(predictor_hidden_states, nan=0.0)
                
                # Ensure predictor outputs require gradients if we're training
                if self.training and not predictor_hidden_states.requires_grad:
                    self.logger.warning("Predictor outputs don't require gradients. Creating a differentiable copy.")
                    # Create a differentiable copy that maintains the computational graph
                    predictor_hidden_states = predictor_hidden_states + 0.0
                
                # Joint network forward pass
                logits = self.joint(
                    encoder_outputs=encoder_hidden_states,
                    predictor_outputs=predictor_hidden_states,
                )
                
                # Check for NaN values in logits
                if torch.isnan(logits).any():
                    self.logger.warning("NaN values detected in logits!")
                    logits = torch.nan_to_num(logits, nan=0.0)
                
                # Debug: Check if logits require gradients
                if self.debug and not logits.requires_grad:
                    print("WARNING: Logits do not require gradients after joint network forward pass!")
                
                return {
                    "logits": logits,
                    "encoder_outputs": encoder_hidden_states,
                }
            else:
                # For inference
                return {
                    "encoder_outputs": encoder_hidden_states,
                }
                
        except Exception as e:
            self.logger.error(f"ERROR in transducer forward pass: {e}")
            self.logger.error(f"Shapes - input_values: {input_values.shape}, ")
            if labels is not None:
                self.logger.error(f"labels: {labels.shape}, ")
            if label_lengths is not None:
                self.logger.error(f"label_lengths: {label_lengths.shape}")
            
            # For training, raise the exception for debugging
            if self.training:
                raise
            
            # For inference, return a minimal output to prevent crashes
            return {
                "encoder_outputs": torch.zeros(
                    input_values.size(0), 
                    input_values.size(1) // 320, 
                    self.encoder.output_dim, 
                    device=input_values.device
                ),
            }
    
    def decode_greedy(
        self,
        encoder_outputs: torch.Tensor,
        max_length: int = 100,
    ) -> List[List[int]]:
        """
        Greedy decoding for inference.
        
        Args:
            encoder_outputs: Encoder outputs of shape (batch_size, time_steps, encoder_dim)
            max_length: Maximum length of the decoded sequence
            
        Returns:
            List of decoded sequences (token IDs)
        """
        batch_size, time_steps, _ = encoder_outputs.size()
        device = encoder_outputs.device
        
        # Initialize decoded sequences
        decoded_sequences = [[] for _ in range(batch_size)]
        
        # Initialize predictor states
        predictor_input = torch.full(
            (batch_size, 1), self.blank_id, dtype=torch.long, device=device
        )
        predictor_hidden = None
        
        # Decode each time step
        for t in range(time_steps):
            # Get encoder output for current time step
            encoder_output = encoder_outputs[:, t, :]  # (batch_size, encoder_dim)
            
            # Initialize predictor state for this time step
            predictor_state = predictor_input
            current_hidden = predictor_hidden
            
            # Decode until blank or max_length
            for _ in range(max_length):
                # Forward through predictor
                predictor_output, current_hidden = self.predictor.forward_step(
                    label=predictor_state,
                    hidden=current_hidden,
                )
                
                # Joint network forward pass
                joint_output = self.joint.forward_step(
                    encoder_output=encoder_output,
                    predictor_output=predictor_output,
                )
                
                # Get the most probable token
                token = torch.argmax(joint_output, dim=-1)  # (batch_size,)
                
                # If blank, move to next time step
                if (token == self.blank_id).all():
                    break
                
                # Update decoded sequences for non-blank tokens
                for b in range(batch_size):
                    if token[b] != self.blank_id:
                        decoded_sequences[b].append(token[b].item())
                
                # Update predictor state
                predictor_state = token.unsqueeze(1)  # (batch_size, 1)
            
            # Update predictor hidden state for next time step
            predictor_hidden = current_hidden
            
            # Update predictor input for next time step
            predictor_input = predictor_state
        
        return decoded_sequences
    
    def decode_beam(
        self,
        encoder_outputs: torch.Tensor,
        beam_size: int = 5,
        max_length: int = 100,
    ) -> List[List[int]]:
        """
        Beam search decoding for inference.
        
        Args:
            encoder_outputs: Encoder outputs of shape (batch_size, time_steps, encoder_dim)
            beam_size: Beam size for search
            max_length: Maximum length of the decoded sequence
            
        Returns:
            List of decoded sequences (token IDs)
        """
        batch_size, time_steps, _ = encoder_outputs.size()
        device = encoder_outputs.device
        
        # Initialize decoded sequences
        decoded_sequences = []
        
        # Decode each batch separately
        for b in range(batch_size):
            # Get encoder outputs for current batch
            encoder_output = encoder_outputs[b]  # (time_steps, encoder_dim)
            
            # Initialize beam hypotheses
            blank_tensor = torch.tensor([self.blank_id], dtype=torch.long, device=device)
            predictor_output, predictor_hidden = self.predictor.forward_step(
                label=blank_tensor.unsqueeze(0),
                hidden=None,
            )
            
            # Initialize beam with blank
            beam = [
                TransducerBeamHypothesis(
                    sequence=[],
                    score=0.0,
                    predictor_hidden=predictor_hidden,
                )
            ]
            
            # Decode each time step
            for t in range(time_steps):
                # Get encoder output for current time step
                encoder_t = encoder_output[t].unsqueeze(0)  # (1, encoder_dim)
                
                # Collect candidates for this time step
                candidates = []
                
                # Expand each hypothesis in the beam
                for hypothesis in beam:
                    # Get predictor output for current hypothesis
                    if hypothesis.sequence:
                        # Use last token as input
                        last_token = torch.tensor(
                            [hypothesis.sequence[-1]], dtype=torch.long, device=device
                        ).unsqueeze(0)
                        predictor_output, predictor_hidden = self.predictor.forward_step(
                            label=last_token,
                            hidden=hypothesis.predictor_hidden,
                        )
                    else:
                        # Use blank as input
                        predictor_output, predictor_hidden = self.predictor.forward_step(
                            label=blank_tensor.unsqueeze(0),
                            hidden=hypothesis.predictor_hidden,
                        )
                    
                    # Joint network forward pass
                    joint_output = self.joint.forward_step(
                        encoder_output=encoder_t,
                        predictor_output=predictor_output,
                    )
                    
                    # Convert to log probabilities
                    log_probs = F.log_softmax(joint_output, dim=-1).squeeze(0)
                    
                    # Add candidates for each token
                    for token_id in range(log_probs.size(0)):
                        # Skip if score is too low
                        if log_probs[token_id].item() < -10:
                            continue
                        
                        # Create new sequence
                        new_sequence = hypothesis.sequence.copy()
                        new_score = hypothesis.score + log_probs[token_id].item()
                        
                        # If not blank, add token to sequence
                        if token_id != self.blank_id:
                            new_sequence.append(token_id)
                        
                        # Add candidate
                        candidates.append(
                            TransducerBeamHypothesis(
                                sequence=new_sequence,
                                score=new_score,
                                predictor_hidden=predictor_hidden,
                            )
                        )
                
                # Sort candidates by score and keep top beam_size
                candidates.sort(key=lambda x: x.score, reverse=True)
                beam = candidates[:beam_size]
            
            # Add best hypothesis to decoded sequences
            decoded_sequences.append(beam[0].sequence)
        
        return decoded_sequences
    
    def decode_streaming(
        self,
        input_values: torch.Tensor,
        chunk_size: int = 10,
        buffer_size: int = 30,
        beam_size: int = 5,
    ) -> List[List[int]]:
        """
        Streaming decoding for real-time ASR.
        
        Args:
            input_values: Input tensor of shape (batch_size, seq_len)
            chunk_size: Size of each chunk for streaming
            buffer_size: Size of the buffer for context
            beam_size: Beam size for search
            
        Returns:
            List of decoded sequences (token IDs)
        """
        batch_size, seq_len = input_values.size()
        device = input_values.device
        
        # Initialize decoded sequences
        decoded_sequences = [[] for _ in range(batch_size)]
        
        # Initialize predictor states
        predictor_input = torch.full(
            (batch_size, 1), self.blank_id, dtype=torch.long, device=device
        )
        predictor_hidden = None
        
        # Process input in chunks
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            
            # Get chunk with buffer
            buffer_start = max(0, start - buffer_size)
            chunk_with_buffer = input_values[:, buffer_start:end]
            
            # Forward through encoder
            encoder_outputs = self.encoder(
                input_values=chunk_with_buffer,
            )["last_hidden_state"]
            
            # Get only the current chunk outputs (remove buffer)
            chunk_offset = start - buffer_start
            chunk_encoder_outputs = encoder_outputs[:, chunk_offset:chunk_offset + (end - start), :]
            
            # Decode chunk
            for t in range(chunk_encoder_outputs.size(1)):
                # Get encoder output for current time step
                encoder_output = chunk_encoder_outputs[:, t, :]  # (batch_size, encoder_dim)
                
                # Forward through predictor
                predictor_output, predictor_hidden = self.predictor.forward_step(
                    label=predictor_input,
                    hidden=predictor_hidden,
                )
                
                # Joint network forward pass
                joint_output = self.joint.forward_step(
                    encoder_output=encoder_output,
                    predictor_output=predictor_output,
                )
                
                # Get the most probable token
                token = torch.argmax(joint_output, dim=-1)  # (batch_size,)
                
                # Update decoded sequences for non-blank tokens
                for b in range(batch_size):
                    if token[b] != self.blank_id:
                        decoded_sequences[b].append(token[b].item())
                
                # Update predictor input for next time step
                predictor_input = token.unsqueeze(1)  # (batch_size, 1)
        
        return decoded_sequences 