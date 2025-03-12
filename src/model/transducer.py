import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from .encoder import XLSREncoder
from .predictor import TransducerPredictor
from .joint import TransducerJoint


@dataclass
class TransducerBeamHypothesis:
    """Hypothesis for beam search decoding."""
    
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
        blank_id: int,
        encoder_params: Dict = None,
        predictor_params: Dict = None,
        joint_params: Dict = None,
    ):
        super().__init__()
        
        # Default parameters
        if encoder_params is None:
            encoder_params = {}
        if predictor_params is None:
            predictor_params = {}
        if joint_params is None:
            joint_params = {}
        
        # Initialize encoder
        self.encoder = XLSREncoder(**encoder_params)
        
        # Initialize predictor
        predictor_params["vocab_size"] = vocab_size
        self.predictor = TransducerPredictor(**predictor_params)
        
        # Initialize joint network
        joint_params["encoder_dim"] = self.encoder.output_dim
        joint_params["predictor_dim"] = self.predictor.output_dim
        joint_params["vocab_size"] = vocab_size
        self.joint = TransducerJoint(**joint_params)
        
        # Blank token ID
        self.blank_id = blank_id
    
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        label_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the transducer model.
        
        Args:
            input_values: Input tensor of shape (batch_size, seq_len)
            attention_mask: Optional mask of shape (batch_size, seq_len)
            labels: Optional labels of shape (batch_size, max_label_length)
            label_lengths: Optional lengths of labels of shape (batch_size,)
            
        Returns:
            Dictionary containing:
                - logits: Output tensor of shape (batch_size, time_steps, label_length, vocab_size)
                - encoder_outputs: Encoder outputs of shape (batch_size, time_steps, encoder_dim)
        """
        # Ensure training mode is set correctly
        if self.training:
            self.encoder.train()
            self.predictor.train()
            self.joint.train()
        
        # Check for NaN values in input
        if torch.isnan(input_values).any():
            print("WARNING: NaN values detected in model input!")
            input_values = torch.nan_to_num(input_values, nan=0.0)
        
        # Encoder forward pass
        encoder_outputs = self.encoder(
            input_values=input_values,
            attention_mask=attention_mask,
        )
        
        # Get encoder outputs
        encoder_hidden_states = encoder_outputs["last_hidden_state"]
        
        # Check for NaN values in encoder outputs
        if torch.isnan(encoder_hidden_states).any():
            print("WARNING: NaN values detected in encoder hidden states!")
            encoder_hidden_states = torch.nan_to_num(encoder_hidden_states, nan=0.0)
        
        # Predictor forward pass (if labels are provided)
        if labels is not None and label_lengths is not None:
            # Shift labels right by adding blank at the beginning
            blank_tensor = torch.full(
                (labels.size(0), 1), self.blank_id, dtype=labels.dtype, device=labels.device
            )
            predictor_labels = torch.cat([blank_tensor, labels[:, :-1]], dim=1)
            
            # Forward through predictor
            predictor_outputs = self.predictor(
                labels=predictor_labels,
                label_lengths=label_lengths,
            )
            
            # Get predictor outputs
            predictor_hidden_states = predictor_outputs["outputs"]
            
            # Check for NaN values in predictor outputs
            if torch.isnan(predictor_hidden_states).any():
                print("WARNING: NaN values detected in predictor hidden states!")
                predictor_hidden_states = torch.nan_to_num(predictor_hidden_states, nan=0.0)
            
            # Joint network forward pass
            logits = self.joint(
                encoder_outputs=encoder_hidden_states,
                predictor_outputs=predictor_hidden_states,
            )
            
            # Check for NaN values in logits
            if torch.isnan(logits).any():
                print("WARNING: NaN values detected in logits!")
                logits = torch.nan_to_num(logits, nan=0.0)
            
            # Debug: Check if logits require gradients
            if not logits.requires_grad:
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