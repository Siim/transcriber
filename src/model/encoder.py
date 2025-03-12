import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from transformers import Wav2Vec2Model, Wav2Vec2Config
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2EncoderLayer,
    Wav2Vec2Attention
)


class StreamingWav2Vec2Attention(nn.Module):
    """
    Modified Wav2Vec2 attention module with streaming capabilities.
    Supports different attention mask types: full, chunk, and attention_sink.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        attention_mask_type: str = "chunk",
        chunk_size: int = 10,
        left_context: int = 25,
        right_context: int = 0,
        attention_sink_size: int = 4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.attention_mask_type = attention_mask_type
        self.chunk_size = chunk_size
        self.left_context = left_context
        self.right_context = right_context
        self.attention_sink_size = attention_sink_size
        
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
    
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    
    def _create_chunk_attention_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create a chunked attention mask for streaming inference."""
        # Initialize with all -inf (no attention)
        mask = torch.full(
            (seq_len, seq_len), float("-inf"), device=device
        )
        
        # For each position, allow attention to positions within the same chunk
        # and to positions in the left context and right context
        for i in range(seq_len):
            # Current chunk start and end
            chunk_idx = i // self.chunk_size
            chunk_start = chunk_idx * self.chunk_size
            chunk_end = min((chunk_idx + 1) * self.chunk_size, seq_len)
            
            # Left context start
            left_start = max(0, i - self.left_context)
            
            # Right context end
            right_end = min(seq_len, i + self.right_context + 1)
            
            # Allow attention within the chunk
            mask[i, chunk_start:chunk_end] = 0.0
            
            # Allow attention to left context
            mask[i, left_start:chunk_start] = 0.0
            
            # Allow attention to right context (if any)
            if right_end > chunk_end:
                mask[i, chunk_end:right_end] = 0.0
        
        return mask
    
    def _create_attention_sink_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create an attention mask with attention sinks for streaming inference."""
        # Initialize with all -inf (no attention)
        mask = torch.full(
            (seq_len, seq_len), float("-inf"), device=device
        )
        
        # For each position, allow attention to:
        # 1. The first attention_sink_size frames (attention sink)
        # 2. Positions within the left context
        for i in range(seq_len):
            # Attention sink (first few frames)
            mask[i, :self.attention_sink_size] = 0.0
            
            # Left context
            left_start = max(0, i - self.left_context)
            mask[i, left_start:i+1] = 0.0
        
        return mask
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with streaming attention.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, embed_dim)
            attention_mask: Optional mask of shape (batch_size, seq_len)
            output_attentions: Whether to return attention weights
            
        Returns:
            output: Output tensor of shape (batch_size, seq_len, embed_dim)
            attention_weights: Optional attention weights
        """
        bsz, seq_len, _ = hidden_states.size()
        
        # Project queries, keys, and values
        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, -1, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        
        # Calculate attention weights
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        
        # Apply streaming attention mask based on the specified type
        if self.attention_mask_type != "full":
            if self.attention_mask_type == "chunk":
                streaming_mask = self._create_chunk_attention_mask(seq_len, hidden_states.device)
            elif self.attention_mask_type == "attention_sink":
                streaming_mask = self._create_attention_sink_mask(seq_len, hidden_states.device)
            else:
                raise ValueError(f"Unknown attention mask type: {self.attention_mask_type}")
            
            # Add streaming mask to attention weights
            attn_weights = attn_weights.view(bsz, self.num_heads, seq_len, seq_len)
            attn_weights = attn_weights + streaming_mask.unsqueeze(0).unsqueeze(0)
            attn_weights = attn_weights.view(bsz * self.num_heads, seq_len, seq_len)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # First, handle attention_mask with any number of dimensions
            # Debug: print the shape
            print(f"Original attention_mask shape: {attention_mask.shape}")
            
            # Ensure we have a 2D attention mask [batch_size, seq_len]
            if attention_mask.dim() > 2:
                # Flatten all extra dimensions
                while attention_mask.dim() > 2:
                    attention_mask = attention_mask.squeeze(1)
                print(f"Squeezed attention_mask shape: {attention_mask.shape}")
            
            # Create a 4D attention mask [batch_size, 1, 1, seq_len]
            expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            print(f"Expanded mask shape after unsqueeze: {expanded_mask.shape}")
            
            # Create attention mask where 0s -> -10000.0 and 1s -> 0.0
            attention_mask_float = (1.0 - expanded_mask) * -10000.0
            print(f"Final attention_mask_float shape: {attention_mask_float.shape}")
            
            # Reshape attention weights to 4D [batch_size, num_heads, seq_len, seq_len]
            attn_weights_4d = attn_weights.view(bsz, self.num_heads, seq_len, seq_len)
            
            # Add the mask to the attention weights
            attn_weights_4d = attn_weights_4d + attention_mask_float
            
            # Reshape back to 3D [batch_size * num_heads, seq_len, seq_len]
            attn_weights = attn_weights_4d.view(bsz * self.num_heads, seq_len, seq_len)
        
        # Apply softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # Apply attention to values
        attn_output = torch.bmm(attn_probs, value_states)
        attn_output = attn_output.view(bsz, self.num_heads, seq_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, self.embed_dim)
        
        # Project back to embedding dimension
        attn_output = self.out_proj(attn_output)
        
        if output_attentions:
            attn_weights = attn_weights.view(bsz, self.num_heads, seq_len, seq_len)
        else:
            attn_weights = None
        
        return attn_output, attn_weights


class StreamingWav2Vec2EncoderLayer(nn.Module):
    """
    Modified Wav2Vec2 encoder layer with streaming attention.
    """
    
    def __init__(
        self,
        config: Wav2Vec2Config,
        attention_mask_type: str = "chunk",
        chunk_size: int = 10,
        left_context: int = 25,
        right_context: int = 0,
        attention_sink_size: int = 4,
    ):
        super().__init__()
        self.attention = StreamingWav2Vec2Attention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            attention_mask_type=attention_mask_type,
            chunk_size=chunk_size,
            left_context=left_context,
            right_context=right_context,
            attention_sink_size=attention_sink_size,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the encoder layer.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional mask of shape (batch_size, seq_len)
            output_attentions: Whether to return attention weights
            
        Returns:
            hidden_states: Output tensor of shape (batch_size, seq_len, hidden_size)
            attn_weights: Optional attention weights
        """
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, attn_weights = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        
        # Feed forward
        ff_residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = ff_residual + hidden_states
        
        return hidden_states, attn_weights


class XLSREncoder(nn.Module):
    """
    XLSR encoder with streaming capabilities.
    Uses the XLSR-53 model as the base and modifies it for streaming inference.
    """
    
    def __init__(
        self,
        pretrained_model_name: str = "facebook/wav2vec2-large-xlsr-53",
        freeze_feature_encoder: bool = True,
        freeze_base_model: bool = False,
        dropout: float = 0.1,
        layerdrop: float = 0.1,
        attention_mask_type: str = "chunk",
        chunk_size: int = 10,
        left_context: int = 25,
        right_context: int = 0,
        attention_sink_size: int = 4,
    ):
        super().__init__()
        
        # Load pretrained model
        self.model = Wav2Vec2Model.from_pretrained(pretrained_model_name)
        config = self.model.config
        
        # Set dropout and layerdrop
        config.hidden_dropout = dropout
        config.attention_dropout = dropout
        config.activation_dropout = dropout
        config.layerdrop = layerdrop
        
        # Freeze feature encoder if specified
        if freeze_feature_encoder:
            for param in self.model.feature_extractor.parameters():
                param.requires_grad = False
        
        # Freeze base model if specified
        if freeze_base_model:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Replace encoder layers with streaming encoder layers if using streaming
        if attention_mask_type != "full":
            streaming_layers = []
            for i, layer in enumerate(self.model.encoder.layers):
                streaming_layer = StreamingWav2Vec2EncoderLayer(
                    config=config,
                    attention_mask_type=attention_mask_type,
                    chunk_size=chunk_size,
                    left_context=left_context,
                    right_context=right_context,
                    attention_sink_size=attention_sink_size,
                )
                
                # Copy weights from original layer
                streaming_layer.attention.k_proj.weight.data = layer.attention.k_proj.weight.data.clone()
                streaming_layer.attention.k_proj.bias.data = layer.attention.k_proj.bias.data.clone()
                streaming_layer.attention.v_proj.weight.data = layer.attention.v_proj.weight.data.clone()
                streaming_layer.attention.v_proj.bias.data = layer.attention.v_proj.bias.data.clone()
                streaming_layer.attention.q_proj.weight.data = layer.attention.q_proj.weight.data.clone()
                streaming_layer.attention.q_proj.bias.data = layer.attention.q_proj.bias.data.clone()
                streaming_layer.attention.out_proj.weight.data = layer.attention.out_proj.weight.data.clone()
                streaming_layer.attention.out_proj.bias.data = layer.attention.out_proj.bias.data.clone()
                
                # Copy feed forward weights
                streaming_layer.feed_forward[0].weight.data = layer.feed_forward.intermediate_dense.weight.data.clone()
                streaming_layer.feed_forward[0].bias.data = layer.feed_forward.intermediate_dense.bias.data.clone()
                streaming_layer.feed_forward[3].weight.data = layer.feed_forward.output_dense.weight.data.clone()
                streaming_layer.feed_forward[3].bias.data = layer.feed_forward.output_dense.bias.data.clone()
                
                # Copy layer norms
                streaming_layer.layer_norm.weight.data = layer.layer_norm.weight.data.clone()
                streaming_layer.layer_norm.bias.data = layer.layer_norm.bias.data.clone()
                streaming_layer.final_layer_norm.weight.data = layer.final_layer_norm.weight.data.clone()
                streaming_layer.final_layer_norm.bias.data = layer.final_layer_norm.bias.data.clone()
                
                streaming_layers.append(streaming_layer)
            
            # Replace encoder layers
            self.model.encoder.layers = nn.ModuleList(streaming_layers)
        
        # Output projection
        self.output_dim = config.hidden_size
        
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the encoder.
        
        Args:
            input_values: Input tensor of shape (batch_size, seq_len)
            attention_mask: Optional mask of shape (batch_size, seq_len)
            output_hidden_states: Whether to return all hidden states
            
        Returns:
            Dictionary containing:
                - last_hidden_state: Output tensor of shape (batch_size, seq_len, hidden_size)
                - hidden_states: Optional tuple of hidden states
        """
        outputs = self.model(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        
        return {
            "last_hidden_state": outputs.last_hidden_state,
            "hidden_states": outputs.hidden_states if output_hidden_states else None,
        } 