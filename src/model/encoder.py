import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from transformers import Wav2Vec2Model, Wav2Vec2Config
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2EncoderLayer,
    Wav2Vec2Attention
)
from model.attention import AttentionSink


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
        
        # Create attention sink if using that mask type
        if attention_mask_type == "attention_sink":
            self.attention_sink = AttentionSink(
                sink_size=attention_sink_size,
                left_context=left_context,
                right_context=right_context,
            )
    
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
        # Use the AttentionSink class if available
        if hasattr(self, 'attention_sink'):
            return self.attention_sink.create_attention_mask(seq_len, device)
        
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
            
            # Right context (if specified)
            if self.right_context > 0:
                right_end = min(seq_len, i + self.right_context + 1)
                if right_end > i + 1:
                    mask[i, i+1:right_end] = 0.0
        
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
            # For the specific case of [batch_size, 1, seq_len, seq_len]
            if attention_mask.dim() == 4 and attention_mask.shape[1] == 1:
                # If the mask is already in the correct format, just use it directly
                attention_mask_float = (1.0 - attention_mask) * -10000.0
                
                # Reshape attention weights to 4D [batch_size, num_heads, seq_len, seq_len]
                attn_weights_4d = attn_weights.view(bsz, self.num_heads, seq_len, seq_len)
                
                # Expand the mask to match the number of heads
                attention_mask_expanded = attention_mask_float.expand(-1, self.num_heads, -1, -1)
                
                # Add the mask to the attention weights
                attn_weights_4d = attn_weights_4d + attention_mask_expanded
                
                # Reshape back to 3D [batch_size * num_heads, seq_len, seq_len]
                attn_weights = attn_weights_4d.view(bsz * self.num_heads, seq_len, seq_len)
            else:
                # Handle other shapes as before
                # Ensure we have a 2D attention mask [batch_size, seq_len]
                if attention_mask.dim() > 2:
                    # Flatten all extra dimensions
                    while attention_mask.dim() > 2:
                        attention_mask = attention_mask.squeeze(1)
                
                # Create a 4D attention mask [batch_size, 1, 1, seq_len]
                expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                
                # Create attention mask where 0s -> -10000.0 and 1s -> 0.0
                attention_mask_float = (1.0 - expanded_mask) * -10000.0
                
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
        # For multi-chunk training
        chunk_size_min: int = 0,
        chunk_size_max: int = 0,
        randomize_chunks: bool = False,
    ):
        super().__init__()
        
        # Load pretrained model with modified config
        from transformers import Wav2Vec2Config
        config = Wav2Vec2Config.from_pretrained(pretrained_model_name)
        
        # Disable masking for fine-tuning
        config.mask_time_prob = 0.0
        config.mask_time_length = 1
        config.mask_feature_prob = 0.0
        config.mask_feature_length = 1
        
        # Set dropout and layerdrop
        config.hidden_dropout = dropout
        config.attention_dropout = dropout
        config.activation_dropout = dropout
        config.layerdrop = layerdrop
        
        # Load model with modified config
        self.model = Wav2Vec2Model.from_pretrained(pretrained_model_name, config=config)
        
        # Explicitly set model to training mode (important for feature extractor)
        self.model.train()
        
        # Freeze feature encoder if specified
        if freeze_feature_encoder:
            for param in self.model.feature_extractor.parameters():
                param.requires_grad = False
            # Even when freezing, need to ensure training mode for batch norm stats
            self.model.feature_extractor.eval()
        
        # Freeze base model if specified
        if freeze_base_model:
            for param in self.model.parameters():
                param.requires_grad = False
            # Still enable dropout even when freezing
            self.model.train()
        
        # Store attention masking parameters
        self.attention_mask_type = attention_mask_type
        self.chunk_size = chunk_size
        self.left_context = left_context
        self.right_context = right_context
        self.attention_sink_size = attention_sink_size
        
        # Store multi-chunk training parameters
        self.chunk_size_min = chunk_size_min
        self.chunk_size_max = chunk_size_max
        self.randomize_chunks = randomize_chunks
        
        # Replace encoder layers with streaming encoder layers if using streaming
        if attention_mask_type != "full":
            streaming_layers = []
            for i, layer in enumerate(self.model.encoder.layers):
                streaming_layer = StreamingWav2Vec2EncoderLayer(
                    config=config,
                    attention_mask_type=attention_mask_type,
                    chunk_size=self._get_current_chunk_size(),
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
        
        # Add feature extractor output normalization
        self.feature_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
        
        # Add output layer normalization for stability
        self.output_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Output projection
        self.output_dim = config.hidden_size
    
    def _get_current_chunk_size(self) -> int:
        """
        Get the current chunk size, which may be randomized for multi-chunk training.
        
        Returns:
            Current chunk size to use
        """
        if self.randomize_chunks and self.chunk_size_min > 0 and self.chunk_size_max > 0:
            # For multi-chunk training, randomly select a chunk size
            import random
            return random.randint(self.chunk_size_min, self.chunk_size_max)
        else:
            # Use the fixed chunk size
            return self.chunk_size
    
    def update_streaming_config(self) -> None:
        """
        Update streaming configuration in all layers.
        Called at the start of each training batch to potentially change chunk sizes.
        """
        if self.attention_mask_type != "full":
            # Get new chunk size if using randomized chunks
            current_chunk_size = self._get_current_chunk_size()
            
            # Update each layer's attention mask
            for layer in self.model.encoder.layers:
                if hasattr(layer.attention, "chunk_size"):
                    layer.attention.chunk_size = current_chunk_size
    
    def _preprocess_input(self, input_values: torch.Tensor) -> torch.Tensor:
        """Preprocess input values for stability."""
        # Ensure input is in the range [-1, 1]
        if torch.max(torch.abs(input_values)) > 1.0:
            input_values = torch.clamp(input_values, min=-1.0, max=1.0)
        
        # Check for zero or constant input
        if torch.std(input_values) < 1e-6:
            print("WARNING: Input has very low variance (possibly silent or DC)")
            # Add small noise to prevent NaN
            input_values = input_values + torch.randn_like(input_values) * 1e-6
        
        return input_values
        
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
        # Update streaming configuration for this batch
        if self.training and self.randomize_chunks:
            self.update_streaming_config()
            
        # Check for NaN values in input
        if torch.isnan(input_values).any():
            print("WARNING: NaN values detected in encoder input!")
            # Replace NaN values with zeros
            input_values = torch.nan_to_num(input_values, nan=0.0)
        
        # Preprocess input
        input_values = self._preprocess_input(input_values)
        
        # Custom forward pass to add stability
        # First, run the feature extractor separately
        with torch.no_grad():
            # Extract features
            extract_features = self.model.feature_extractor(input_values)
            extract_features = extract_features.transpose(1, 2)
            
            # Apply layer norm to stabilize feature extractor output
            extract_features = self.feature_norm(extract_features)
            
            # Clip extreme values
            extract_features = torch.clamp(extract_features, min=-10.0, max=10.0)
            
            # Check for NaN in feature extractor output
            if torch.isnan(extract_features).any():
                print("WARNING: NaN values detected after feature extraction!")
                extract_features = torch.nan_to_num(extract_features, nan=0.0)
        
        # Then run the rest of the model
        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature extractor's output
            attention_mask = self.model._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )
            
            # For scaled_dot_product_attention used in newer PyTorch versions,
            # we need to properly format the attention mask for each layer
            # The mask needs to be [batch_size, 1, seq_len, seq_len] 
            # where 1s allow attention and 0s don't
            batch_size, seq_len = attention_mask.shape
            
            # Create a square mask from the sequential mask
            # First, create a square matrix where each row allows attention to previous positions
            causal_mask = torch.tril(torch.ones((seq_len, seq_len), 
                                               device=attention_mask.device))
            
            # Apply the batch's actual mask values to the causal mask
            # Expand attention_mask to [batch_size, 1, 1, seq_len]
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            
            # Combine with the causal mask (outer product)
            # This creates a mask of shape [batch_size, 1, seq_len, seq_len]
            extended_attention_mask = extended_attention_mask * causal_mask.unsqueeze(0)
            
            # Convert to the format expected by PyTorch (1.0 for positions to attend to)
            attention_mask = extended_attention_mask
        
        # Apply feature projection
        feature_projection_output = self.model.feature_projection(extract_features)
        
        # Handle the feature projection output which might be a tuple
        if isinstance(feature_projection_output, tuple):
            hidden_states = feature_projection_output[0]
        else:
            hidden_states = feature_projection_output
        
        # MODIFIED: Instead of calling the encoder directly, we'll process each layer manually
        # to avoid the issue with hidden_states being a tuple
        
        # Apply position embeddings if available
        if hasattr(self.model.encoder, "pos_conv_embed"):
            hidden_states = self.model.encoder.pos_conv_embed(hidden_states)
            
        # Apply layer norm if available
        if hasattr(self.model.encoder, "layer_norm"):
            hidden_states = self.model.encoder.layer_norm(hidden_states)
            
        # Apply dropout if available
        if hasattr(self.model.encoder, "dropout"):
            hidden_states = self.model.encoder.dropout(hidden_states)
        
        # Process through each encoder layer
        encoder_states = () if hasattr(self.model.encoder, "layers") else None
        
        # Process through encoder layers
        for layer in self.model.encoder.layers:
            # Check if this is one of our streaming layers or an original HF layer
            if isinstance(layer, StreamingWav2Vec2EncoderLayer):
                # Handle our custom streaming layer
                if self.training and layer.dropout.p > 0:
                    # Apply layerdrop if training
                    dropout_probability = torch.rand(())
                    # Get layerdrop probability - handle different encoder implementations
                    layerdrop_prob = 0.0
                    if hasattr(self.model.encoder, "layerdrop"):
                        layerdrop_prob = self.model.encoder.layerdrop
                    else:
                        # Fallback to the config value we set earlier
                        layerdrop_prob = self.model.config.layerdrop if hasattr(self.model.config, "layerdrop") else 0.0
                    
                    if dropout_probability > layerdrop_prob:
                        hidden_states, _ = layer(hidden_states, attention_mask=attention_mask)
                else:
                    hidden_states, _ = layer(hidden_states, attention_mask=attention_mask)
            else:
                # Handle original HF layer
                # Note: transformers layers may have different interfaces
                try:
                    outputs = layer(hidden_states, attention_mask=attention_mask)
                    
                    # Handle various return types
                    if isinstance(outputs, tuple):
                        hidden_states = outputs[0]
                    else:
                        hidden_states = outputs
                except Exception as e:
                    print(f"Error in layer forward pass: {e}")
                    print(f"Layer type: {type(layer)}")
                    print(f"Attention mask shape: {attention_mask.shape if attention_mask is not None else None}")
                    # Try fallback with different attention mask shape if needed
                    try:
                        # Reshape mask if necessary
                        if attention_mask is not None and attention_mask.dim() == 4:
                            # Make a simple binary mask for older HF layers
                            simple_mask = (attention_mask.squeeze(1).sum(-1) > 0).float()
                            outputs = layer(hidden_states, attention_mask=simple_mask)
                        else:
                            outputs = layer(hidden_states)
                            
                        # Handle various return types
                        if isinstance(outputs, tuple):
                            hidden_states = outputs[0]
                        else:
                            hidden_states = outputs
                    except Exception as inner_e:
                        print(f"Error in fallback attempt: {inner_e}")
                        # Last resort: try without mask
                        outputs = layer(hidden_states)
                        
                        # Handle various return types
                        if isinstance(outputs, tuple):
                            hidden_states = outputs[0]
                        else:
                            hidden_states = outputs
                
            if encoder_states is not None:
                encoder_states = encoder_states + (hidden_states,)
        
        # Handle final layer norm if available
        if hasattr(self.model.encoder, "layer_norm"):
            hidden_states = self.model.encoder.layer_norm(hidden_states)
            
        # Check for NaN values in hidden states
        if torch.isnan(hidden_states).any():
            print("WARNING: NaN values detected in encoder hidden states!")
            hidden_states = torch.nan_to_num(hidden_states, nan=0.0)
            
        # Apply layer normalization for stability
        last_hidden_state = self.output_layer_norm(hidden_states)
        
        # Clip extreme values to prevent NaN propagation
        last_hidden_state = torch.clamp(last_hidden_state, min=-50.0, max=50.0)
        
        # Check for NaN values in output
        if torch.isnan(last_hidden_state).any():
            print("WARNING: NaN values detected in encoder output!")
            # Replace NaN values with zeros
            last_hidden_state = torch.nan_to_num(last_hidden_state, nan=0.0)
        
        return {
            "last_hidden_state": last_hidden_state,
            "hidden_states": encoder_states if output_hidden_states else None,
        } 