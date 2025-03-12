# Multi-Stage Training for XLSR-Transducer

This document describes the multi-stage training approach for the XLSR-Transducer model for Estonian ASR. The approach is based on the paper "Efficient Streaming Language Models with Attention Sinks" and adapted for our XLSR-Transducer architecture.

## Training Stages Overview

The training process is divided into multiple stages, each building upon the previous one:

1. **Stage 1**: Non-streaming XLSR-Transducer with full attention
2. **Stage 2**: Chunked masking with varying chunk sizes
   - 2a: 320ms chunks
   - 2b: 640ms chunks
   - 2c: 1280ms chunks
   - 2d: 2560ms chunks
3. **Stage 3**: Variable left context sizes
   - 3a: 1x left context
   - 3b: 2x left context
   - 3c: Full left context
4. **Stage 4**: Multi-chunk training with randomized chunk sizes
5. **Stage 5**: Attention sink implementation
   - 5a: 4-frame attention sink
   - 5b: 16-frame attention sink

## Configuration

The training stages are defined in `config/stages.yaml`. Each stage specifies:
- Encoder parameters (attention mask type, chunk size, etc.)
- Training parameters (learning rate, batch size, etc.)
- Checkpoint directory
- Which previous stage to load from

## Running Training

To run the multi-stage training, use the `scripts/train_by_stages.py` script:

```bash
# Run all stages
./scripts/train_by_stages.py

# Run specific stages
./scripts/train_by_stages.py --start_stage 1 --end_stage 3

# Run with specific GPU
./scripts/train_by_stages.py --gpu 0

# Resume from a specific checkpoint
./scripts/train_by_stages.py --resume checkpoints/stage1/best_model.pt

# Debug mode (limited data)
./scripts/train_by_stages.py --debug
```

## Attention Mechanisms

### Full Attention
- Each token can attend to all previous tokens
- Highest quality but not efficient for streaming

### Chunked Attention
- Divides the sequence into chunks
- Each token can attend to tokens within its chunk and a limited left context
- More efficient for streaming but may lose some quality

### Attention Sink
- Each token can attend to:
  1. A fixed number of "sink" tokens at the beginning of the sequence
  2. A limited left context
- Provides a good balance between quality and efficiency

To visualize the attention patterns, use:

```bash
./scripts/test_attention_sink.py --sink_size 4 --left_context 25 --seq_len 100
```

## Inference

After training, you can use the model for inference in different modes:

1. **Non-streaming mode**: Use the model trained with full attention (Stage 1)
2. **Chunked streaming mode**: Use models from Stage 2 or 3
3. **Attention sink streaming mode**: Use models from Stage 5

## Performance Considerations

- **Quality**: Full attention > Attention sink > Chunked attention
- **Latency**: Chunked attention < Attention sink < Full attention
- **Memory**: Attention sink < Chunked attention < Full attention

## References

1. "Efficient Streaming Language Models with Attention Sinks" (https://arxiv.org/abs/2309.17453)
2. "Transformer-Transducer: End-to-End Speech Recognition with Self-attention" (https://arxiv.org/abs/1910.12977)
3. "Streaming Transformer-based Acoustic Models Using Self-attention with Augmented Memory" (https://arxiv.org/abs/2005.08042) 