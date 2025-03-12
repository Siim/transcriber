# Server-Side Multi-Stage Training Guide

This guide explains how to run the multi-stage training for the XLSR-Transducer model on a server.

## Setup

1. Ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Verify the directory structure:
   ```
   transcriber/
   ├── config/
   │   ├── config.yaml        # Main configuration
   │   └── stages.yaml        # Multi-stage training configuration
   ├── src/
   │   ├── data/              # Dataset and data preprocessing modules
   │   ├── model/             # Model architecture modules
   │   ├── training/          # Training and loss modules
   │   └── utils/             # Utility functions
   ├── scripts/
   │   └── train_by_stages.py # Multi-stage training script
   └── run_training.py        # Wrapper script that handles Python paths
   ```

## Running the Training

There are two ways to run the multi-stage training:

### 1. Using the Wrapper Script (Recommended)

The wrapper script handles Python import paths correctly:

```bash
python run_training.py --start_stage 1 --end_stage 5 --gpu 0
```

### 2. Using the Direct Script with PYTHONPATH

```bash
PYTHONPATH=. python scripts/train_by_stages.py --start_stage 1 --end_stage 5 --gpu 0
```

## Command-line Arguments

- `--config`: Path to main config file (default: "config/config.yaml")
- `--stages`: Path to stages config file (default: "config/stages.yaml")
- `--start_stage`: Stage to start training from (default: 1)
- `--end_stage`: Stage to end training at (default: 5)
- `--gpu`: GPU device ID to use (default: 0)
- `--resume`: Path to checkpoint to resume from (overrides stage)
- `--debug`: Enable debug mode with limited data (for testing)

## Training Stages

The training follows the stages defined in the paper:

1. **Stage 1**: Non-streaming XLSR-Transducer with full attention
2. **Stage 2**: Chunked masking with varying chunk sizes (320ms, 640ms, 1280ms, 2560ms)
3. **Stage 3**: Variable left context sizes (1x, 2x, full)
4. **Stage 4**: Multi-chunk training with randomized chunk sizes
5. **Stage 5**: Attention sink implementation (4-frame and 16-frame sinks)

## Monitoring Training

Each stage saves checkpoints in its own directory (`checkpoints/stage{n}`). You can monitor the training progress via:

```bash
tail -f checkpoints/stage1/train.log
```

## Testing Attention Sink Patterns

To visualize the attention patterns:

```bash
python scripts/test_attention_sink.py --sink_size 4 --left_context 25 --seq_len 100
```

This will generate visualizations in the `attention_sink_plots` directory.

## Debugging

If you encounter any issues with imports or Python paths, you can manually install the project as a development package:

```bash
pip install -e .
```

This requires a `setup.py` file in the root directory.

## Server Resource Management

- Training requires approximately 16GB of GPU memory per batch
- For multi-GPU training, adjust your environment accordingly:
  ```bash
  CUDA_VISIBLE_DEVICES=0,1 python run_training.py --start_stage 1
  ```

## Resuming Training

You can resume training from a specific checkpoint:

```bash
python run_training.py --start_stage 3 --resume checkpoints/stage2b/best_model.pt
``` 