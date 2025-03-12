# XLSR-Transducer: Streaming ASR for Estonian

This repository implements a streaming Automatic Speech Recognition (ASR) system based on the paper [XLSR-Transducer: Streaming ASR for Self-Supervised Pretrained Models](https://arxiv.org/abs/2407.04439).

## Overview

The implementation uses Facebook's XLSR-53 model as the encoder part of a Transducer architecture, enabling streaming ASR capabilities. Key features include:

- Streaming ASR with low latency
- Transducer-based architecture for efficient inference
- Support for Estonian language ASR
- Attention masking patterns for streaming capability
- Attention sinks to reduce left context requirements

## Project Structure

```
.
├── README.md
├── requirements.txt
├── config/
│   └── config.yaml
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   └── processor.py
│   ├── model/
│   │   ├── encoder.py
│   │   ├── predictor.py
│   │   ├── joint.py
│   │   └── transducer.py
│   ├── training/
│   │   ├── trainer.py
│   │   └── loss.py
│   └── utils/
│       ├── metrics.py
│       └── audio.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── transcribe.py
└── notebooks/
    └── demo.ipynb
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

The model expects data in the following format:
```
path_to_audio_file|transcription|speaker_id
```

### Training

```bash
python scripts/train.py
```

### Evaluation

```bash
python scripts/evaluate.py --checkpoint path/to/checkpoint
```

### Transcription

```bash
python scripts/transcribe.py --audio path/to/audio.wav --checkpoint path/to/checkpoint
```

## Citation

```
@article{xlsrtransducer2023,
  title={XLSR-Transducer: Streaming ASR for Self-Supervised Pretrained Models},
  author={...},
  journal={arXiv preprint arXiv:2407.04439},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 