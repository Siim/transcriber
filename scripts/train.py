#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import yaml
import logging
from omegaconf import OmegaConf
from typing import Dict, List, Optional, Tuple, Union

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.processor import XLSRTransducerProcessor, CharacterTokenizer
from src.data.dataset import create_dataloader
from src.model.transducer import XLSRTransducer
from src.training.trainer import XLSRTransducerTrainer, TrainingArguments
from src.utils.metrics import compute_metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train XLSR-Transducer model")
    
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--train_manifest", type=str, help="Path to training manifest file"
    )
    parser.add_argument(
        "--valid_manifest", type=str, help="Path to validation manifest file"
    )
    parser.add_argument(
        "--audio_dir", type=str, help="Path to audio directory"
    )
    parser.add_argument(
        "--output_dir", type=str, help="Path to output directory"
    )
    parser.add_argument(
        "--log_dir", type=str, help="Path to log directory"
    )
    parser.add_argument(
        "--num_epochs", type=int, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, help="Batch size"
    )
    parser.add_argument(
        "--learning_rate", type=float, help="Learning rate"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    parser.add_argument(
        "--device", type=str, help="Device to use (cuda or cpu)"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load config
    config = OmegaConf.load(args.config)
    
    # Override config with command line arguments
    if args.train_manifest:
        config.data.train_manifest = args.train_manifest
    if args.valid_manifest:
        config.data.valid_manifest = args.valid_manifest
    if args.audio_dir:
        config.data.audio_dir = args.audio_dir
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.log_dir:
        config.training.log_dir = args.log_dir
    if args.num_epochs:
        config.training.num_epochs = args.num_epochs
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.device:
        config.training.device = args.device
    
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directories
    os.makedirs(config.training.output_dir, exist_ok=True)
    os.makedirs(config.training.log_dir, exist_ok=True)
    
    # Save config
    config_path = os.path.join(config.training.output_dir, "config.yaml")
    with open(config_path, "w") as f:
        OmegaConf.save(config, f)
    
    # Create tokenizer
    tokenizer = CharacterTokenizer(
        special_tokens=config.tokenizer.special_tokens
    )
    
    # Create processor
    processor = XLSRTransducerProcessor(
        tokenizer=tokenizer,
        sample_rate=config.data.sample_rate,
        max_duration=config.data.max_duration,
    )
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_dataloader = create_dataloader(
        manifest_path=config.data.train_manifest,
        processor=processor,
        batch_size=config.data.batch_size,
        max_duration=config.data.max_duration,
        min_duration=config.data.min_duration,
        audio_dir=config.data.audio_dir,
        num_workers=config.data.num_workers,
        shuffle=True,
    )
    
    valid_dataloader = create_dataloader(
        manifest_path=config.data.valid_manifest,
        processor=processor,
        batch_size=config.data.batch_size,
        max_duration=config.data.max_duration,
        min_duration=config.data.min_duration,
        audio_dir=config.data.audio_dir,
        num_workers=config.data.num_workers,
        shuffle=False,
    )
    
    # Create model
    logger.info("Creating model...")
    model = XLSRTransducer(
        vocab_size=processor.vocab_size,
        blank_id=processor.blank_id,
        encoder_params=config.model.encoder,
        predictor_params=config.model.predictor,
        joint_params=config.model.joint,
    )
    
    # Create training arguments
    training_args = TrainingArguments(
        output_dir=config.training.output_dir,
        log_dir=config.training.log_dir,
        num_epochs=config.training.num_epochs,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_steps=config.training.warmup_steps,
        grad_clip=config.training.grad_clip,
        log_interval=config.training.log_interval,
        eval_interval=config.training.eval_interval,
        save_interval=config.training.save_interval,
        early_stopping_patience=config.training.early_stopping_patience,
        scheduler=config.training.scheduler,
        use_fp16=config.training.use_fp16,
        device=config.training.device,
    )
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = XLSRTransducerTrainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=valid_dataloader,
        args=training_args,
        compute_metrics=lambda preds, labels: compute_metrics(preds, labels, tokenizer),
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Training completed")


if __name__ == "__main__":
    main() 