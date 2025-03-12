#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import yaml
import logging
import json
from omegaconf import OmegaConf
from typing import Dict, List, Optional, Tuple, Union

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.processor import XLSRTransducerProcessor, CharacterTokenizer
from src.data.dataset import create_dataloader
from src.model.transducer import XLSRTransducer
from src.utils.metrics import compute_metrics, compute_wer, compute_cer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate XLSR-Transducer model")
    
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--test_manifest", type=str, help="Path to test manifest file"
    )
    parser.add_argument(
        "--audio_dir", type=str, help="Path to audio directory"
    )
    parser.add_argument(
        "--output_dir", type=str, help="Path to output directory"
    )
    parser.add_argument(
        "--batch_size", type=int, help="Batch size"
    )
    parser.add_argument(
        "--beam_size", type=int, help="Beam size for decoding"
    )
    parser.add_argument(
        "--device", type=str, help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--save_results", action="store_true", help="Save evaluation results to file"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load config
    config = OmegaConf.load(args.config)
    
    # Override config with command line arguments
    if args.test_manifest:
        config.data.test_manifest = args.test_manifest
    if args.audio_dir:
        config.data.audio_dir = args.audio_dir
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.beam_size:
        config.inference.beam_size = args.beam_size
    if args.device:
        config.training.device = args.device
    
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    
    # Create output directory
    os.makedirs(config.training.output_dir, exist_ok=True)
    
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
    
    # Create dataloader
    logger.info("Creating dataloader...")
    test_dataloader = create_dataloader(
        manifest_path=config.data.test_manifest,
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
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=config.training.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Set device
    device = torch.device(config.training.device)
    model = model.to(device)
    
    # Evaluate model
    logger.info("Evaluating model...")
    model.eval()
    
    all_preds = []
    all_labels = []
    all_texts = []
    all_pred_texts = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(
                input_values=batch["input_values"],
                attention_mask=batch["attention_mask"],
            )
            
            # Decode predictions
            encoder_outputs = outputs["encoder_outputs"]
            
            if config.inference.beam_size > 1:
                predictions = model.decode_beam(
                    encoder_outputs=encoder_outputs,
                    beam_size=config.inference.beam_size,
                    max_length=config.inference.max_length,
                )
            else:
                predictions = model.decode_greedy(
                    encoder_outputs=encoder_outputs,
                    max_length=config.inference.max_length,
                )
            
            # Collect predictions and labels
            all_preds.extend(predictions)
            
            for i, length in enumerate(batch["label_lengths"]):
                label = batch["labels"][i, :length].cpu().tolist()
                all_labels.append(label)
                
                # Decode to text
                text = tokenizer.decode(label)
                pred_text = tokenizer.decode(predictions[i])
                
                all_texts.append(text)
                all_pred_texts.append(pred_text)
    
    # Compute metrics
    logger.info("Computing metrics...")
    wer = compute_wer(all_preds, all_labels, tokenizer)
    cer = compute_cer(all_preds, all_labels, tokenizer)
    
    logger.info(f"WER: {wer:.4f}")
    logger.info(f"CER: {cer:.4f}")
    
    # Save results
    if args.save_results:
        results_path = os.path.join(config.training.output_dir, "evaluation_results.json")
        
        results = {
            "wer": wer,
            "cer": cer,
            "samples": [
                {
                    "reference": text,
                    "prediction": pred_text,
                }
                for text, pred_text in zip(all_texts, all_pred_texts)
            ]
        }
        
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main() 