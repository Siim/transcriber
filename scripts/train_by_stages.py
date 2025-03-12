#!/usr/bin/env python3

import os
import sys
import argparse
import yaml
from pathlib import Path
import torch
from datetime import datetime
import shutil

# Add the src directory to the path
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.append(src_dir)

from trainer.trainer import Trainer
from model.transducer import XLSRTransducer
from utils.file_utils import read_json, read_yaml, save_yaml
from utils.metrics import compute_wer
from utils.logging import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train XLSR-Transducer model through multiple stages")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to main config file")
    parser.add_argument("--stages", type=str, default="config/stages.yaml", help="Path to stages config file")
    parser.add_argument("--start_stage", type=int, default=1, help="Stage to start training from")
    parser.add_argument("--end_stage", type=int, default=5, help="Stage to end training at")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID to use")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from (overrides stage)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (limited data)")
    return parser.parse_args()


def stage_name(stage_id):
    """Convert stage ID to stage name (e.g., 1 -> 'stage1', 2a -> 'stage2a')"""
    if isinstance(stage_id, int):
        return f"stage{stage_id}"
    else:
        return f"stage{stage_id}"


def get_stage_config(stages_config, stage_id):
    """Get configuration for a specific stage"""
    stage = stage_name(stage_id)
    if stage not in stages_config:
        raise ValueError(f"Stage {stage} not found in stages config")
    return stages_config[stage]


def update_config_with_stage(config, stage_config):
    """Update main config with stage-specific settings"""
    # Create a deep copy of the config
    updated_config = {**config}
    
    # Update encoder params
    encoder_params = updated_config.get("model", {}).get("encoder_params", {})
    encoder_params.update(stage_config.get("encoder_params", {}))
    
    if "model" not in updated_config:
        updated_config["model"] = {}
    updated_config["model"]["encoder_params"] = encoder_params
    
    # Update training params
    if "training" in stage_config:
        if "training" not in updated_config:
            updated_config["training"] = {}
        updated_config["training"].update(stage_config["training"])
    
    # Set checkpoint dir
    if "checkpoint_dir" in stage_config:
        updated_config["training"]["checkpoint_dir"] = stage_config["checkpoint_dir"]
    
    return updated_config


def run_stage(stage_id, args, main_config, stages_config):
    """Run a specific training stage"""
    logger = get_logger(f"Stage {stage_id}")
    logger.info(f"Starting training for stage {stage_id}")
    
    # Get stage config and update main config
    stage_config = get_stage_config(stages_config, stage_id)
    config = update_config_with_stage(main_config, stage_config)
    
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = config["training"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save the merged config for reference
    save_yaml(config, os.path.join(checkpoint_dir, "merged_config.yaml"))
    
    # Check if we need to load a checkpoint from a previous stage
    resume_checkpoint = None
    if "load_from" in stage_config:
        prev_stage = stage_config["load_from"]
        prev_stage_config = get_stage_config(stages_config, prev_stage)
        prev_checkpoint_dir = prev_stage_config["checkpoint_dir"]
        
        # Find the best checkpoint from the previous stage
        best_model_path = os.path.join(prev_checkpoint_dir, "best_model.pt")
        if os.path.exists(best_model_path):
            resume_checkpoint = best_model_path
            logger.info(f"Loading best model from previous stage {prev_stage}: {resume_checkpoint}")
        else:
            logger.warning(f"Could not find best model from previous stage {prev_stage}")
    
    # If a resume path was provided via CLI, it overrides everything
    if args.resume:
        resume_checkpoint = args.resume
        logger.info(f"Resuming from specified checkpoint: {resume_checkpoint}")
    
    # Initialize model and trainer
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    
    # Set debug mode if specified
    if args.debug:
        config["training"]["max_train_samples"] = 100
        config["training"]["max_eval_samples"] = 50
        config["training"]["log_every_n_steps"] = 5
        config["training"]["eval_every_n_steps"] = 20
        logger.info("Debug mode enabled: Using reduced dataset and more frequent logging")
    
    # Create trainer
    trainer = Trainer(config, device, resume_checkpoint=resume_checkpoint)
    
    # Run training
    best_model_path = trainer.train()
    
    return best_model_path


def get_stage_list(start_stage, end_stage):
    """Get list of stages to run"""
    basic_stages = [1, 2, 3, 4, 5]
    detailed_stages = [
        1, 
        "2a", "2b", "2c", "2d",
        "3a", "3b", "3c",
        4,
        "5a", "5b"
    ]
    
    # Find start and end indices in the detailed stages list
    try:
        start_idx = detailed_stages.index(start_stage)
    except ValueError:
        # If not found, find the nearest stage
        start_idx = 0
        for i, stage in enumerate(detailed_stages):
            if isinstance(stage, int) and stage > start_stage:
                break
            if str(stage).startswith(str(start_stage)):
                start_idx = i
                break
    
    try:
        end_idx = detailed_stages.index(end_stage)
    except ValueError:
        # If not found, find the nearest stage
        end_idx = len(detailed_stages) - 1
        for i, stage in enumerate(detailed_stages):
            if isinstance(stage, int) and stage > end_stage:
                end_idx = i - 1
                break
            if str(stage).startswith(str(end_stage)):
                end_idx = i
    
    return detailed_stages[start_idx:end_idx+1]


def main():
    args = parse_args()
    
    # Load configs
    main_config = read_yaml(args.config)
    stages_config = read_yaml(args.stages)
    
    # Get list of stages to run
    stages_to_run = get_stage_list(args.start_stage, args.end_stage)
    
    print(f"Will run training for stages: {stages_to_run}")
    
    # Run each stage in sequence
    for stage_id in stages_to_run:
        best_model_path = run_stage(stage_id, args, main_config, stages_config)
        print(f"Completed stage {stage_id}. Best model saved at: {best_model_path}")


if __name__ == "__main__":
    main() 