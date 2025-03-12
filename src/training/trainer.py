import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union, Callable
import logging
import time
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass, field

from ..model.transducer import XLSRTransducer
from .loss import TransducerLossWrapper


@dataclass
class TrainingArguments:
    """Arguments for training the XLSR-Transducer model."""
    
    output_dir: str = "checkpoints"
    log_dir: str = "logs"
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    grad_clip: float = 5.0
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    early_stopping_patience: int = 5
    scheduler: str = "linear"  # Options: "linear", "cosine", "constant"
    use_fp16: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class XLSRTransducerTrainer:
    """Trainer for the XLSR-Transducer model."""
    
    def __init__(
        self,
        model: XLSRTransducer,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        args: Optional[TrainingArguments] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        compute_metrics: Optional[Callable] = None,
    ):
        """
        Initialize the trainer.
        
        Args:
            model: XLSR-Transducer model
            train_dataloader: DataLoader for training data
            eval_dataloader: Optional DataLoader for evaluation data
            args: Training arguments
            optimizer: Optional optimizer
            scheduler: Optional learning rate scheduler
            compute_metrics: Optional function to compute metrics
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.args = args or TrainingArguments()
        self.compute_metrics = compute_metrics
        
        # Set up device
        self.device = torch.device(self.args.device)
        self.model = self.model.to(self.device)
        
        # Set up loss function
        self.loss_fn = TransducerLossWrapper(blank_id=model.blank_id)
        
        # Set up optimizer
        if optimizer is None:
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )
        else:
            self.optimizer = optimizer
        
        # Set up scheduler
        if scheduler is None:
            total_steps = len(train_dataloader) * self.args.num_epochs
            
            if self.args.scheduler == "linear":
                self.scheduler = optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=1.0,
                    end_factor=0.1,
                    total_iters=total_steps,
                )
            elif self.args.scheduler == "cosine":
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=total_steps,
                )
            else:  # constant
                self.scheduler = optim.lr_scheduler.ConstantLR(
                    self.optimizer,
                    factor=1.0,
                    total_iters=total_steps,
                )
        else:
            self.scheduler = scheduler
        
        # Set up scaler for mixed precision training
        self.scaler = torch.amp.GradScaler('cuda') if self.args.use_fp16 else None
        
        # Set up logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        os.makedirs(self.args.output_dir, exist_ok=True)
        os.makedirs(self.args.log_dir, exist_ok=True)
        
        # Initialize training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float("inf")
        self.no_improvement_count = 0
    
    def train(self) -> Dict[str, float]:
        """
        Train the model.
        
        Returns:
            Dictionary of training metrics
        """
        self.logger.info("Starting training...")
        
        # Training loop
        for epoch in range(self.args.num_epochs):
            self.epoch = epoch
            self.logger.info(f"Epoch {epoch + 1}/{self.args.num_epochs}")
            
            # Train for one epoch
            train_metrics = self._train_epoch()
            
            # Evaluate
            if self.eval_dataloader is not None:
                eval_metrics = self.evaluate()
                
                # Early stopping
                if eval_metrics["eval_loss"] < self.best_eval_loss:
                    self.best_eval_loss = eval_metrics["eval_loss"]
                    self.no_improvement_count = 0
                    
                    # Save best model
                    self._save_checkpoint("best")
                else:
                    self.no_improvement_count += 1
                    
                    if self.no_improvement_count >= self.args.early_stopping_patience:
                        self.logger.info(f"Early stopping after {epoch + 1} epochs")
                        break
            
            # Save checkpoint
            self._save_checkpoint(f"epoch_{epoch + 1}")
        
        self.logger.info("Training completed")
        
        return {
            "train_loss": train_metrics["train_loss"],
            "best_eval_loss": self.best_eval_loss,
        }
    
    def _train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        # Debugging: Check if model parameters require gradients
        param_requires_grad = {}
        for name, param in self.model.named_parameters():
            param_requires_grad[name] = param.requires_grad
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        non_trainable_params = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {non_trainable_params:,}")
        
        # Print a few key parameters to check
        if 'encoder.model.feature_extractor.conv_layers.0.conv.weight' in param_requires_grad:
            print(f"Encoder feature extractor requires_grad: {param_requires_grad['encoder.model.feature_extractor.conv_layers.0.conv.weight']}")
        if 'predictor.embedding.weight' in param_requires_grad:
            print(f"Predictor embedding requires_grad: {param_requires_grad['predictor.embedding.weight']}")
        if 'joint.output_proj.weight' in param_requires_grad:
            print(f"Joint output_proj requires_grad: {param_requires_grad['joint.output_proj.weight']}")
        
        total_loss = 0.0
        num_batches = len(self.train_dataloader)
        start_time = time.time()
        
        # Progress bar
        progress_bar = tqdm(
            total=num_batches,
            desc=f"Epoch {self.epoch + 1}",
            leave=False,
        )
        
        # Training loop
        for batch_idx, batch in enumerate(self.train_dataloader):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            if self.args.use_fp16:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(
                        input_values=batch["input_values"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        label_lengths=batch["label_lengths"],
                    )
                    
                    loss = self.loss_fn(
                        outputs=outputs,
                        labels=batch["labels"],
                        label_lengths=batch["label_lengths"],
                        attention_mask=batch["attention_mask"],
                    )
                    
                    # Check if loss requires gradient
                    if batch_idx == 0:
                        print(f"Loss requires_grad: {loss.requires_grad}")
                        if 'logits' in outputs:
                            print(f"Logits requires_grad: {outputs['logits'].requires_grad}")
                
                # Backward pass with gradient scaling
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.args.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.grad_clip
                    )
                
                # Update weights
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Forward pass
                outputs = self.model(
                    input_values=batch["input_values"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    label_lengths=batch["label_lengths"],
                )
                
                loss = self.loss_fn(
                    outputs=outputs,
                    labels=batch["labels"],
                    label_lengths=batch["label_lengths"],
                    attention_mask=batch["attention_mask"],
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if self.args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.grad_clip
                    )
                
                # Update weights
                self.optimizer.step()
            
            # Update scheduler
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            self.global_step += 1
            
            # Log metrics
            if self.global_step % self.args.log_interval == 0:
                lr = self.scheduler.get_last_lr()[0]
                self.logger.info(
                    f"Step {self.global_step}: loss = {loss.item():.4f}, lr = {lr:.6f}"
                )
            
            # Evaluate
            if self.eval_dataloader is not None and self.global_step % self.args.eval_interval == 0:
                eval_metrics = self.evaluate()
                self.model.train()
            
            # Save checkpoint
            if self.global_step % self.args.save_interval == 0:
                self._save_checkpoint(f"step_{self.global_step}")
            
            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item()})
        
        # Close progress bar
        progress_bar.close()
        
        # Compute epoch metrics
        avg_loss = total_loss / num_batches
        elapsed_time = time.time() - start_time
        
        self.logger.info(
            f"Epoch {self.epoch + 1}: loss = {avg_loss:.4f}, "
            f"time = {elapsed_time:.2f}s"
        )
        
        return {
            "train_loss": avg_loss,
            "epoch": self.epoch + 1,
            "global_step": self.global_step,
        }
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Returns:
            Dictionary of evaluation metrics
        """
        if self.eval_dataloader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        num_batches = len(self.eval_dataloader)
        all_preds = []
        all_labels = []
        
        # Progress bar
        progress_bar = tqdm(
            total=num_batches,
            desc="Evaluating",
            leave=False,
        )
        
        # Evaluation loop
        with torch.no_grad():
            for batch in self.eval_dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_values=batch["input_values"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    label_lengths=batch["label_lengths"],
                )
                
                # Compute loss
                loss = self.loss_fn(
                    outputs=outputs,
                    labels=batch["labels"],
                    label_lengths=batch["label_lengths"],
                    attention_mask=batch["attention_mask"],
                )
                
                # Update metrics
                total_loss += loss.item()
                
                # Decode predictions
                encoder_outputs = outputs["encoder_outputs"]
                predictions = self.model.decode_greedy(encoder_outputs)
                
                # Collect predictions and labels for metrics
                all_preds.extend(predictions)
                
                for i, length in enumerate(batch["label_lengths"]):
                    all_labels.append(batch["labels"][i, :length].cpu().tolist())
                
                # Update progress bar
                progress_bar.update(1)
        
        # Close progress bar
        progress_bar.close()
        
        # Compute metrics
        avg_loss = total_loss / num_batches
        
        self.logger.info(f"Evaluation: loss = {avg_loss:.4f}")
        
        # Compute additional metrics if provided
        metrics = {"eval_loss": avg_loss}
        
        if self.compute_metrics is not None:
            additional_metrics = self.compute_metrics(all_preds, all_labels)
            metrics.update(additional_metrics)
            
            # Log additional metrics
            for name, value in additional_metrics.items():
                self.logger.info(f"Evaluation: {name} = {value:.4f}")
        
        return metrics
    
    def _save_checkpoint(self, name: str) -> None:
        """
        Save a checkpoint of the model.
        
        Args:
            name: Name of the checkpoint
        """
        checkpoint_path = os.path.join(self.args.output_dir, f"{name}.pt")
        
        # Save model, optimizer, scheduler, and training state
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "epoch": self.epoch,
                "global_step": self.global_step,
                "best_eval_loss": self.best_eval_loss,
            },
            checkpoint_path,
        )
        
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load a checkpoint of the model.
        
        Args:
            checkpoint_path: Path to the checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model, optimizer, scheduler, and training state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_eval_loss = checkpoint["best_eval_loss"]
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")


def compute_wer(predictions: List[List[int]], labels: List[List[int]]) -> float:
    """
    Compute Word Error Rate (WER).
    
    Args:
        predictions: List of predicted token IDs
        labels: List of ground truth token IDs
        
    Returns:
        Word Error Rate
    """
    total_errors = 0
    total_words = 0
    
    for pred, label in zip(predictions, labels):
        # Compute Levenshtein distance
        distance = levenshtein_distance(pred, label)
        
        # Update metrics
        total_errors += distance
        total_words += len(label)
    
    # Compute WER
    wer = total_errors / max(1, total_words)
    
    return wer


def levenshtein_distance(seq1: List[int], seq2: List[int]) -> int:
    """
    Compute Levenshtein distance between two sequences.
    
    Args:
        seq1: First sequence
        seq2: Second sequence
        
    Returns:
        Levenshtein distance
    """
    # Initialize distance matrix
    m, n = len(seq1), len(seq2)
    distance = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    
    # Initialize first row and column
    for i in range(m + 1):
        distance[i][0] = i
    for j in range(n + 1):
        distance[0][j] = j
    
    # Fill distance matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                distance[i][j] = distance[i - 1][j - 1]
            else:
                distance[i][j] = min(
                    distance[i - 1][j] + 1,  # Deletion
                    distance[i][j - 1] + 1,  # Insertion
                    distance[i - 1][j - 1] + 1,  # Substitution
                )
    
    return distance[m][n] 