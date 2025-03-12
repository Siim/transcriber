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
import contextlib

from model.transducer import XLSRTransducer
from training.loss import TransducerLoss
from data.dataset import EstonianASRDataset, collate_fn, create_length_sorted_dataloader
from data.processor import XLSRTransducerProcessor


# Define robust collate function outside the Trainer class to make it picklable
def robust_collate_fn(
    batch: List[Dict[str, torch.Tensor]], 
    processor: Optional[XLSRTransducerProcessor] = None,
    logger: Optional[logging.Logger] = None,
    max_input_length: Optional[int] = None,
    debug: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Wrapper around collate_fn that handles edge cases and adds logging.
    
    Args:
        batch: List of samples from the dataset
        processor: Processor for audio and text
        logger: Logger for warnings
        max_input_length: Maximum input length to limit sequence length
        debug: Whether to print debug information
        
    Returns:
        Dictionary with batched tensors
    """
    # Filter out samples with missing keys or empty tensors
    valid_batch = []
    for sample in batch:
        if all(k in sample for k in ["input_values", "labels", "speaker_id"]):
            if sample["input_values"].numel() > 0 and sample["labels"].numel() > 0:
                # Add processor to each sample for vocabulary size check
                if processor is not None:
                    sample["processor"] = processor
                valid_batch.append(sample)
            elif debug and logger:
                logger.debug(f"Skipping sample with empty tensor: input={sample['input_values'].shape}, labels={sample['labels'].shape}")
    
    # If we have no valid samples, create a dummy batch with minimal inputs
    if len(valid_batch) == 0:
        if logger:
            logger.warning("No valid samples in batch, creating dummy batch")
        
        blank_id = 0
        if processor is not None:
            blank_id = processor.blank_id
            
        dummy_sample = {
            "input_values": torch.zeros(1, 16000),  # 1 second at 16kHz
            "labels": torch.tensor([blank_id], dtype=torch.long),
            "speaker_id": torch.tensor([0], dtype=torch.long),
            "processor": processor
        }
        valid_batch = [dummy_sample, dummy_sample]  # Need at least 2 for batch
    
    # Use the optimized collate function with the valid batch
    return collate_fn(
        valid_batch, 
        max_input_length=max_input_length,
        debug=debug
    )


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
    gradient_accumulation_steps: int = 1  # Number of steps to accumulate gradients


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
        # Ensure torch is available in this scope
        import torch
        import torch.optim as optim
        
        # Set up logging FIRST, before using it
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )
        self.logger = logging.getLogger(__name__)
        
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.args = args or TrainingArguments()
        self.compute_metrics = compute_metrics
        
        # Set up device
        self.device = torch.device(self.args.device)
        self.model = self.model.to(self.device)
        
        # Try to use torch.compile if available (PyTorch 2.0+)
        try:
            import torch._dynamo
            compile_supported = hasattr(torch, 'compile')
            if compile_supported:
                # Check if we're on a supported device
                use_compile = (
                    self.device.type == "cuda" or  # NVIDIA GPUs 
                    self.device.type == "mps"      # Apple Silicon GPUs
                )
                
                if use_compile:
                    self.logger.info("Using torch.compile for optimized model execution")
                    # Use the most appropriate backend based on the device
                    backend = "inductor" if self.device.type == "cuda" else "eager"
                    self.model = torch.compile(self.model, backend=backend)
                    
                    # Set config to avoid recompilations
                    torch._dynamo.config.suppress_errors = True
                    torch._dynamo.config.cache_size_limit = 512
                else:
                    self.logger.info(f"Device {self.device.type} not supported for torch.compile, using standard execution")
        except (ImportError, AttributeError):
            # torch.compile not available, continue without it
            self.logger.info("torch.compile not available, using standard model execution")
        
        # Set up loss function
        self.loss_fn = TransducerLoss(blank_id=model.blank_id)
        
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
        
        # Set up scaler for mixed precision training - handle both CUDA and MPS
        if self.args.use_fp16:
            if self.device.type == "cuda":
                self.scaler = torch.amp.GradScaler()
            elif self.device.type == "mps":
                # MPS doesn't support GradScaler yet, but can use autocast
                self.scaler = None
                self.logger.info("MPS device detected: using autocast without GradScaler")
            else:
                self.scaler = None
                self.logger.info("Mixed precision requested but not supported on this device")
        else:
            self.scaler = None
        
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
        # Set the model to training mode
        self.model.train()
        
        # Get gradient accumulation steps
        gradient_accumulation_steps = self.args.gradient_accumulation_steps
        
        # Create progress bar
        num_batches = len(self.train_dataloader)
        progress_bar = tqdm(total=num_batches, desc=f"Epoch {self.epoch + 1}")
        
        # Initialize metrics
        total_loss = 0.0
        
        # Initialize performance tracking
        start_time = time.time()
        batch_times = []
        data_loading_times = []
        forward_times = []
        backward_times = []
        
        # Track GPU memory usage if available
        if torch.cuda.is_available() and self.device.type == "cuda":
            start_memory = torch.cuda.memory_allocated() / (1024 ** 3)
            memory_usage = []
        else:
            memory_usage = None
            start_memory = 0

        # Determine which device type to use for autocast (if fp16 is enabled)
        amp_device_type = "cpu"
        if self.args.use_fp16:
            if self.device.type == "cuda":
                amp_device_type = "cuda"
            elif self.device.type == "mps":
                amp_device_type = "mps"
        
        # Use Async data loading to overlap data loading with computation
        data_load_start = time.time()
        
        # Training loop
        for batch_idx, batch in enumerate(self.train_dataloader):
            # Measure data loading time
            data_loading_time = time.time() - data_load_start
            data_loading_times.append(data_loading_time)
            
            # Track batch start time for computation
            batch_start_time = time.time()
            
            # Move batch to device with non_blocking=True for parallel transfer
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            
            # Ensure data is on device before computation - safely handle different device types
            if torch.cuda.is_available() and self.device.type == "cuda":
                torch.cuda.synchronize()
            elif hasattr(torch, 'mps') and hasattr(torch.mps, 'synchronize') and self.device.type == "mps":
                torch.mps.synchronize()
            
            # Forward pass
            forward_start = time.time()
            
            # Context manager for mixed precision (if enabled)
            cm = torch.amp.autocast(device_type=amp_device_type) if self.args.use_fp16 else contextlib.nullcontext()
            
            with cm:
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
                
                # Scale loss for gradient accumulation
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
                
                # Debug logging for first batch
                if batch_idx == 0:
                    self.logger.debug(f"Loss requires_grad: {loss.requires_grad}")
                    if 'logits' in outputs:
                        self.logger.debug(f"Logits requires_grad: {outputs['logits'].requires_grad}")
            
            # Check for NaN loss
            if torch.isnan(loss).any():
                self.logger.warning(f"NaN loss detected at step {self.global_step}. Skipping batch.")
                # Skip this batch
                continue
            
            # Record forward pass time
            forward_time = time.time() - forward_start
            forward_times.append(forward_time)
            
            # Backward pass with gradient scaling if available
            backward_start = time.time()
            
            # Use scaler only on CUDA devices
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                # Regular backward on CPU or MPS
                loss.backward()
            
            # Record backward pass time
            backward_time = time.time() - backward_start
            backward_times.append(backward_time)
            
            # Only update weights after accumulating gradients
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                # Check for NaN gradients
                has_nan_grad = False
                for name, param in self.model.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        self.logger.warning(f"NaN gradient detected in {name} at step {self.global_step}")
                        has_nan_grad = True
                        break
                
                if has_nan_grad:
                    self.logger.warning(f"Skipping parameter update due to NaN gradients at step {self.global_step}")
                    # Skip parameter update for this batch
                    self.optimizer.zero_grad()
                    continue
                
                # Gradient clipping - handle with or without scaler
                if self.args.grad_clip > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.grad_clip
                    )
                
                # Update weights - handle with or without scaler
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                # Zero gradients
                self.optimizer.zero_grad()
            
            # Check for NaN parameters after update
            has_nan_param = False
            for name, param in self.model.named_parameters():
                if torch.isnan(param).any():
                    self.logger.warning(f"NaN parameter detected in {name} at step {self.global_step}")
                    has_nan_param = True
                    break
            
            if has_nan_param:
                self.logger.warning(f"NaN parameters detected after update at step {self.global_step}")
                # Load last checkpoint or reinitialize model
                # For now, we'll just continue and hope for the best
            
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
            
            # Performance tracking
            batch_compute_time = time.time() - batch_start_time
            batch_times.append(batch_compute_time)
            
            if torch.cuda.is_available() and self.device.type == "cuda":
                memory_usage.append(torch.cuda.memory_allocated() / (1024 ** 3) - start_memory)
                
            # Start timing data loading for the next batch
            data_load_start = time.time()
            
            # Use appropriate stream synchronization based on device type
            if torch.cuda.is_available() and self.device.type == "cuda":
                torch.cuda.current_stream().synchronize()
            elif hasattr(torch, 'mps') and hasattr(torch.mps, 'synchronize') and self.device.type == "mps":
                torch.mps.synchronize()
        
        # Close progress bar
        progress_bar.close()
        
        # Compute epoch metrics
        avg_loss = total_loss / num_batches
        elapsed_time = time.time() - start_time
        
        self.logger.info(
            f"Epoch {self.epoch + 1}: loss = {avg_loss:.4f}, "
            f"time = {elapsed_time:.2f}s"
        )
        
        # Log performance summary with more detailed breakdowns
        self.logger.info(f"Average batch time: {np.mean(batch_times):.4f}s")
        self.logger.info(f"Average data loading time: {np.mean(data_loading_times):.4f}s")
        self.logger.info(f"Average forward pass time: {np.mean(forward_times):.4f}s")
        self.logger.info(f"Average backward pass time: {np.mean(backward_times):.4f}s")
        self.logger.info(f"Maximum batch time: {max(batch_times):.4f}s")
        
        # Memory usage stats if available
        if memory_usage:
            self.logger.info(f"Peak GPU memory usage: {max(memory_usage):.2f}GB")
        
        return {"loss": avg_loss}
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on the evaluation dataset.
        
        Returns:
            Dictionary of evaluation metrics
        """
        # Set the model to evaluation mode
        self.model.eval()
        
        # Skip if no eval dataloader
        if self.eval_dataloader is None:
            return {}
        
        # Initialize metrics
        total_loss = 0.0
        
        # Create progress bar
        num_batches = len(self.eval_dataloader)
        progress_bar = tqdm(total=num_batches, desc="Evaluating")
        
        # Determine which device type to use for autocast (if fp16 is enabled)
        amp_device_type = "cpu"
        if self.args.use_fp16:
            if self.device.type == "cuda":
                amp_device_type = "cuda"
            elif self.device.type == "mps":
                amp_device_type = "mps"
        
        # Evaluation loop
        with torch.no_grad():
            for batch in self.eval_dataloader:
                # Move batch to device with non_blocking=True
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                
                # Ensure data is on device before computation
                if torch.cuda.is_available() and self.device.type == "cuda":
                    torch.cuda.synchronize()
                elif hasattr(torch, 'mps') and hasattr(torch.mps, 'synchronize') and self.device.type == "mps":
                    torch.mps.synchronize()
                
                # Context manager for mixed precision
                cm = torch.amp.autocast(device_type=amp_device_type) if self.args.use_fp16 else contextlib.nullcontext()
                
                with cm:
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
                
                # Check for NaN loss
                if torch.isnan(loss).any():
                    self.logger.warning(f"NaN loss detected during evaluation. Skipping batch.")
                    continue
                
                # Update metrics
                total_loss += loss.item()
                
                # Update progress bar
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": loss.item()})
        
        # Close progress bar
        progress_bar.close()
        
        # Compute metrics
        avg_loss = total_loss / num_batches
        
        # Log metrics
        self.logger.info(f"Evaluation loss: {avg_loss:.4f}")
        
        return {"loss": avg_loss}
    
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

# Add a proper adapter class
class Trainer:
    """
    Adapter class for the XLSRTransducerTrainer to match the expected interface
    from the train_by_stages.py script.
    """
    
    def __init__(self, config, device, resume_checkpoint=None, debug=False):
        """
        Initialize the trainer adapter.
        
        Args:
            config: Configuration dictionary
            device: Device to run on ("cuda" or "cpu")
            resume_checkpoint: Optional path to a checkpoint to resume from
            debug: Whether to run in debug mode
        """
        from model.transducer import XLSRTransducer
        from data.dataset import EstonianASRDataset, collate_fn
        from data.processor import XLSRTransducerProcessor
        from torch.utils.data import DataLoader
        import logging
        import torch  # Ensure torch is explicitly imported here
        
        self.config = config
        self.device = device
        self.resume_checkpoint = resume_checkpoint
        self.debug = debug
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO if not debug else logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        if debug:
            self.logger.info("Debug mode enabled - using reduced dataset and more frequent logging")
        
        # Get tokenizer configuration
        tokenizer_config = config.get("tokenizer", {})
        
        # Set vocabulary size and blank ID
        # For character tokenizer, we'll use a default size of 60 as mentioned in the config comments
        vocab_size = tokenizer_config.get("vocab_size", 60)
        
        # Get the blank token index (typically the last token in the vocabulary)
        blank_id = vocab_size - 1
        
        # Initialize joint network parameters if they don't exist
        if "joint_params" not in config.get("model", {}):
            config["model"]["joint_params"] = {}
        
        # Make sure hidden_dim is set for the joint network
        if "hidden_dim" not in config["model"]["joint_params"]:
            # Look for hidden_dim in the joint section of config
            joint_config = config.get("model", {}).get("joint", {})
            if "hidden_dim" in joint_config:
                config["model"]["joint_params"]["hidden_dim"] = joint_config["hidden_dim"]
            else:
                # Default to 640 as shown in the config file
                config["model"]["joint_params"]["hidden_dim"] = 640
        
        # Create model
        self.model = XLSRTransducer(
            vocab_size=vocab_size,
            blank_id=blank_id,
            encoder_params=config["model"].get("encoder_params", {}),
            predictor_params=config["model"].get("predictor_params", {}),
            joint_params=config["model"]["joint_params"]
        ).to(device)
        
        # Get processor configuration
        processor_config = config.get("processor", {})
        
        # Add tokenizer configuration to processor config
        if "type" in tokenizer_config:
            processor_config["tokenizer_type"] = tokenizer_config["type"]
        if "vocab_size" in tokenizer_config:
            processor_config["vocab_size"] = tokenizer_config["vocab_size"]
        if "special_tokens" in tokenizer_config:
            processor_config["special_tokens"] = tokenizer_config["special_tokens"]
        
        # Add tokenizer directory
        processor_config["tokenizer_dir"] = tokenizer_config.get("dir", "data/tokenizer")
        
        # Create processor
        self.processor = XLSRTransducerProcessor(**processor_config)
        
        # Create datasets
        data_config = config.get("data", {})
        
        # Modify dataset configuration for debug mode
        if debug:
            self.logger.info("Debug mode: Using a small subset of data for faster iteration")
            # Use the first 100 examples for training and 20 for validation in debug mode
            max_train_samples = min(100, data_config.get("max_train_samples", 100))
            max_eval_samples = min(20, data_config.get("max_eval_samples", 20))
        else:
            max_train_samples = None
            max_eval_samples = None
        
        self.train_dataset = EstonianASRDataset(
            manifest_path=data_config.get("train_manifest", ""),
            processor=self.processor,
            audio_dir=data_config.get("audio_dir", ""),
            max_duration=data_config.get("max_duration", 30.0),
            min_duration=data_config.get("min_duration", 0.5),
            debug=debug
        )
        
        # Limit dataset size in debug mode
        if max_train_samples is not None and max_train_samples < len(self.train_dataset):
            self.logger.info(f"Debug mode: Limiting training dataset to {max_train_samples} samples")
            self.train_dataset.samples = self.train_dataset.samples[:max_train_samples]
        
        self.eval_dataset = EstonianASRDataset(
            manifest_path=data_config.get("valid_manifest", ""),
            processor=self.processor,
            audio_dir=data_config.get("audio_dir", ""),
            max_duration=data_config.get("max_duration", 30.0),
            min_duration=data_config.get("min_duration", 0.5),
            debug=debug
        )
        
        # Limit dataset size in debug mode
        if max_eval_samples is not None and max_eval_samples < len(self.eval_dataset):
            self.logger.info(f"Debug mode: Limiting evaluation dataset to {max_eval_samples} samples")
            self.eval_dataset.samples = self.eval_dataset.samples[:max_eval_samples]
        
        # Validate dataset sizes
        if len(self.train_dataset) == 0:
            raise ValueError("Training dataset is empty. Check manifest path and audio files.")
        
        if len(self.eval_dataset) == 0:
            raise ValueError("Evaluation dataset is empty. Check manifest path and audio files.")
        
        self.logger.info(f"Training dataset size: {len(self.train_dataset)}")
        self.logger.info(f"Evaluation dataset size: {len(self.eval_dataset)}")
        
        # Create dataloaders
        batch_size = data_config.get("batch_size", 2)
        num_workers = data_config.get("num_workers", 4)
        
        # For debug mode, use smaller batch size and fewer workers
        if debug:
            batch_size = min(batch_size, 2)
            # Use 0 workers in debug mode to avoid pickling issues
            num_workers = 0
            self.logger.info(f"Debug mode: Using batch_size={batch_size}, num_workers={num_workers}")
        else:
            # For production training, optimize CPU utilization
            # Set number of workers to a reasonable value based on available cores
            import multiprocessing
            max_workers = max(1, multiprocessing.cpu_count() // 2)  # Use half of available cores
            num_workers = min(max_workers, num_workers)  # Don't exceed user-specified limit
            
            # Enable multiprocessing optimizations
            torch.set_num_threads(max(1, multiprocessing.cpu_count() - num_workers))  # Leave cores for workers
            
            # Configure multiprocessing method - 'fork' is faster on Linux/MacOS
            torch.multiprocessing.set_start_method('fork', force=True)
            
            self.logger.info(f"Optimized multiprocessing: num_workers={num_workers}, "
                           f"torch_threads={torch.get_num_threads()}")
        
        # Calculate maximum input length in samples (e.g., 10 seconds worth)
        max_input_length = None
        if max_duration := data_config.get("max_duration", None):
            max_input_length = int(max_duration * self.processor.audio_preprocessor.sample_rate)
            self.logger.info(f"Limiting maximum sequence length to {max_input_length} samples ({max_duration}s)")
        
        # Get dataset optimization settings
        dataset_config = data_config.get("dataset", {})
        bucket_size_multiplier = dataset_config.get("bucket_size_multiplier", 5)
        use_log_buckets = dataset_config.get("use_log_buckets", True)
        drop_last = dataset_config.get("drop_last", False)
        pin_memory = dataset_config.get("pin_memory", True)
        
        if not debug:
            self.logger.info(f"Using optimized dataset with bucketing: bucket_size_multiplier={bucket_size_multiplier}, "
                            f"use_log_buckets={use_log_buckets}, drop_last={drop_last}")
        
        # Use length-sorted dataloader for more efficient training
        train_dataloader = create_length_sorted_dataloader(
            manifest_path=data_config.get("train_manifest", ""),
            processor=self.processor,
            batch_size=batch_size,
            max_duration=data_config.get("max_duration", 30.0),
            min_duration=data_config.get("min_duration", 0.5),
            audio_dir=data_config.get("audio_dir", ""),
            num_workers=num_workers,
            shuffle=True,
            drop_last=drop_last,
            bucket_size_multiplier=bucket_size_multiplier,
            use_log_buckets=use_log_buckets,
            pin_memory=pin_memory,
            debug=debug
        )
        
        eval_dataloader = create_length_sorted_dataloader(
            manifest_path=data_config.get("valid_manifest", ""),
            processor=self.processor,
            batch_size=batch_size,
            max_duration=data_config.get("max_duration", 30.0),
            min_duration=data_config.get("min_duration", 0.5),
            audio_dir=data_config.get("audio_dir", ""),
            num_workers=num_workers,
            shuffle=False,  # No need to shuffle evaluation data
            drop_last=False,  # Don't drop last batch for evaluation
            bucket_size_multiplier=1,  # No randomness for evaluation
            use_log_buckets=False,  # No need for log buckets in evaluation
            pin_memory=pin_memory,
            debug=debug
        )
        
        # Create training arguments
        training_config = config.get("training", {})
        
        # Adjust training parameters for debug mode
        if debug:
            log_interval = 1
            eval_interval = 5
            save_interval = 10
            self.logger.info(f"Debug mode: Using modified training parameters for faster iterations")
        else:
            log_interval = training_config.get("log_interval", 100)
            eval_interval = training_config.get("eval_interval", 1000)
            save_interval = training_config.get("save_interval", 5000)
        
        args = TrainingArguments(
            output_dir=training_config.get("checkpoint_dir", "checkpoints"),
            log_dir=training_config.get("log_dir", "logs"),
            num_epochs=training_config.get("epochs", 10),
            learning_rate=training_config.get("learning_rate", 1e-4),
            weight_decay=training_config.get("weight_decay", 1e-4),
            warmup_steps=training_config.get("warmup_steps", 1000),
            grad_clip=training_config.get("grad_clip", 5.0),
            log_interval=log_interval,
            eval_interval=eval_interval,
            save_interval=save_interval,
            early_stopping_patience=training_config.get("early_stopping_patience", 5),
            scheduler=training_config.get("scheduler", "linear"),
            use_fp16=training_config.get("use_fp16", False),
            device=device,
            gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 1)
        )
        
        # Create the actual trainer
        self.trainer = XLSRTransducerTrainer(
            model=self.model,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            args=args
        )
        
        # Load checkpoint if provided
        if resume_checkpoint is not None and os.path.exists(resume_checkpoint):
            self.trainer.load_checkpoint(resume_checkpoint)
    
    def train(self):
        """Run training and return the path to the best checkpoint."""
        results = self.trainer.train()
        
        # Return the path to the best checkpoint
        output_dir = self.trainer.args.output_dir
        best_model_path = os.path.join(output_dir, "best_model.pt")
        
        return best_model_path 