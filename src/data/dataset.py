import os
import torch
import torchaudio
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset, DataLoader
from .processor import XLSRTransducerProcessor


class EstonianASRDataset(Dataset):
    """Dataset for Estonian ASR using XLSR-Transducer."""
    
    def __init__(
        self,
        manifest_path: str,
        processor: XLSRTransducerProcessor,
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        audio_dir: Optional[str] = None,
        debug: bool = False,
    ):
        """
        Initialize the dataset.
        
        Args:
            manifest_path: Path to the manifest file with format:
                           path_to_audio|transcription|speaker_id
            processor: Processor for audio and text
            max_duration: Maximum duration of audio in seconds
            min_duration: Minimum duration of audio in seconds
            audio_dir: Base directory for audio files if paths in manifest are relative
            debug: Whether to run in debug mode (include missing files with dummy audio)
        """
        self.processor = processor
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.audio_dir = audio_dir
        self.debug = debug
        
        # Load manifest
        self.samples = self._load_manifest(manifest_path)
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in manifest: {manifest_path}")
    
    def _load_manifest(self, manifest_path: str) -> List[Dict[str, Union[str, int]]]:
        """Load and parse the manifest file."""
        samples = []
        
        # Speaker ID mapping (for non-numeric speaker IDs)
        speaker_to_id = {}
        
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split("|")
                if len(parts) != 3:
                    print(f"Warning: Skipping malformed line: {line}")
                    continue
                
                audio_path, text, speaker_id = parts
                
                # Convert speaker_id to a numeric value if it's not already numeric
                if not speaker_id.isdigit():
                    if speaker_id not in speaker_to_id:
                        # Assign a new numeric ID to this speaker
                        speaker_to_id[speaker_id] = len(speaker_to_id)
                    speaker_id_numeric = speaker_to_id[speaker_id]
                else:
                    speaker_id_numeric = int(speaker_id)
                
                # If audio_dir is provided, join with audio_path
                if self.audio_dir is not None:
                    audio_path = os.path.join(self.audio_dir, audio_path)
                
                # Check if file exists
                file_exists = os.path.exists(audio_path)
                if not file_exists:
                    if self.debug:
                        # In debug mode, include the file with a flag indicating it's missing
                        print(f"Debug mode: Using dummy audio for missing file: {audio_path}")
                        samples.append({
                            "audio_path": audio_path,
                            "text": text,
                            "speaker_id": speaker_id_numeric,
                            "missing_audio": True
                        })
                        continue
                    else:
                        print(f"Warning: Audio file not found: {audio_path}")
                        continue
                
                # Check duration if min/max duration is specified
                if self.min_duration is not None or self.max_duration is not None:
                    try:
                        info = torchaudio.info(audio_path)
                        duration = info.num_frames / info.sample_rate
                        
                        if self.min_duration is not None and duration < self.min_duration:
                            if self.debug:
                                # In debug mode, include short files
                                print(f"Debug mode: Including short audio file: {audio_path} ({duration:.2f}s)")
                            else:
                                print(f"Warning: Audio file too short: {audio_path} ({duration:.2f}s)")
                                continue
                        
                        if self.max_duration is not None and duration > self.max_duration:
                            if self.debug:
                                # In debug mode, include long files
                                print(f"Debug mode: Including long audio file: {audio_path} ({duration:.2f}s)")
                            else:
                                print(f"Warning: Audio file too long: {audio_path} ({duration:.2f}s)")
                                continue
                    except Exception as e:
                        if self.debug:
                            # In debug mode, include files with errors
                            print(f"Debug mode: Using dummy audio for file with error: {audio_path}: {e}")
                            samples.append({
                                "audio_path": audio_path,
                                "text": text,
                                "speaker_id": speaker_id_numeric,
                                "missing_audio": True
                            })
                            continue
                        else:
                            print(f"Warning: Could not get duration for {audio_path}: {e}")
                            continue
                
                samples.append({
                    "audio_path": audio_path,
                    "text": text,
                    "speaker_id": speaker_id_numeric,
                    "missing_audio": False
                })
        
        return samples
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset."""
        sample = self.samples[idx]
        
        # Process audio and text
        if sample.get("missing_audio", False):
            # Generate dummy audio for missing files
            # Use a short, fixed-length audio tensor of zeros
            dummy_audio = torch.zeros(16000)  # 1 second of silence at 16kHz
            processed = self.processor(
                audio=dummy_audio,
                text=sample["text"]
            )
        else:
            processed = self.processor(
                audio_path=sample["audio_path"],
                text=sample["text"]
            )
        
        # Add speaker ID
        processed["speaker_id"] = torch.tensor(sample["speaker_id"], dtype=torch.long)
        
        return processed


def collate_fn(batch: List[Dict[str, torch.Tensor]], vocab_size: int = 60) -> Dict[str, torch.Tensor]:
    """
    Robust collate function for DataLoader.
    
    Args:
        batch: List of samples from the dataset
        vocab_size: Size of the vocabulary for clamping label values
        
    Returns:
        Batched samples with padded sequences
    """
    # Filter out any samples with empty tensors
    valid_batch = []
    for sample in batch:
        if all(k in sample for k in ["input_values", "labels", "speaker_id"]):
            if sample["input_values"].numel() > 0 and sample["labels"].numel() > 0:
                # Check if any labels are out of range and clip them
                if (sample["labels"] >= vocab_size).any():
                    print(f"WARNING: Found label values >= vocab_size ({vocab_size}). Clipping to valid range.")
                    sample["labels"] = torch.clamp(sample["labels"], max=vocab_size-1)
                valid_batch.append(sample)
            else:
                print(f"Skipping sample with empty tensor: input_values shape={sample['input_values'].shape}, labels shape={sample['labels'].shape}")
    
    # If the entire batch is invalid, create a dummy batch
    if len(valid_batch) == 0:
        print("WARNING: Entire batch invalid, creating dummy batch")
        # Create a dummy sample with valid data
        dummy_sample = {
            "input_values": torch.zeros(1, 16000),  # 1 second at 16kHz
            "labels": torch.ones(1, dtype=torch.long),  # Single token
            "speaker_id": torch.tensor(0, dtype=torch.long)
        }
        valid_batch = [dummy_sample, dummy_sample]  # Need at least 2 samples
    
    # Get max lengths
    max_input_length = max(x["input_values"].shape[1] for x in valid_batch)
    max_label_length = max(max(x["labels"].shape[0], 1) for x in valid_batch)  # Ensure at least length 1
    
    # Initialize tensors
    batch_size = len(valid_batch)
    input_values = torch.zeros(batch_size, 1, max_input_length)
    attention_mask = torch.zeros(batch_size, max_input_length, dtype=torch.long)
    
    # Determine pad token - use first token of first sample's labels, or 0 if unavailable
    pad_token = valid_batch[0]["labels"][0].item() if valid_batch[0]["labels"].numel() > 0 else 0
    
    # Initialize labels with pad tokens
    labels = torch.full((batch_size, max_label_length), pad_token, dtype=torch.long)
    label_lengths = torch.ones(batch_size, dtype=torch.long)  # Initialize with 1 (minimum valid length)
    speaker_ids = torch.zeros(batch_size, dtype=torch.long)
    
    # Fill tensors with robust handling
    for i, sample in enumerate(valid_batch):
        # Process input values
        input_length = min(sample["input_values"].shape[1], max_input_length)
        input_values[i, 0, :input_length] = sample["input_values"][:, :input_length]
        attention_mask[i, :input_length] = 1
        
        # Process labels
        label_tensor = sample["labels"]
        label_length = min(label_tensor.shape[0], max_label_length)
        
        # Ensure label_length is at least 1
        if label_length == 0:
            label_length = 1
            label_tensor = torch.full((1,), pad_token, dtype=torch.long, device=label_tensor.device)
        
        # Fill labels and set length
        labels[i, :label_length] = label_tensor[:label_length]
        label_lengths[i] = label_length
        
        # Set speaker ID
        speaker_ids[i] = sample["speaker_id"]
    
    # Validate the tensors
    try:
        # Check for NaN values
        for name, tensor in [
            ("input_values", input_values), 
            ("labels", labels), 
            ("label_lengths", label_lengths)
        ]:
            if torch.isnan(tensor.float()).any():
                print(f"WARNING: NaN values in {name} after collation!")
                tensor = torch.nan_to_num(tensor.float(), nan=0.0)
                if name == "labels" or name == "label_lengths":
                    tensor = tensor.long()
        
        # Ensure label_lengths are valid
        label_lengths = torch.clamp(label_lengths, min=1, max=max_label_length)
        
        # Squeeze input_values to match expected format
        input_values = input_values.squeeze(1)  # [batch, 1, length] -> [batch, length]
        
        print(f"DEBUG - Batch shapes: input={input_values.shape}, attn={attention_mask.shape}, labels={labels.shape}, lengths={label_lengths.shape}")
    except Exception as e:
        print(f"ERROR in collate_fn: {e}")
        # Return minimal valid batch as fallback
        return {
            "input_values": torch.zeros(batch_size, 16000),
            "attention_mask": torch.ones(batch_size, 16000, dtype=torch.long),
            "labels": torch.ones(batch_size, 1, dtype=torch.long),
            "label_lengths": torch.ones(batch_size, dtype=torch.long),
            "speaker_ids": torch.zeros(batch_size, dtype=torch.long)
        }
    
    return {
        "input_values": input_values,
        "attention_mask": attention_mask,
        "labels": labels,
        "label_lengths": label_lengths,
        "speaker_ids": speaker_ids
    }


def create_dataloader(
    manifest_path: str,
    processor: XLSRTransducerProcessor,
    batch_size: int = 8,
    max_duration: Optional[float] = None,
    min_duration: Optional[float] = None,
    audio_dir: Optional[str] = None,
    num_workers: int = 4,
    shuffle: bool = True
) -> DataLoader:
    """
    Create a DataLoader for the Estonian ASR dataset.
    
    Args:
        manifest_path: Path to the manifest file
        processor: Processor for audio and text
        batch_size: Batch size
        max_duration: Maximum duration of audio in seconds
        min_duration: Minimum duration of audio in seconds
        audio_dir: Base directory for audio files if paths in manifest are relative
        num_workers: Number of workers for DataLoader
        shuffle: Whether to shuffle the dataset
        
    Returns:
        DataLoader for the dataset
    """
    dataset = EstonianASRDataset(
        manifest_path=manifest_path,
        processor=processor,
        max_duration=max_duration,
        min_duration=min_duration,
        audio_dir=audio_dir
    )
    
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    ) 