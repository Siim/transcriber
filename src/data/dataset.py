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


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    
    Args:
        batch: List of samples from the dataset
        
    Returns:
        Batched samples with padded sequences
    """
    # Get max lengths
    max_input_length = max(x["input_values"].shape[1] for x in batch)
    max_label_length = max(x["labels"].shape[0] for x in batch)
    
    # Initialize tensors
    batch_size = len(batch)
    input_values = torch.zeros(batch_size, 1, max_input_length)
    attention_mask = torch.zeros(batch_size, max_input_length, dtype=torch.long)
    labels = torch.ones(batch_size, max_label_length, dtype=torch.long) * batch[0]["labels"][0]  # Pad with first token (usually pad token)
    label_lengths = torch.zeros(batch_size, dtype=torch.long)
    speaker_ids = torch.zeros(batch_size, dtype=torch.long)
    
    # Fill tensors
    for i, sample in enumerate(batch):
        input_length = sample["input_values"].shape[1]
        label_length = sample["labels"].shape[0]
        
        # Ensure label_length is at least 1 to prevent CUDA errors
        if label_length == 0:
            label_length = 1
            sample["labels"] = torch.ones(1, dtype=torch.long, device=sample["labels"].device) * batch[0]["labels"][0]
        
        input_values[i, 0, :input_length] = sample["input_values"]
        attention_mask[i, :input_length] = 1
        labels[i, :label_length] = sample["labels"]
        label_lengths[i] = label_length
        speaker_ids[i] = sample["speaker_id"]
    
    return {
        "input_values": input_values.squeeze(1),  # Change from [batch, 1, length] to [batch, length]
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