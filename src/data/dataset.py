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
    
    def _preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """
        Load and preprocess audio more efficiently.
        Limits audio length to max_duration.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Preprocessed audio tensor with shape [1, length]
        """
        try:
            # Check if file exists
            if not os.path.exists(audio_path):
                if self.debug:
                    print(f"Debug mode: Using dummy audio for missing file: {audio_path}")
                    # Generate a short dummy audio
                    return torch.zeros(1, 16000) + torch.randn(1, 16000) * 1e-6
                else:
                    raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono if needed - ensure shape is [1, length]
            if waveform.dim() > 1:
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
            else:
                # If for some reason it's 1D, make it 2D [1, length]
                waveform = waveform.unsqueeze(0)
            
            # Resample if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=16000
                )
                waveform = resampler(waveform)
            
            # Limit audio length to max_duration to prevent extremely long sequences
            if self.max_duration is not None:
                max_samples = int(self.max_duration * 16000)
                if waveform.shape[1] > max_samples:
                    # Either take the first max_samples (consistent) or randomly crop (for data augmentation)
                    # using consistent for now for stability
                    waveform = waveform[:, :max_samples]
            
            # Normalize audio
            if waveform.abs().max() > 0:
                waveform = waveform / waveform.abs().max()
            
            # Ensure the shape is exactly [1, length]
            if waveform.dim() > 2:
                waveform = waveform.view(1, -1)
                
            return waveform
        
        except Exception as e:
            if self.debug:
                print(f"Error loading audio {audio_path}: {str(e)}. Using dummy audio.")
                return torch.zeros(1, 16000) + torch.randn(1, 16000) * 1e-6
            else:
                raise
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset."""
        sample = self.samples[idx]
        
        # Process audio and text
        if sample.get("missing_audio", False):
            # Generate dummy audio for missing files
            dummy_audio = torch.zeros(1, 16000) + torch.randn(1, 16000) * 1e-6  # Add small noise to prevent NaN
            processed = self.processor(
                audio=dummy_audio,
                text=sample["text"]
            )
        else:
            # Use optimized audio preprocessing
            audio = self._preprocess_audio(sample["audio_path"])
            processed = self.processor(
                audio=audio,
                text=sample["text"]
            )
        
        # Add speaker ID
        processed["speaker_id"] = torch.tensor(sample["speaker_id"], dtype=torch.long)
        
        return processed


def collate_fn(batch, max_input_length=None, debug=False):
    """
    Collate function for the DataLoader.
    
    Args:
        batch: List of samples from the dataset
        max_input_length: Maximum input length to limit sequence length
        debug: Whether to print debug information
        
    Returns:
        Dictionary with batched tensors
    """
    # Filter out samples with empty tensors
    valid_batch = []
    for sample in batch:
        # Skip samples with empty input values
        if sample["input_values"].numel() == 0:
            continue
            
        # Skip samples with labels that exceed vocabulary size
        if torch.max(sample["labels"]) >= sample["processor"].tokenizer.vocab_size:
            print(f"WARNING: Label exceeds vocabulary size: {torch.max(sample['labels']).item()} >= {sample['processor'].tokenizer.vocab_size}")
            continue
            
        valid_batch.append(sample)
    
    # If all samples are invalid, create a dummy batch
    if len(valid_batch) == 0:
        print("WARNING: All samples in batch are invalid, creating dummy batch")
        return {
            "input_values": torch.zeros(1, 1, 1000),
            "attention_mask": torch.zeros(1, 1000),
            "labels": torch.zeros(1, 1, dtype=torch.long),
            "label_lengths": torch.ones(1, dtype=torch.long),
            "speaker_id": torch.zeros(1, dtype=torch.long)
        }
    
    # Get batch size and maximum lengths
    batch_size = len(valid_batch)
    max_input_length_in_batch = max([sample["input_values"].shape[-1] for sample in valid_batch])
    
    # Limit input length if specified
    if max_input_length is not None and max_input_length > 0:
        max_input_length_in_batch = min(max_input_length_in_batch, max_input_length)
        if debug:
            print(f"Limiting maximum sequence length to {max_input_length_in_batch}")
    
    max_label_length = max([sample["labels"].shape[0] for sample in valid_batch])
    
    # Initialize input tensors with zeros
    input_values = torch.zeros(batch_size, max_input_length_in_batch)
    attention_mask = torch.zeros(batch_size, max_input_length_in_batch)
    
    # Determine pad token - use a default of 0
    pad_token = 0
    
    # Initialize labels with pad tokens
    labels = torch.full((batch_size, max_label_length), pad_token, dtype=torch.long)
    label_lengths = torch.zeros(batch_size, dtype=torch.long)
    speaker_ids = torch.zeros(batch_size, dtype=torch.long)
    
    # Fill tensors efficiently
    for i, sample in enumerate(valid_batch):
        # Extract input values (make sure it's 2D)
        # Handle potentially different shapes by getting the raw tensor
        if sample["input_values"].dim() == 3:  # [1, 1, length]
            sample_input = sample["input_values"].squeeze(0)
        elif sample["input_values"].dim() == 2:  # [1, length]
            sample_input = sample["input_values"]
        else:  # Just in case
            sample_input = sample["input_values"].view(1, -1)
            
        # Input values and attention mask
        input_length = min(sample_input.shape[1], max_input_length_in_batch)
        input_values[i, :input_length] = sample_input[0, :input_length]
        attention_mask[i, :input_length] = 1
        
        # Labels
        label_length = sample["labels"].shape[0]
        label_lengths[i] = label_length
        labels[i, :label_length] = sample["labels"]
        
        # Speaker ID
        speaker_ids[i] = sample["speaker_id"]
    
    # Add channel dimension - XLSR expects [batch_size, 1, sequence_length]
    input_values = input_values.unsqueeze(1)
    
    # Debug information for batch shapes
    if debug:
        print(f"DEBUG - Batch shapes: input={input_values.shape}, attn={attention_mask.shape}, labels={labels.shape}, lengths={label_lengths.shape}")
    
    return {
        "input_values": input_values,
        "attention_mask": attention_mask,
        "labels": labels,
        "label_lengths": label_lengths,
        "speaker_id": speaker_ids
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