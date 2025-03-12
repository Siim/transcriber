import os
import torch
import torchaudio
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from torch.utils.data import Dataset, DataLoader
from .processor import XLSRTransducerProcessor
import random


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


def collate_fn(batch, max_input_length=None, debug=False, processor=None):
    """
    Collate function for the DataLoader.
    
    Args:
        batch: List of samples from the dataset
        max_input_length: Maximum input length to limit sequence length
        debug: Whether to print debug information
        processor: The processor to use for vocabulary size checking
        
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
        # Check if processor key exists in the sample
        if "processor" in sample and "labels" in sample:
            if torch.max(sample["labels"]) >= sample["processor"].tokenizer.vocab_size:
                print(f"WARNING: Label exceeds vocabulary size: {torch.max(sample['labels']).item()} >= {sample['processor'].tokenizer.vocab_size}")
                continue
        elif "labels" in sample and processor is not None:
            # Use provided processor for vocabulary size check
            if torch.max(sample["labels"]) >= processor.vocab_size:
                print(f"WARNING: Label exceeds provided processor vocabulary size: {torch.max(sample['labels']).item()} >= {processor.vocab_size}")
                continue
        elif "labels" in sample:
            # If processor is not in the sample, we can't check the vocab size
            # Just make sure the labels are not too large - use 1000 which matches our BPE config
            if torch.max(sample["labels"]) >= 1000:  # Match BPE vocab size from config
                print(f"WARNING: Label exceeds BPE vocabulary size: {torch.max(sample['labels']).item()} >= 1000")
                continue
            
        valid_batch.append(sample)
    
    # If all samples are invalid, create a dummy batch
    if len(valid_batch) == 0:
        print("WARNING: All samples in batch are invalid, creating dummy batch")
        if debug:
            print("DEBUG: Creating dummy batch with label_lengths min: 1, max: 1")
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


class LengthBatchSampler(torch.utils.data.Sampler):
    """
    Batch sampler that groups samples of similar lengths together.
    This reduces padding and improves training efficiency.
    
    Args:
        dataset: Dataset to sample from
        batch_size: Batch size
        shuffle: Whether to shuffle the dataset
        drop_last: Whether to drop the last incomplete batch
        sort_key: Function to extract the length from a sample
        bucket_size_multiplier: Multiplier for bucket size (larger values = more randomness)
    """
    
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        sort_key: Optional[Callable] = None,
        bucket_size_multiplier: int = 5
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.bucket_size_multiplier = bucket_size_multiplier
        
        # Default sort key extracts audio path and estimates length
        if sort_key is None:
            self.sort_key = lambda idx: self._get_audio_length(idx)
        else:
            self.sort_key = sort_key
            
        # Cache audio lengths to avoid recomputing
        self.lengths = {}
        
    def _get_audio_length(self, idx: int) -> int:
        """Get the length of an audio file in samples."""
        sample = self.dataset.samples[idx]
        audio_path = sample["audio_path"]
        
        if audio_path in self.lengths:
            return self.lengths[audio_path]
        
        # Check if the audio file is missing
        if sample.get("missing_audio", False):
            # Use a default length for missing files
            length = 16000 * 1  # Default to 1 second for missing files
        else:
            try:
                # Try to get actual audio length
                info = torchaudio.info(audio_path)
                length = info.num_frames
            except Exception:
                # If file doesn't exist or has issues, use a default length
                length = 16000 * 5  # Default to 5 seconds
            
        self.lengths[audio_path] = length
        return length
    
    def __iter__(self):
        # Get indices and their corresponding lengths
        indices = list(range(len(self.dataset)))
        
        # Sort indices by length
        try:
            indices.sort(key=self.sort_key)
        except Exception as e:
            print(f"Warning: Error sorting by length: {e}. Using unsorted indices.")
        
        # Create buckets of size batch_size * bucket_size_multiplier
        bucket_size = self.batch_size * self.bucket_size_multiplier
        
        # Split indices into buckets and shuffle each bucket
        buckets = [indices[i:i + bucket_size] for i in range(0, len(indices), bucket_size)]
        
        if self.shuffle:
            # Shuffle the order of buckets
            random.shuffle(buckets)
            
            # Shuffle samples within each bucket
            for bucket in buckets:
                random.shuffle(bucket)
        
        # Create batches from buckets
        batches = []
        for bucket in buckets:
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)
        
        # Shuffle the order of batches if needed
        if self.shuffle:
            random.shuffle(batches)
        
        # Flatten batches
        for batch in batches:
            yield batch
    
    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def create_length_sorted_dataloader(
    manifest_path: str,
    processor: XLSRTransducerProcessor,
    batch_size: int = 8,
    max_duration: Optional[float] = None,
    min_duration: Optional[float] = None,
    audio_dir: Optional[str] = None,
    num_workers: int = 4,
    shuffle: bool = True,
    drop_last: bool = False,
    bucket_size_multiplier: int = 5,
    debug: bool = False
) -> DataLoader:
    """
    Create a DataLoader with length-based batch sampling for the Estonian ASR dataset.
    This reduces padding and improves training efficiency.
    
    Args:
        manifest_path: Path to the manifest file
        processor: Processor for audio and text
        batch_size: Batch size
        max_duration: Maximum duration of audio in seconds
        min_duration: Minimum duration of audio in seconds
        audio_dir: Base directory for audio files if paths in manifest are relative
        num_workers: Number of workers for DataLoader
        shuffle: Whether to shuffle the dataset
        drop_last: Whether to drop the last incomplete batch
        bucket_size_multiplier: Multiplier for bucket size (larger values = more randomness)
        debug: Whether to run in debug mode (include missing files with dummy audio)
        
    Returns:
        DataLoader for the dataset with length-based batch sampling
    """
    import random
    
    dataset = EstonianASRDataset(
        manifest_path=manifest_path,
        processor=processor,
        max_duration=max_duration,
        min_duration=min_duration,
        audio_dir=audio_dir,
        debug=debug
    )
    
    # Create length-based batch sampler
    batch_sampler = LengthBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        bucket_size_multiplier=bucket_size_multiplier
    )
    
    # Create a collate function wrapper that passes the processor
    collate_fn_with_processor = lambda batch: collate_fn(
        batch, 
        max_input_length=int(max_duration * processor.audio_preprocessor.sample_rate) if max_duration else None,
        debug=debug,
        processor=processor
    )
    
    return DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn_with_processor,
        pin_memory=True
    ) 