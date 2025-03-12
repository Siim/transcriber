import os
import torch
import torchaudio
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from torch.utils.data import Dataset, DataLoader
from .processor import XLSRTransducerProcessor
import random
from tqdm import tqdm
import time
import concurrent.futures
import multiprocessing
from functools import lru_cache


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
        
        # Enforce stricter duration limits for more efficient training
        self.min_duration = max(1.0, min_duration) if min_duration is not None else 1.0
        self.max_duration = min(10.0, max_duration) if max_duration is not None else 10.0
        
        if min_duration is not None and min_duration < 1.0:
            print(f"Warning: min_duration {min_duration}s is less than recommended 1.0s. Using 1.0s.")
            
        if max_duration is not None and max_duration > 10.0:
            print(f"Warning: max_duration {max_duration}s is greater than recommended 10.0s. Using 10.0s.")
        
        self.audio_dir = audio_dir
        self.debug = debug
        
        # Cache for processed audio - use a small LRU cache to avoid memory issues
        self._preprocess_audio_cached = lru_cache(maxsize=64)(self._preprocess_audio_impl)
        
        # Load manifest
        self.samples = self._load_manifest(manifest_path)
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in manifest: {manifest_path}")
            
        # Print duration statistics
        if debug:
            durations = [sample["duration"] for sample in self.samples]
            if durations:
                min_dur = min(durations)
                max_dur = max(durations)
                avg_dur = sum(durations) / len(durations)
                print(f"Audio duration stats: min={min_dur:.2f}s, max={max_dur:.2f}s, avg={avg_dur:.2f}s")
    
    def _load_manifest(self, manifest_path: str) -> List[Dict[str, Union[str, int]]]:
        """Load and parse the manifest file with multithreaded audio duration checking."""
        # Function to check audio file and get duration
        def check_audio_file(audio_path, text, speaker_id, min_duration, max_duration):
            # If audio_dir is provided, join with audio_path
            full_audio_path = os.path.join(self.audio_dir, audio_path) if self.audio_dir is not None else audio_path
            
            # Convert speaker_id to a numeric value if it's not already numeric
            if not speaker_id.isdigit():
                # We'll handle speaker ID mapping in the main function
                speaker_id_numeric = -1  # temporary value, will be updated later
            else:
                speaker_id_numeric = int(speaker_id)
            
            # Check if file exists
            file_exists = os.path.exists(full_audio_path)
            if not file_exists:
                if self.debug:
                    # In debug mode, include the file with a flag indicating it's missing
                    return {
                        "audio_path": full_audio_path,
                        "text": text,
                        "speaker_id": speaker_id_numeric,  # Always an integer
                        "speaker_id_raw": speaker_id,      # Original value for mapping
                        "missing_audio": True,
                        "duration": 2.0,  # Default duration for dummy audio
                        "status": "missing"
                    }
                else:
                    return {"status": "missing"}
            
            # Check duration - always check to ensure consistent batching
            try:
                info = torchaudio.info(full_audio_path)
                duration = info.num_frames / info.sample_rate
                
                # Strictly enforce duration limits for more consistent batching
                if duration < min_duration:
                    return {"status": "too_short", "duration": duration}
                
                if duration > max_duration:
                    return {"status": "too_long", "duration": duration}
                
                return {
                    "audio_path": full_audio_path,
                    "text": text,
                    "speaker_id": speaker_id_numeric,
                    "missing_audio": False,
                    "duration": duration,
                    "status": "valid"
                }
                
            except Exception as e:
                if self.debug:
                    # In debug mode, include files with errors
                    return {
                        "audio_path": full_audio_path,
                        "text": text,
                        "speaker_id": speaker_id_numeric,
                        "missing_audio": True,
                        "duration": 2.0,  # Default duration for dummy audio
                        "status": "error",
                        "error": str(e)
                    }
                else:
                    return {"status": "error", "error": str(e)}
        
        samples = []
        
        # Speaker ID mapping (for non-numeric speaker IDs)
        speaker_to_id = {}
        
        # Set strict duration limits
        min_duration = max(1.0, self.min_duration) if self.min_duration is not None else 1.0
        max_duration = min(10.0, self.max_duration) if self.max_duration is not None else 10.0
        
        # Count total and filtered samples for debug
        total_samples = 0
        filtered_samples = 0
        too_short = 0
        too_long = 0
        missing_files = 0
        
        with open(manifest_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        
        total_samples = len(lines)
        
        # Process lines in parallel with a thread pool (more appropriate for I/O bound tasks)
        # Determine number of threads - for I/O bound tasks, we can use more threads
        num_threads = min(32, multiprocessing.cpu_count() * 4)  # Use more threads for I/O
        
        print(f"Processing manifest with {num_threads} threads")
        
        # Prepare data for thread processing
        tasks = []
        for line in lines:
            parts = line.split("|")
            if len(parts) != 3:
                print(f"Warning: Skipping malformed line: {line}")
                filtered_samples += 1
                continue
            
            audio_path, text, speaker_id = parts
            tasks.append((audio_path, text, speaker_id, min_duration, max_duration))
        
        # Use tqdm in debug mode to show progress
        if self.debug:
            print(f"Checking {len(tasks)} audio files for validity...")
            progress_bar = tqdm(total=len(tasks), desc="Loading manifest")
        
        # Process in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(check_audio_file, *task) for task in tasks]
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                
                if self.debug:
                    progress_bar.update(1)
                
                # Handle different status codes
                if result["status"] == "valid":
                    # Handle speaker ID mapping for valid files
                    speaker_id = result.get("speaker_id_raw", result["speaker_id"])
                    if isinstance(speaker_id, str) and not speaker_id.isdigit():
                        if speaker_id not in speaker_to_id:
                            speaker_to_id[speaker_id] = len(speaker_to_id)
                        result["speaker_id"] = speaker_to_id[speaker_id]
                    
                    samples.append(result)
                elif result["status"] == "missing":
                    missing_files += 1
                    filtered_samples += 1
                    if self.debug and "audio_path" in result:
                        # Handle speaker ID mapping for missing files in debug mode
                        speaker_id = result.get("speaker_id_raw", result["speaker_id"])
                        if isinstance(speaker_id, str) and not speaker_id.isdigit():
                            if speaker_id not in speaker_to_id:
                                speaker_to_id[speaker_id] = len(speaker_to_id)
                            result["speaker_id"] = speaker_to_id[speaker_id]
                        
                        samples.append(result)  # Include with dummy audio in debug mode
                elif result["status"] == "too_short":
                    too_short += 1
                    filtered_samples += 1
                    if self.debug:
                        print(f"Debug mode: Skipping too short audio file: {result.get('audio_path', 'unknown')} ({result.get('duration', 0):.2f}s < {min_duration:.2f}s)")
                elif result["status"] == "too_long":
                    too_long += 1
                    filtered_samples += 1
                    if self.debug:
                        print(f"Debug mode: Skipping too long audio file: {result.get('audio_path', 'unknown')} ({result.get('duration', 0):.2f}s > {max_duration:.2f}s)")
                elif result["status"] == "error":
                    filtered_samples += 1
                    if self.debug and "audio_path" in result:
                        print(f"Debug mode: Using dummy audio for file with error: {result['audio_path']}: {result.get('error', 'unknown error')}")
                        
                        # Handle speaker ID mapping for error files in debug mode
                        speaker_id = result.get("speaker_id_raw", result["speaker_id"])
                        if isinstance(speaker_id, str) and not speaker_id.isdigit():
                            if speaker_id not in speaker_to_id:
                                speaker_to_id[speaker_id] = len(speaker_to_id)
                            result["speaker_id"] = speaker_to_id[speaker_id]
                        
                        samples.append(result)  # Include with dummy audio in debug mode
        
        if self.debug:
            progress_bar.close()
        
        # Print summary of filtered samples
        if filtered_samples > 0:
            print(f"Filtered {filtered_samples}/{total_samples} samples "
                  f"(too short: {too_short}, too long: {too_long}, missing: {missing_files})")
            
        return samples
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def _preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """
        Load and preprocess audio more efficiently using cached results.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Preprocessed audio tensor with shape [1, length]
        """
        return self._preprocess_audio_cached(audio_path)
    
    def _preprocess_audio_impl(self, audio_path: str) -> torch.Tensor:
        """
        Implementation of audio preprocessing with optimized memory operations.
        
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
                    # Generate a short dummy audio - use low noise for efficiency
                    return torch.zeros(1, 16000, dtype=torch.float) + torch.randn(1, 16000, dtype=torch.float) * 1e-6
                else:
                    raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Load audio using a more memory-efficient approach
            # Use torchaudio.load with memory mapping when available
            try:
                # First try loading with normal method
                waveform, sample_rate = torchaudio.load(audio_path)
            except Exception:
                # If that fails, try reading as numpy array (more memory efficient for large files)
                import numpy as np
                import soundfile as sf
                data, sample_rate = sf.read(audio_path, dtype='float32')
                waveform = torch.from_numpy(data.reshape(1, -1) if data.ndim == 1 else data.T)
            
            # Convert to mono if needed - ensure shape is [1, length]
            if waveform.dim() > 1:
                if waveform.shape[0] > 1:
                    # Use more efficient operation for sum
                    waveform = waveform.mean(dim=0, keepdim=True)
            else:
                # If for some reason it's 1D, make it 2D [1, length]
                waveform = waveform.unsqueeze(0)
            
            # Resample if needed
            if sample_rate != 16000:
                # Create resampler only when needed (avoid recreation)
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=16000
                )
                waveform = resampler(waveform)
            
            # Limit audio length to max_duration to prevent extremely long sequences
            # Use more memory-efficient approach for long files
            if self.max_duration is not None:
                max_samples = int(self.max_duration * 16000)
                if waveform.shape[1] > max_samples:
                    # Use narrow instead of slicing for memory efficiency
                    waveform = waveform.narrow(1, 0, max_samples)
            
            # Normalize audio - use in-place operations
            if waveform.abs().max() > 0:
                max_val = waveform.abs().max()
                waveform.div_(max_val)
            
            # Ensure the shape is exactly [1, length]
            if waveform.dim() > 2:
                waveform = waveform.reshape(1, -1)
                
            return waveform
        
        except Exception as e:
            if self.debug:
                print(f"Error loading audio {audio_path}: {str(e)}. Using dummy audio.")
                return torch.zeros(1, 16000, dtype=torch.float) + torch.randn(1, 16000, dtype=torch.float) * 1e-6
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


def collate_fn(batch, max_input_length=None, debug=False, processor=None, static_memory=True):
    """
    Collate function for the DataLoader.
    
    Args:
        batch: List of samples from the dataset
        max_input_length: Maximum input length to limit sequence length
        debug: Whether to print debug information
        processor: The processor to use for vocabulary size checking
        static_memory: Whether to use static memory allocation for batch tensors
        
    Returns:
        Dictionary with batched tensors
    """
    start_time = time.time() if debug else None
    
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
    
    # Measure filtering time
    if debug and start_time:
        filter_time = time.time() - start_time
        start_shape_time = time.time()
    
    # Get batch size and maximum lengths
    batch_size = len(valid_batch)
    
    # For static memory allocation (faster), we need to pre-determine dimensions
    if static_memory:
        # Pre-compute dimensions for allocation
        max_input_lengths = [sample["input_values"].shape[-1] for sample in valid_batch]
        max_input_length_in_batch = max(max_input_lengths)
        
        # Limit input length if specified
        if max_input_length is not None and max_input_length > 0:
            max_input_length_in_batch = min(max_input_length_in_batch, max_input_length)
            if debug:
                print(f"Limiting maximum sequence length to {max_input_length_in_batch}")
                
        # Pre-compute label dimensions
        label_lengths = torch.tensor([sample["labels"].shape[0] for sample in valid_batch], dtype=torch.long)
        max_label_length = label_lengths.max().item()
        
        # Initialize input tensors with zeros - use static allocation
        input_values = torch.zeros(batch_size, 1, max_input_length_in_batch)
        attention_mask = torch.zeros(batch_size, max_input_length_in_batch)
        
        # Determine pad token - use a default of 0
        pad_token = 0
        
        # Initialize labels with pad tokens
        labels = torch.full((batch_size, max_label_length), pad_token, dtype=torch.long)
        speaker_ids = torch.zeros(batch_size, dtype=torch.long)
        
        # Fill tensors efficiently
        for i, sample in enumerate(valid_batch):
            # Extract input values (make sure it's 2D)
            # Handle potentially different shapes by getting the raw tensor
            input_tensor = sample["input_values"]
            if input_tensor.dim() == 3:  # [1, 1, length]
                input_tensor = input_tensor.squeeze(0)
            elif input_tensor.dim() == 1:  # Convert [length] to [1, length]
                input_tensor = input_tensor.unsqueeze(0)
                
            # Input values and attention mask
            input_length = min(input_tensor.shape[1], max_input_length_in_batch)
            input_values[i, 0, :input_length] = input_tensor[0, :input_length]
            attention_mask[i, :input_length] = 1
            
            # Labels - already have the label lengths from earlier computation
            label_length = label_lengths[i].item()
            labels[i, :label_length] = sample["labels"]
            
            # Speaker ID
            speaker_ids[i] = sample["speaker_id"]
    else:
        # Dynamic memory allocation (slower but more flexible)
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
    
    # Measure tensor construction time
    if debug and start_time:
        shape_time = time.time() - start_shape_time
        total_time = time.time() - start_time
        
        # Calculate tensor sizes for memory usage tracking
        input_size_mb = input_values.element_size() * input_values.nelement() / (1024 * 1024)
        labels_size_mb = labels.element_size() * labels.nelement() / (1024 * 1024)
        total_size_mb = input_size_mb + labels_size_mb
        
        print(f"DEBUG - Batch collation timing: filter={filter_time:.4f}s, shape={shape_time:.4f}s, total={total_time:.4f}s")
        print(f"DEBUG - Batch memory: input={input_size_mb:.2f}MB, labels={labels_size_mb:.2f}MB, total={total_size_mb:.2f}MB")
        print(f"DEBUG - Batch shapes: input={input_values.shape}, attn={attention_mask.shape}, labels={labels.shape}, lengths={label_lengths.shape}")
    
    # More debug info for label lengths
    if debug:
        if label_lengths.numel() > 0:
            min_len = label_lengths.min().item()
            max_len = label_lengths.max().item()
            avg_len = label_lengths.float().mean().item()
            print(f"After fixing: Label lengths min: {min_len}, max: {max_len}, avg: {avg_len:.1f}")
            print(f"DEBUG: label_lengths min: {min_len}, max: {max_len}, shape: {label_lengths.shape}")
            print(f"DEBUG: labels shape: {labels.shape}")
    
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
        use_log_buckets: Whether to use logarithmic bucketing for better grouping
    """
    
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        sort_key: Optional[Callable] = None,
        bucket_size_multiplier: int = 5,
        use_log_buckets: bool = True
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.bucket_size_multiplier = bucket_size_multiplier
        self.use_log_buckets = use_log_buckets
        
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
        
        # Use pre-computed duration if available
        if "duration" in sample:
            # Convert to samples for sorting
            length = int(sample["duration"] * 16000)
        # Check if the audio file is missing
        elif sample.get("missing_audio", False):
            # Use a default length for missing files
            length = 16000 * 2  # Default to 2 seconds for missing files
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
        
        # Get lengths for all indices
        lengths = [self.sort_key(idx) for idx in indices]
        
        # Apply logarithmic bucketing for better grouping if enabled
        if self.use_log_buckets:
            # Apply log transformation to smoothly group similar lengths
            # Use log base 1.1 to create more fine-grained buckets
            min_length = max(1, min(lengths))  # Avoid log(0)
            bucket_fn = lambda l: int(np.log(max(l, min_length) / min_length) / np.log(1.1))
            bucket_indices = [(bucket_fn(lengths[i]), i) for i in range(len(indices))]
            
            # Sort by bucket first, then by actual length within bucket
            bucket_indices.sort()
            
            # Extract sorted indices
            indices = [idx for _, idx in bucket_indices]
        else:
            # Sort indices by length directly
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
    debug: bool = False,
    use_log_buckets: bool = True,
    pin_memory: bool = True,
    prefetch_factor: int = 2
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
        use_log_buckets: Whether to use logarithmic bucketing for better length grouping
        pin_memory: Whether to use pin_memory for faster GPU transfers
        prefetch_factor: Number of batches to prefetch per worker (higher = more memory, faster loading)
        
    Returns:
        DataLoader for the dataset with length-based batch sampling
    """
    import random
    
    if debug:
        print(f"Creating length-sorted dataloader with min_duration={min_duration}s, max_duration={max_duration}s")
    
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
        bucket_size_multiplier=bucket_size_multiplier,
        use_log_buckets=use_log_buckets
    )
    
    # Create a collate function wrapper that passes the processor
    collate_fn_with_processor = lambda batch: collate_fn(
        batch, 
        max_input_length=int(max_duration * processor.audio_preprocessor.sample_rate) if max_duration else None,
        debug=debug,
        processor=processor,
        static_memory=True
    )
    
    # Report dataloader settings in debug mode
    if debug:
        print(f"DataLoader settings: batch_size={batch_size}, num_workers={num_workers}, "
              f"shuffle={shuffle}, drop_last={drop_last}, bucket_size_multiplier={bucket_size_multiplier}")
    
    # Create DataLoader with optimized settings
    dataloader_kwargs = {
        'dataset': dataset,
        'batch_sampler': batch_sampler,
        'num_workers': num_workers,
        'collate_fn': collate_fn_with_processor,
        'pin_memory': pin_memory
    }
    
    # Only add prefetch_factor if num_workers > 0 (otherwise it causes an error)
    if num_workers > 0:
        dataloader_kwargs['prefetch_factor'] = prefetch_factor
    
    return DataLoader(**dataloader_kwargs) 