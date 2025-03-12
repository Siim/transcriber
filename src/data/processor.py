import os
import torch
import torchaudio
import numpy as np
from typing import Dict, List, Optional, Union
from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor
from dataclasses import dataclass


class CharacterTokenizer:
    """Simple character-level tokenizer for Estonian language."""
    
    def __init__(self, special_tokens: Dict[str, str] = None):
        if special_tokens is None:
            special_tokens = {
                "pad": "<pad>",
                "unk": "<unk>",
                "bos": "<s>",
                "eos": "</s>",
                "blank": "<blank>"
            }
        
        self.special_tokens = special_tokens
        self.special_tokens_list = list(special_tokens.values())
        
        # Estonian alphabet (including both uppercase and lowercase)
        # a through z + Estonian-specific characters
        estonian_chars = list("abcdefghijklmnopqrsšzžtuvwõäöüxy")
        
        # Digits
        digits = list("0123456789")
        
        # Punctuation and special characters
        punctuation = list(",.!?-':;\"()[]{}/%@#$&*+=<>|~^ ")
        
        # Create full alphabet
        self.alphabet = estonian_chars + digits + punctuation
        
        # Create vocabulary
        self.vocab = self.special_tokens_list + self.alphabet
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}
        
        # Special token IDs
        self.pad_id = self.token_to_id[special_tokens["pad"]]
        self.unk_id = self.token_to_id[special_tokens["unk"]]
        self.bos_id = self.token_to_id[special_tokens["bos"]]
        self.eos_id = self.token_to_id[special_tokens["eos"]]
        self.blank_id = self.token_to_id[special_tokens["blank"]]
    
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        tokens = []
        for char in text.lower():
            if char in self.token_to_id:
                tokens.append(self.token_to_id[char])
            else:
                tokens.append(self.unk_id)
        return tokens
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Convert token IDs to text."""
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if skip_special_tokens and token in self.special_tokens_list:
                    continue
                tokens.append(token)
            else:
                tokens.append(self.id_to_token[self.unk_id])
        return "".join(tokens)
    
    @property
    def vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        return len(self.vocab)


class BPETokenizer:
    """BPE tokenizer for Estonian language using HuggingFace's transformers."""
    
    def __init__(self, tokenizer_dir: str = "data/tokenizer", special_tokens: Dict[str, str] = None):
        """
        Initialize the BPE tokenizer.
        
        Args:
            tokenizer_dir: Directory containing the tokenizer files
            special_tokens: Dictionary of special tokens
        """
        from transformers import PreTrainedTokenizerFast
        import os
        
        if special_tokens is None:
            special_tokens = {
                "pad": "<pad>",
                "unk": "<unk>",
                "bos": "<s>",
                "eos": "</s>",
                "blank": "<blank>"
            }
        
        self.special_tokens = special_tokens
        self.special_tokens_list = list(special_tokens.values())
        
        # Check if tokenizer files exist
        tokenizer_file = os.path.join(tokenizer_dir, "tokenizer.json")
        if not os.path.exists(tokenizer_file):
            raise FileNotFoundError(
                f"Tokenizer file not found at {tokenizer_file}. "
                "Run create_bpe_tokenizer.py to create the BPE tokenizer."
            )
        
        # Load the tokenizer
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_file,
            bos_token=special_tokens["bos"],
            eos_token=special_tokens["eos"],
            unk_token=special_tokens["unk"],
            pad_token=special_tokens["pad"],
        )
        
        # Add blank token if it's not already in the tokenizer
        if special_tokens["blank"] not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": [special_tokens["blank"]]})
        
        # Map special token IDs
        self.pad_id = self.tokenizer.convert_tokens_to_ids(special_tokens["pad"])
        self.unk_id = self.tokenizer.convert_tokens_to_ids(special_tokens["unk"])
        self.bos_id = self.tokenizer.convert_tokens_to_ids(special_tokens["bos"])
        self.eos_id = self.tokenizer.convert_tokens_to_ids(special_tokens["eos"])
        self.blank_id = self.tokenizer.convert_tokens_to_ids(special_tokens["blank"])
        
        # Create reverse mapping for decoding
        self.id_to_token = {v: k for k, v in self.tokenizer.get_vocab().items()}
    
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        return self.tokenizer.encode(text.lower(), add_special_tokens=False)
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Convert token IDs to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    @property
    def vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        return len(self.tokenizer.get_vocab())


@dataclass
class AudioPreprocessor:
    """Audio preprocessing for XLSR-Transducer."""
    
    sample_rate: int = 16000
    feature_extractor: Optional[Wav2Vec2FeatureExtractor] = None
    
    def __post_init__(self):
        if self.feature_extractor is None:
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                "facebook/wav2vec2-large-xlsr-53",
                sampling_rate=self.sample_rate,
                return_attention_mask=True,
                do_normalize=True  # Ensure normalization is enabled
            )
    
    def _verify_audio(self, waveform: torch.Tensor, sample_rate: int, file_path: str = None) -> torch.Tensor:
        """Verify that audio meets XLSR requirements."""
        import logging
        # Check sample rate
        if sample_rate != self.sample_rate:
            raise ValueError(f"Sample rate must be {self.sample_rate}Hz, got {sample_rate}Hz")

        # Check that audio is mono
        if waveform.dim() > 1 and waveform.shape[0] > 1:
            logging.warning(f"Audio should be mono, got {waveform.shape[0]} channels for {file_path}. Converting to mono.")
            waveform = waveform.mean(dim=0, keepdim=True)

        # Check normalization range
        min_val, max_val = waveform.min().item(), waveform.max().item()
        if min_val < -1.01 or max_val > 1.01:
            logging.warning(f"Audio values outside range [-1, 1]: min={min_val:.4f}, max={max_val:.4f} for {file_path}")
            if waveform.abs().max() > 0:
                waveform = waveform / waveform.abs().max()

        # Log zero or constant audio
        if waveform.std() < 1e-6:
            logging.warning(f"Audio has very low variance (possibly silent or DC): std={waveform.std().item():.8f} for {file_path}")

        return waveform
    
    def __call__(
        self, 
        audio_path: str, 
        max_duration: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """Process audio file to input features."""
        try:
            # Check if file exists
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
                
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Verify audio waveform
            waveform = self._verify_audio(waveform, sample_rate, audio_path)
        except Exception as e:
            # If audio file doesn't exist or has issues, generate dummy audio for debug purposes
            import logging
            logging.warning(f"Error loading audio file {audio_path}: {str(e)}. Generating dummy audio.")
            # Generate 2 seconds of silence at the target sample rate
            sample_rate = self.sample_rate
            waveform = torch.zeros(1, 2 * sample_rate)
            
            # Add a tiny bit of noise to prevent NaN in feature extractors
            waveform = waveform + torch.randn_like(waveform) * 1e-6
        
        # Trim to max_duration if specified
        if max_duration is not None:
            max_samples = int(max_duration * self.sample_rate)
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]
        
        # Convert to numpy array
        waveform = waveform.squeeze().numpy()
        
        # Process with feature extractor
        inputs = self.feature_extractor(
            waveform, 
            sampling_rate=self.sample_rate, 
            return_tensors="pt"
        )
        
        return inputs


class XLSRTransducerProcessor:
    """Processor for XLSR-Transducer model."""
    
    def __init__(
        self,
        tokenizer_type: str = "character",
        tokenizer_dir: str = "data/tokenizer",
        tokenizer: Optional[Union[CharacterTokenizer, BPETokenizer]] = None,
        audio_preprocessor: Optional[AudioPreprocessor] = None,
        sample_rate: int = 16000,
        max_duration: Optional[float] = None,
        vocab_size: Optional[int] = None,
        special_tokens: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the XLSR-Transducer processor.
        
        Args:
            tokenizer_type: Type of tokenizer to use ("character" or "bpe")
            tokenizer_dir: Directory containing BPE tokenizer files (used if tokenizer_type="bpe")
            tokenizer: Optional pre-initialized tokenizer
            audio_preprocessor: Optional pre-initialized audio preprocessor
            sample_rate: Audio sample rate
            max_duration: Maximum duration of audio in seconds
            vocab_size: Vocabulary size for BPE tokenizer
            special_tokens: Dictionary of special tokens
        """
        # Set up tokenizer based on type if not provided
        if tokenizer is None:
            if tokenizer_type.lower() == "bpe":
                try:
                    self.tokenizer = BPETokenizer(
                        tokenizer_dir=tokenizer_dir,
                        special_tokens=special_tokens
                    )
                    print(f"Using BPE tokenizer with vocab size {self.tokenizer.vocab_size}")
                except FileNotFoundError as e:
                    print(f"Warning: {str(e)}")
                    print("Falling back to character tokenizer. Run create_bpe_tokenizer.py to create the BPE tokenizer.")
                    self.tokenizer = CharacterTokenizer(special_tokens=special_tokens)
            else:  # default to character tokenizer
                self.tokenizer = CharacterTokenizer(special_tokens=special_tokens)
                print(f"Using character tokenizer with vocab size {self.tokenizer.vocab_size}")
        else:
            self.tokenizer = tokenizer
        
        # Set up audio preprocessor
        self.audio_preprocessor = audio_preprocessor or AudioPreprocessor(sample_rate=sample_rate)
        self.max_duration = max_duration
    
    def process_audio(self, audio_path: Optional[str] = None, audio: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Process audio file or tensor."""
        if audio is not None:
            # Process audio tensor directly
            # Ensure we have a 2D tensor [channels, length]
            if audio.dim() == 1:
                # Add channel dimension if needed
                audio = audio.unsqueeze(0)
            elif audio.dim() > 2:
                # If somehow we have more dimensions, reshape to [1, length]
                audio = audio.view(1, -1)
            
            # Verify audio tensor
            audio = self.audio_preprocessor._verify_audio(audio, self.audio_preprocessor.sample_rate, "direct_tensor")
            
            # Trim to max_duration if specified
            if self.max_duration is not None:
                max_samples = int(self.max_duration * self.audio_preprocessor.sample_rate)
                if audio.shape[1] > max_samples:
                    audio = audio[:, :max_samples]
            
            # Convert to numpy array - feature extractor expects 1D array for single audio
            audio_np = audio.squeeze().numpy()
            
            # Process with feature extractor
            return self.audio_preprocessor.feature_extractor(
                audio_np, 
                sampling_rate=self.audio_preprocessor.sample_rate, 
                return_tensors="pt"
            )
        elif audio_path is not None:
            return self.audio_preprocessor(audio_path, self.max_duration)
        else:
            raise ValueError("Either audio_path or audio must be provided")
    
    def process_text(self, text: str) -> List[int]:
        """Process text to token IDs."""
        return self.tokenizer.encode(text)
    
    def __call__(
        self, 
        audio_path: Optional[str] = None,
        audio: Optional[torch.Tensor] = None,
        text: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """Process both audio and text."""
        if audio_path is None and audio is None:
            raise ValueError("Either audio_path or audio must be provided")
            
        result = self.process_audio(audio_path=audio_path, audio=audio)
        
        if text is not None:
            result["labels"] = torch.tensor(self.process_text(text))
        
        return result
    
    @property
    def vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        return self.tokenizer.vocab_size
    
    @property
    def blank_id(self) -> int:
        """Return the ID of the blank token."""
        return self.tokenizer.blank_id 