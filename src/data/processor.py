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
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Verify audio waveform
        waveform = self._verify_audio(waveform, self.sample_rate, audio_path)
        
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
        tokenizer: Optional[CharacterTokenizer] = None,
        audio_preprocessor: Optional[AudioPreprocessor] = None,
        sample_rate: int = 16000,
        max_duration: Optional[float] = None,
    ):
        self.tokenizer = tokenizer or CharacterTokenizer()
        self.audio_preprocessor = audio_preprocessor or AudioPreprocessor(sample_rate=sample_rate)
        self.max_duration = max_duration
    
    def process_audio(self, audio_path: str) -> Dict[str, torch.Tensor]:
        """Process audio file."""
        return self.audio_preprocessor(audio_path, self.max_duration)
    
    def process_text(self, text: str) -> List[int]:
        """Process text to token IDs."""
        return self.tokenizer.encode(text)
    
    def __call__(
        self, 
        audio_path: str, 
        text: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """Process both audio and text."""
        result = self.process_audio(audio_path)
        
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