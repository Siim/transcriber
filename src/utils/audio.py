import torch
import torchaudio
import numpy as np
import librosa
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union


def load_audio(
    file_path: str,
    target_sample_rate: int = 16000,
    max_duration: Optional[float] = None,
) -> Tuple[torch.Tensor, int]:
    """
    Load audio file and resample if needed.
    
    Args:
        file_path: Path to audio file
        target_sample_rate: Target sample rate
        max_duration: Maximum duration in seconds
        
    Returns:
        Tuple of (waveform, sample_rate)
    """
    # Load audio
    waveform, sample_rate = torchaudio.load(file_path)
    
    # Convert to mono if needed
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if needed
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_sample_rate
        )
        waveform = resampler(waveform)
        sample_rate = target_sample_rate
    
    # Trim to max_duration if specified
    if max_duration is not None:
        max_samples = int(max_duration * sample_rate)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]
    
    return waveform, sample_rate


def save_audio(
    waveform: torch.Tensor,
    file_path: str,
    sample_rate: int = 16000,
) -> None:
    """
    Save audio to file.
    
    Args:
        waveform: Audio waveform
        file_path: Path to save audio file
        sample_rate: Sample rate
    """
    torchaudio.save(file_path, waveform, sample_rate)


def plot_waveform(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    title: str = "Waveform",
    ax=None,
) -> plt.Axes:
    """
    Plot audio waveform.
    
    Args:
        waveform: Audio waveform
        sample_rate: Sample rate
        title: Plot title
        ax: Optional matplotlib axes
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(12, 3))
    
    waveform = waveform.numpy()
    
    if waveform.ndim > 1:
        waveform = waveform[0]
    
    num_frames = waveform.shape[0]
    time_axis = torch.arange(0, num_frames) / sample_rate
    
    ax.plot(time_axis, waveform, linewidth=1)
    ax.grid(True)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    
    return ax


def plot_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    title: str = "Spectrogram",
    ax=None,
) -> plt.Axes:
    """
    Plot audio spectrogram.
    
    Args:
        waveform: Audio waveform
        sample_rate: Sample rate
        title: Plot title
        ax: Optional matplotlib axes
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(12, 3))
    
    waveform = waveform.numpy()
    
    if waveform.ndim > 1:
        waveform = waveform[0]
    
    # Compute spectrogram
    spectrogram = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=80,
    )
    
    # Convert to dB
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    
    # Plot spectrogram
    librosa.display.specshow(
        spectrogram,
        sr=sample_rate,
        hop_length=512,
        x_axis="time",
        y_axis="mel",
        ax=ax,
    )
    
    ax.set_title(title)
    
    return ax


def compute_fbank(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> torch.Tensor:
    """
    Compute log Mel filterbank features.
    
    Args:
        waveform: Audio waveform
        sample_rate: Sample rate
        n_mels: Number of Mel bands
        n_fft: FFT size
        hop_length: Hop length
        
    Returns:
        Log Mel filterbank features
    """
    # Convert to mono if needed
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Compute Mel spectrogram
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )(waveform)
    
    # Convert to dB
    log_mel_spectrogram = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
    
    return log_mel_spectrogram


def vad_split(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    threshold: float = -40.0,
    min_silence_duration: float = 0.3,
    min_speech_duration: float = 0.1,
) -> List[torch.Tensor]:
    """
    Split audio by voice activity detection.
    
    Args:
        waveform: Audio waveform
        sample_rate: Sample rate
        threshold: Energy threshold in dB
        min_silence_duration: Minimum silence duration in seconds
        min_speech_duration: Minimum speech duration in seconds
        
    Returns:
        List of speech segments
    """
    # Convert to mono if needed
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Convert to numpy
    waveform_np = waveform.squeeze().numpy()
    
    # Compute energy
    energy = librosa.feature.rms(y=waveform_np, frame_length=1024, hop_length=512)[0]
    energy_db = 20 * np.log10(energy + 1e-10)
    
    # Find speech segments
    speech_mask = energy_db > threshold
    
    # Convert to frames
    min_silence_frames = int(min_silence_duration * sample_rate / 512)
    min_speech_frames = int(min_speech_duration * sample_rate / 512)
    
    # Smooth mask
    speech_mask = np.convolve(speech_mask, np.ones(min_silence_frames) / min_silence_frames, mode="same") > 0.5
    
    # Find speech segments
    speech_segments = []
    in_speech = False
    start_frame = 0
    
    for i, is_speech in enumerate(speech_mask):
        if is_speech and not in_speech:
            # Start of speech
            start_frame = i
            in_speech = True
        elif not is_speech and in_speech:
            # End of speech
            end_frame = i
            
            # Check if segment is long enough
            if end_frame - start_frame >= min_speech_frames:
                # Convert frames to samples
                start_sample = start_frame * 512
                end_sample = end_frame * 512
                
                # Extract segment
                segment = waveform[:, start_sample:end_sample]
                speech_segments.append(segment)
            
            in_speech = False
    
    # Add last segment if needed
    if in_speech:
        end_frame = len(speech_mask)
        
        # Check if segment is long enough
        if end_frame - start_frame >= min_speech_frames:
            # Convert frames to samples
            start_sample = start_frame * 512
            end_sample = end_frame * 512
            
            # Extract segment
            segment = waveform[:, start_sample:end_sample]
            speech_segments.append(segment)
    
    return speech_segments 