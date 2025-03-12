import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
import re


def compute_wer(predictions: List[List[int]], labels: List[List[int]], tokenizer=None) -> float:
    """
    Compute Word Error Rate (WER).
    
    Args:
        predictions: List of predicted token IDs
        labels: List of ground truth token IDs
        tokenizer: Optional tokenizer for decoding tokens to text
        
    Returns:
        Word Error Rate
    """
    if tokenizer is not None:
        # Decode token IDs to text
        pred_texts = [tokenizer.decode(pred) for pred in predictions]
        label_texts = [tokenizer.decode(label) for label in labels]
        
        # Tokenize text to words
        pred_words = [text.split() for text in pred_texts]
        label_words = [text.split() for text in label_texts]
    else:
        # Use token IDs directly
        pred_words = predictions
        label_words = labels
    
    total_errors = 0
    total_words = 0
    
    for pred, label in zip(pred_words, label_words):
        # Compute Levenshtein distance
        distance = levenshtein_distance(pred, label)
        
        # Update metrics
        total_errors += distance
        total_words += len(label)
    
    # Compute WER
    wer = total_errors / max(1, total_words)
    
    return wer


def compute_cer(predictions: List[List[int]], labels: List[List[int]], tokenizer=None) -> float:
    """
    Compute Character Error Rate (CER).
    
    Args:
        predictions: List of predicted token IDs
        labels: List of ground truth token IDs
        tokenizer: Optional tokenizer for decoding tokens to text
        
    Returns:
        Character Error Rate
    """
    if tokenizer is not None:
        # Decode token IDs to text
        pred_texts = [tokenizer.decode(pred) for pred in predictions]
        label_texts = [tokenizer.decode(label) for label in labels]
        
        # Convert text to character lists
        pred_chars = [list(text) for text in pred_texts]
        label_chars = [list(text) for text in label_texts]
    else:
        # Use token IDs directly
        pred_chars = predictions
        label_chars = labels
    
    total_errors = 0
    total_chars = 0
    
    for pred, label in zip(pred_chars, label_chars):
        # Compute Levenshtein distance
        distance = levenshtein_distance(pred, label)
        
        # Update metrics
        total_errors += distance
        total_chars += len(label)
    
    # Compute CER
    cer = total_errors / max(1, total_chars)
    
    return cer


def levenshtein_distance(seq1: List, seq2: List) -> int:
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


def compute_metrics(predictions: List[List[int]], labels: List[List[int]], tokenizer=None) -> Dict[str, float]:
    """
    Compute all metrics for ASR evaluation.
    
    Args:
        predictions: List of predicted token IDs
        labels: List of ground truth token IDs
        tokenizer: Optional tokenizer for decoding tokens to text
        
    Returns:
        Dictionary of metrics
    """
    wer = compute_wer(predictions, labels, tokenizer)
    cer = compute_cer(predictions, labels, tokenizer)
    
    return {
        "wer": wer,
        "cer": cer,
    }


def normalize_text(text: str) -> str:
    """
    Normalize text for evaluation.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation except apostrophes
    text = re.sub(r'[^\w\s\']', '', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading and trailing spaces
    text = text.strip()
    
    return text 