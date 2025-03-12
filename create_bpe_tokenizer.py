#!/usr/bin/env python
# coding=utf-8
"""
Script to create a BPE tokenizer from transcripts in the training data.
"""

import os
import argparse
import logging
from pathlib import Path
from typing import List, Set, Optional
from tqdm import tqdm
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def extract_transcripts(
    manifest_files: List[str], 
    output_file: Optional[str] = None
) -> List[str]:
    """
    Extract transcripts from manifest files.
    
    Args:
        manifest_files: List of manifest file paths
        output_file: Optional path to save transcripts to
        
    Returns:
        List of unique transcripts
    """
    all_transcripts = []
    unique_transcripts = set()
    
    for manifest_file in manifest_files:
        logger.info(f"Extracting transcripts from {manifest_file}")
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                for line in tqdm(f):
                    if line.strip():
                        # Format: path_to_audio|transcript|speaker_id
                        parts = line.strip().split("|")
                        if len(parts) >= 2:
                            transcript = parts[1].strip()
                            if transcript and transcript not in unique_transcripts:
                                all_transcripts.append(transcript)
                                unique_transcripts.add(transcript)
        except Exception as e:
            logger.error(f"Error extracting transcripts from {manifest_file}: {e}")
    
    logger.info(f"Extracted {len(all_transcripts)} unique transcripts")
    
    # Optionally save transcripts to file
    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for transcript in all_transcripts:
                f.write(f"{transcript}\n")
        logger.info(f"Saved transcripts to {output_file}")
    
    return all_transcripts


def train_tokenizer(
    transcripts: List[str],
    vocab_size: int = 1000,
    min_frequency: int = 2,
    output_dir: str = "data/tokenizer",
) -> Tokenizer:
    """
    Train a BPE tokenizer on the transcripts.
    
    Args:
        transcripts: List of transcripts
        vocab_size: Size of the vocabulary
        min_frequency: Minimum frequency of a token
        output_dir: Directory to save the tokenizer
        
    Returns:
        Trained tokenizer
    """
    logger.info(f"Training BPE tokenizer with vocab size {vocab_size}")
    
    # Initialize a BPE tokenizer
    tokenizer = Tokenizer(models.BPE())
    
    # Use character-level pre-tokenization
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.WhitespaceSplit(),
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.CharDelimiterSplit(" ")
    ])
    
    # Use WordPiece decoder
    tokenizer.decoder = decoders.WordPiece()
    
    # Define special tokens
    special_tokens = ["<pad>", "<s>", "</s>", "<unk>", "<mask>", "<blank>"]
    
    # Define BPE trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )
    
    # Train tokenizer
    tokenizer.train_from_iterator(transcripts, trainer=trainer)
    
    # Setup post-processor (for handling special tokens in encoding)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="$A",
        special_tokens=[
            ("<s>", tokenizer.token_to_id("<s>")),
            ("</s>", tokenizer.token_to_id("</s>")),
        ],
    )
    
    # Save tokenizer
    os.makedirs(output_dir, exist_ok=True)
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    logger.info(f"Saved tokenizer to {tokenizer_path}")
    
    # Log vocabulary info
    vocab_size = tokenizer.get_vocab_size()
    logger.info(f"Vocabulary size: {vocab_size}")
    
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="Create BPE tokenizer from transcripts")
    parser.add_argument(
        "--manifest_files", 
        nargs="+", 
        type=str, 
        default=["data/train_list.txt", "data/dev_list.txt"],
        help="List of manifest files containing transcripts"
    )
    parser.add_argument(
        "--transcripts_file", 
        type=str, 
        default="data/tokenizer/transcripts.txt",
        help="Path to save extracted transcripts"
    )
    parser.add_argument(
        "--vocab_size", 
        type=int, 
        default=1000,
        help="Vocabulary size for BPE tokenizer"
    )
    parser.add_argument(
        "--min_frequency", 
        type=int, 
        default=2,
        help="Minimum frequency of a token"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/tokenizer",
        help="Directory to save the tokenizer"
    )
    
    args = parser.parse_args()
    
    # Check if manifest files exist
    for manifest_file in args.manifest_files:
        if not os.path.exists(manifest_file):
            logger.warning(f"Manifest file {manifest_file} does not exist, skipping")
            args.manifest_files.remove(manifest_file)
    
    if not args.manifest_files:
        logger.error("No valid manifest files provided")
        return
    
    # Extract transcripts
    transcripts = extract_transcripts(
        manifest_files=args.manifest_files,
        output_file=args.transcripts_file
    )
    
    if not transcripts:
        logger.error("No transcripts found in the provided manifest files")
        return
    
    # Train tokenizer
    tokenizer = train_tokenizer(
        transcripts=transcripts,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        output_dir=args.output_dir
    )
    
    # Verify tokenizer
    test_transcript = transcripts[0]
    encoded = tokenizer.encode(test_transcript)
    logger.info(f"Test transcript: {test_transcript}")
    logger.info(f"Encoded: {encoded.tokens}")
    logger.info(f"Decoded: {tokenizer.decode(encoded.ids)}")
    
    logger.info("BPE tokenizer creation completed successfully")


if __name__ == "__main__":
    main() 