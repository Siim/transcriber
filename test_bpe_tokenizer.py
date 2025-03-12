#!/usr/bin/env python
# coding=utf-8
"""
Test script for the BPE tokenizer.
"""

import argparse
import logging
import os
import sys
from typing import List, Dict, Optional, Union

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the processor
from src.data.processor import BPETokenizer, CharacterTokenizer, XLSRTransducerProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def compare_tokenizers(text: str, 
                       bpe_tokenizer: BPETokenizer, 
                       char_tokenizer: CharacterTokenizer) -> None:
    """
    Compare BPE and character tokenizers on the same text.
    
    Args:
        text: Text to tokenize
        bpe_tokenizer: BPE tokenizer
        char_tokenizer: Character tokenizer
    """
    # Encode with BPE tokenizer
    bpe_tokens = bpe_tokenizer.encode(text)
    bpe_decoded = bpe_tokenizer.decode(bpe_tokens)
    
    # Encode with character tokenizer
    char_tokens = char_tokenizer.encode(text)
    char_decoded = char_tokenizer.decode(char_tokens)
    
    # Print results
    logger.info(f"Original text: {text}")
    logger.info(f"BPE tokenization: {bpe_tokens} (length: {len(bpe_tokens)})")
    logger.info(f"BPE decoded: {bpe_decoded}")
    logger.info(f"Character tokenization: {char_tokens} (length: {len(char_tokens)})")
    logger.info(f"Character decoded: {char_decoded}")
    logger.info(f"BPE vocab size: {bpe_tokenizer.vocab_size}")
    logger.info(f"Character vocab size: {char_tokenizer.vocab_size}")


def test_processor(text: str, use_bpe: bool = True) -> None:
    """
    Test the XLSR-Transducer processor with BPE tokenizer.
    
    Args:
        text: Text to process
        use_bpe: Whether to use BPE tokenizer
    """
    # Initialize processor
    tokenizer_type = "bpe" if use_bpe else "character"
    processor = XLSRTransducerProcessor(tokenizer_type=tokenizer_type)
    
    # Process text
    token_ids = processor.process_text(text)
    
    # Print results
    logger.info(f"Processor using {tokenizer_type} tokenizer")
    logger.info(f"Original text: {text}")
    logger.info(f"Token IDs: {token_ids} (length: {len(token_ids)})")
    logger.info(f"Vocab size: {processor.vocab_size}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test BPE tokenizer")
    parser.add_argument(
        "--text", 
        type=str, 
        default="See on eestikeelne lause, mille BPE tokenizer peaks tükeldama.",
        help="Text to tokenize"
    )
    parser.add_argument(
        "--compare", 
        action="store_true", 
        help="Compare BPE and character tokenizers"
    )
    
    args = parser.parse_args()
    
    # Test the tokenizers
    if args.compare:
        # Initialize tokenizers
        bpe_tokenizer = BPETokenizer(tokenizer_dir="data/tokenizer")
        char_tokenizer = CharacterTokenizer()
        
        # Compare tokenizers
        compare_tokenizers(args.text, bpe_tokenizer, char_tokenizer)
    else:
        # Test processor with BPE tokenizer
        test_processor(args.text, use_bpe=True)
        logger.info("-" * 50)
        # Test processor with character tokenizer
        test_processor(args.text, use_bpe=False)


if __name__ == "__main__":
    main() 