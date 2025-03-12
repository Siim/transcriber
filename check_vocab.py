#!/usr/bin/env python3
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.processor import CharacterTokenizer

# Create tokenizer
tokenizer = CharacterTokenizer()

# Print vocabulary information
print(f"Vocabulary size: {tokenizer.vocab_size}")
print(f"Special tokens: {tokenizer.special_tokens_list}")
print(f"Alphabet size: {len(tokenizer.alphabet)}")
print(f"Full vocabulary: {tokenizer.vocab}")

# Count by type
special_tokens_count = len(tokenizer.special_tokens_list)
estonian_chars_count = 26  # Basic Latin
estonian_special_chars_count = 6  # õ, ä, ö, ü, š, ž
digits_count = 10  # 0-9
punctuation_count = len(tokenizer.alphabet) - (26 + 6 + 10)  # Remaining characters

print("\nVocabulary breakdown:")
print(f"Special tokens: {special_tokens_count}")
print(f"Estonian basic chars: {estonian_chars_count}")
print(f"Estonian special chars: {estonian_special_chars_count}")
print(f"Digits: {digits_count}")
print(f"Punctuation and spaces: {punctuation_count}") 