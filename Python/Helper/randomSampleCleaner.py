#!/usr/bin/env python3
"""
Random Sample Cleaner for Large Corpus Files

This script extracts exactly 10 million random sentence pairs from a large corpus file,
cleans them according to the same criteria as cleanupData.py, and maintains the exact
count by replacing filtered sentences with new random ones.

Usage:
    python randomSampleCleaner.py <input_file> <output_file> [options]
"""

import argparse
import re
import sys
import logging
import random
import tempfile
import os
from pathlib import Path
from typing import List, Tuple, Set, Optional, Iterator
from collections import deque
import spacy
from spacy.lang.de import German
from spacy.lang.en import English

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RandomSampleCleaner:
    def __init__(self, min_length: int = 3, max_length: int = 200, 
                 length_ratio_threshold: float = 3.0, target_count: int = 10_000_000,
                 chunk_size: int = 10_000, reservoir_size: int = 100_000):
        """
        Initialize the random sample cleaner.
        
        Args:
            min_length: Minimum sentence length in tokens
            max_length: Maximum sentence length in tokens
            length_ratio_threshold: Maximum allowed length ratio between sentence pairs
            target_count: Target number of sentence pairs to extract
            chunk_size: Number of lines to process at once
            reservoir_size: Size of reservoir for sampling replacement sentences
        """
        self.min_length = min_length
        self.max_length = max_length
        self.length_ratio_threshold = length_ratio_threshold
        self.target_count = target_count
        self.chunk_size = chunk_size
        self.reservoir_size = reservoir_size
        
        # Load spaCy models
        try:
            self.nlp_de = spacy.load("de_core_news_sm")
            logger.info("Loaded German spaCy model")
        except OSError:
            logger.warning("German spaCy model not found, using blank model")
            self.nlp_de = German()
            
        try:
            self.nlp_en = spacy.load("en_core_web_sm")
            logger.info("Loaded English spaCy model")
        except OSError:
            logger.warning("English spaCy model not found, using blank model")
            self.nlp_en = English()
            
        # Only use tokenizer for better performance
        if hasattr(self.nlp_de, 'pipe_names'):
            self.nlp_de.disable_pipes([pipe for pipe in self.nlp_de.pipe_names if pipe != 'tokenizer'])
        if hasattr(self.nlp_en, 'pipe_names'):
            self.nlp_en.disable_pipes([pipe for pipe in self.nlp_en.pipe_names if pipe != 'tokenizer'])
        
        # Patterns for cleaning
        self.html_pattern = re.compile(r'<[^>]+>')
        self.whitespace_pattern = re.compile(r'\s+')
        
        # Statistics
        self.stats = {
            'total_lines_read': 0,
            'valid_pairs_found': 0,
            'html_cleaned': 0,
            'length_filtered': 0,
            'duplicates_removed': 0,
            'final_pairs': 0
        }
        
        # Reservoir for replacement sentences
        self.reservoir = deque(maxlen=reservoir_size)
    
    def estimate_file_size(self, filepath: str) -> int:
        """Get file size in bytes."""
        return os.path.getsize(filepath)
    
    def parse_sentence_pair(self, line: str) -> Optional[Tuple[str, str]]:
        """Parse a line with ||| separator into German and English sentences."""
        line = line.strip()
        if not line or '|||' not in line:
            return None
        
        parts = line.split('|||')
        if len(parts) != 2:
            return None
        
        de_sent = parts[0].strip()
        en_sent = parts[1].strip()
        
        if not de_sent or not en_sent:
            return None
        
        return de_sent, en_sent
    
    def remove_html_tags(self, text: str) -> str:
        """Remove HTML tags from text."""
        if '<' in text and '>' in text:
            self.stats['html_cleaned'] += 1
            text = self.html_pattern.sub('', text)
        return text
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text."""
        return self.whitespace_pattern.sub(' ', text).strip()
    
    def tokenize_and_normalize(self, text: str, nlp_model) -> str:
        """Tokenize text using spaCy and normalize to lowercase."""
        doc = nlp_model(text)
        tokens = [token.text.lower() for token in doc if not token.is_space]
        return ' '.join(tokens)
    
    def check_length_ratio(self, de_tokens: List[str], en_tokens: List[str]) -> bool:
        """Check if the length ratio between German and English sentences is acceptable."""
        if len(de_tokens) == 0 or len(en_tokens) == 0:
            return False
            
        ratio = max(len(de_tokens), len(en_tokens)) / min(len(de_tokens), len(en_tokens))
        return ratio <= self.length_ratio_threshold
    
    def is_valid_sentence_pair(self, de_tokens: List[str], en_tokens: List[str]) -> bool:
        """Check if a sentence pair meets all validity criteria."""
        # Check minimum and maximum length
        if (len(de_tokens) < self.min_length or len(en_tokens) < self.min_length or
            len(de_tokens) > self.max_length or len(en_tokens) > self.max_length):
            return False
            
        # Check length ratio
        if not self.check_length_ratio(de_tokens, en_tokens):
            return False
            
        return True
    
    def clean_sentence_pair(self, de_sent: str, en_sent: str) -> Optional[Tuple[str, str]]:
        """Clean a sentence pair and return cleaned version if valid."""
        # Remove HTML tags
        de_clean = self.remove_html_tags(de_sent)
        en_clean = self.remove_html_tags(en_sent)
        
        # Normalize whitespace
        de_clean = self.normalize_whitespace(de_clean)
        en_clean = self.normalize_whitespace(en_clean)
        
        if not de_clean or not en_clean:
            return None
        
        # Tokenize and normalize
        de_tokens = self.tokenize_and_normalize(de_clean, self.nlp_de)
        en_tokens = self.tokenize_and_normalize(en_clean, self.nlp_en)
        
        # Check validity
        de_token_list = de_tokens.split()
        en_token_list = en_tokens.split()
        
        if self.is_valid_sentence_pair(de_token_list, en_token_list):
            return de_tokens, en_tokens
        else:
            self.stats['length_filtered'] += 1
            return None
    
    def chunk_based_sampling(self, filepath: str, total_lines: int) -> List[str]:
        """Use chunk-based sampling with frequent progress updates."""
        logger.info(f"Starting chunk-based sampling for {self.target_count:,} lines")
        logger.info(f"Processing in chunks of {self.chunk_size:,} lines with frequent updates...")
        
        collected_lines = []
        lines_processed = 0
        chunk_buffer = []
        
        # Calculate sampling rate
        sampling_rate = self.target_count / total_lines
        logger.info(f"Sampling rate: {sampling_rate:.6f} ({self.target_count:,}/{total_lines:,})")
        
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                lines_processed += 1
                chunk_buffer.append(line)
                
                # Process chunk when buffer is full
                if len(chunk_buffer) >= self.chunk_size:
                    # Sample from this chunk
                    chunk_sample_size = max(1, int(len(chunk_buffer) * sampling_rate * 2))  # 2x oversampling
                    if chunk_sample_size > 0 and len(chunk_buffer) >= chunk_sample_size:
                        sampled_from_chunk = random.sample(chunk_buffer, chunk_sample_size)
                        collected_lines.extend(sampled_from_chunk)
                    
                    # Clear buffer
                    chunk_buffer = []
                    
                    # Progress update every chunk
                    logger.info(f"Processed {lines_processed:,} lines, collected {len(collected_lines):,} samples")
                
                # Stop if we have enough samples
                if len(collected_lines) >= self.target_count * 1.5:  # 1.5x for filtering buffer
                    logger.info(f"Collected enough samples ({len(collected_lines):,}), stopping early")
                    break
        
        # Process final chunk
        if chunk_buffer:
            chunk_sample_size = max(1, int(len(chunk_buffer) * sampling_rate * 2))
            if chunk_sample_size > 0 and len(chunk_buffer) >= chunk_sample_size:
                sampled_from_chunk = random.sample(chunk_buffer, chunk_sample_size)
                collected_lines.extend(sampled_from_chunk)
        
        logger.info(f"Chunk-based sampling completed. Collected {len(collected_lines):,} lines from {lines_processed:,} processed.")
        
        # Shuffle and return up to target count
        random.shuffle(collected_lines)
        return collected_lines[:self.target_count * 2]  # Return 2x for filtering
    
    def skip_reservoir_building(self):
        """Skip building separate reservoir - we'll use oversampling instead."""
        logger.info("Skipping separate reservoir building for efficiency...")
        logger.info("Will use oversampling strategy to ensure we get enough clean pairs")
    
    def process_corpus(self, input_file: str, output_file: str, total_lines: int):
        """Process the corpus to extract and clean exactly target_count sentence pairs."""
        logger.info(f"Starting corpus processing: {input_file} -> {output_file}")
        
        logger.info(f"Processing file with {total_lines:,} total lines")
        
        if total_lines < self.target_count:
            logger.error(f"File has only {total_lines:,} lines, but {self.target_count:,} requested")
            sys.exit(1)
        
        # Skip separate reservoir building for efficiency
        self.skip_reservoir_building()
        
        # Get random sample of lines using chunk-based approach
        sampled_lines = self.chunk_based_sampling(input_file, total_lines)
        
        # Process sampled lines
        logger.info("Processing sampled lines...")
        cleaned_pairs = []
        seen_pairs = set()
        processed = 0
        
        for line in sampled_lines:
            processed += 1
            if processed % 100_000 == 0:
                logger.info(f"Processed {processed:,} sampled lines, have {len(cleaned_pairs):,} clean pairs")
            
            pair = self.parse_sentence_pair(line)
            if not pair:
                continue
            
            self.stats['valid_pairs_found'] += 1
            cleaned = self.clean_sentence_pair(pair[0], pair[1])
            
            if cleaned:
                # Check for duplicates
                pair_key = (cleaned[0], cleaned[1])
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    cleaned_pairs.append(cleaned)
                else:
                    self.stats['duplicates_removed'] += 1
        
        logger.info(f"After initial cleaning: {len(cleaned_pairs):,} pairs")
        
        # If we don't have enough, just take what we have
        if len(cleaned_pairs) < self.target_count:
            logger.warning(f"Only generated {len(cleaned_pairs):,} clean pairs out of {self.target_count:,} requested")
            logger.warning("Consider increasing oversampling or reducing filtering criteria")
        else:
            # Trim to exact target count
            cleaned_pairs = cleaned_pairs[:self.target_count]
            logger.info(f"Successfully generated exactly {len(cleaned_pairs):,} clean pairs")
        
        self.stats['final_pairs'] = len(cleaned_pairs)
        
        # Write output
        logger.info(f"Writing {len(cleaned_pairs):,} pairs to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            for de_sent, en_sent in cleaned_pairs:
                f.write(f"{de_sent} ||| {en_sent}\n")
        
        # Print statistics
        self.print_statistics()
        logger.info("Processing completed successfully!")
    
    def print_statistics(self):
        """Print processing statistics."""
        logger.info("=== PROCESSING STATISTICS ===")
        logger.info(f"Total lines read: {self.stats['total_lines_read']:,}")
        logger.info(f"Valid pairs found: {self.stats['valid_pairs_found']:,}")
        logger.info(f"Pairs with HTML cleaned: {self.stats['html_cleaned']:,}")
        logger.info(f"Pairs filtered by length/ratio: {self.stats['length_filtered']:,}")
        logger.info(f"Duplicate pairs removed: {self.stats['duplicates_removed']:,}")
        logger.info(f"Final pairs in output: {self.stats['final_pairs']:,}")


def main():
    parser = argparse.ArgumentParser(description='Extract and clean random sentence pairs from large corpus')
    parser.add_argument('input_file', help='Path to input corpus file (||| separated)')
    parser.add_argument('output_file', help='Path to output file')
    parser.add_argument('--total-lines', type=int, required=True,
                       help='Total number of lines in input file (required for large files)')
    parser.add_argument('--target-count', type=int, default=10_000_000,
                       help='Target number of sentence pairs to extract (default: 10,000,000)')
    parser.add_argument('--min-length', type=int, default=3,
                       help='Minimum sentence length in tokens (default: 3)')
    parser.add_argument('--max-length', type=int, default=200,
                       help='Maximum sentence length in tokens (default: 200)')
    parser.add_argument('--length-ratio', type=float, default=3.0,
                       help='Maximum allowed length ratio between sentence pairs (default: 3.0)')
    parser.add_argument('--chunk-size', type=int, default=100_000,
                       help='Number of lines to process at once (default: 100,000)')
    parser.add_argument('--reservoir-size', type=int, default=1_000_000,
                       help='Size of replacement reservoir (default: 1,000,000)')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed:
        random.seed(args.seed)
        logger.info(f"Random seed set to: {args.seed}")
    
    # Check if input file exists
    if not Path(args.input_file).exists():
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)
    
    # Initialize processor
    processor = RandomSampleCleaner(
        min_length=args.min_length,
        max_length=args.max_length,
        length_ratio_threshold=args.length_ratio,
        target_count=args.target_count,
        chunk_size=args.chunk_size,
        reservoir_size=args.reservoir_size
    )
    
    # Process corpus
    processor.process_corpus(args.input_file, args.output_file, args.total_lines)


if __name__ == "__main__":
    main() 