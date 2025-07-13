#!/usr/bin/env python3
"""
Data Cleaning Script for Word Alignment Corpus

This script cleans parallel corpus data according to the following criteria:
1. Segmentation and cleanup: Remove HTML tags, duplicates, and sentence pairs 
   with strong length ratio deviation
2. Tokenization: Tokenize sentences using spaCy standardized tokenizer
3. Normalization: Convert all tokens to lowercase for data variance reduction

Usage:
    python cleanupData.py <german_file> <english_file> [options]
"""

import argparse
import re
import sys
import logging
from pathlib import Path
from typing import List, Tuple, Set
from collections import Counter
import spacy
from spacy.lang.de import German
from spacy.lang.en import English

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CorpusDataCleaner:
    def __init__(self, min_length: int = 3, max_length: int = 200, 
                 length_ratio_threshold: float = 3.0):
        """
        Initialize the corpus data cleaner.
        
        Args:
            min_length: Minimum sentence length in tokens
            max_length: Maximum sentence length in tokens
            length_ratio_threshold: Maximum allowed length ratio between sentence pairs
        """
        self.min_length = min_length
        self.max_length = max_length
        self.length_ratio_threshold = length_ratio_threshold
        
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
        self.nlp_de.disable_pipes([pipe for pipe in self.nlp_de.pipe_names if pipe != 'tokenizer'])
        self.nlp_en.disable_pipes([pipe for pipe in self.nlp_en.pipe_names if pipe != 'tokenizer'])
        
        # HTML tag removal pattern
        self.html_pattern = re.compile(r'<[^>]+>')
        
        # Multiple whitespace pattern
        self.whitespace_pattern = re.compile(r'\s+')
        
        # Statistics
        self.stats = {
            'total_pairs': 0,
            'html_cleaned': 0,
            'length_filtered': 0,
            'duplicates_removed': 0,
            'final_pairs': 0
        }
    
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
        """
        Tokenize text using spaCy and normalize to lowercase.
        
        Args:
            text: Input text to tokenize
            nlp_model: spaCy model to use for tokenization
            
        Returns:
            Normalized tokenized text
        """
        doc = nlp_model(text)
        tokens = [token.text.lower() for token in doc if not token.is_space]
        return ' '.join(tokens)
    
    def check_length_ratio(self, de_tokens: List[str], en_tokens: List[str]) -> bool:
        """
        Check if the length ratio between German and English sentences is acceptable.
        
        Args:
            de_tokens: German sentence tokens
            en_tokens: English sentence tokens
            
        Returns:
            True if length ratio is acceptable, False otherwise
        """
        if len(de_tokens) == 0 or len(en_tokens) == 0:
            return False
            
        ratio = max(len(de_tokens), len(en_tokens)) / min(len(de_tokens), len(en_tokens))
        return ratio <= self.length_ratio_threshold
    
    def is_valid_sentence_pair(self, de_tokens: List[str], en_tokens: List[str]) -> bool:
        """
        Check if a sentence pair meets all validity criteria.
        
        Args:
            de_tokens: German sentence tokens
            en_tokens: English sentence tokens
            
        Returns:
            True if sentence pair is valid, False otherwise
        """
        # Check minimum and maximum length
        if (len(de_tokens) < self.min_length or len(en_tokens) < self.min_length or
            len(de_tokens) > self.max_length or len(en_tokens) > self.max_length):
            return False
            
        # Check length ratio
        if not self.check_length_ratio(de_tokens, en_tokens):
            return False
            
        return True
    
    def load_sentences(self, filepath: str) -> List[str]:
        """Load sentences from file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                sentences = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(sentences)} sentences from {filepath}")
            return sentences
        except Exception as e:
            logger.error(f"Error loading file {filepath}: {e}")
            sys.exit(1)
    
    def save_sentences(self, sentences: List[str], filepath: str):
        """Save sentences to file."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                for sentence in sentences:
                    f.write(sentence + '\n')
            logger.info(f"Saved {len(sentences)} sentences to {filepath}")
        except Exception as e:
            logger.error(f"Error saving file {filepath}: {e}")
            sys.exit(1)
    
    def clean_corpus(self, de_file: str, en_file: str, output_prefix: str = None):
        """
        Clean the parallel corpus according to specified criteria.
        
        Args:
            de_file: Path to German sentences file
            en_file: Path to English sentences file
            output_prefix: Prefix for output filenames (optional)
        """
        logger.info("Starting corpus cleaning process...")
        
        # Load sentences
        de_sentences = self.load_sentences(de_file)
        en_sentences = self.load_sentences(en_file)
        
        if len(de_sentences) != len(en_sentences):
            logger.error(f"File length mismatch: German={len(de_sentences)}, English={len(en_sentences)}")
            sys.exit(1)
        
        self.stats['total_pairs'] = len(de_sentences)
        logger.info(f"Processing {self.stats['total_pairs']} sentence pairs...")
        
        # Step 1: HTML tag removal and basic cleaning
        logger.info("Step 1: Removing HTML tags and normalizing whitespace...")
        cleaned_de = []
        cleaned_en = []
        
        for de_sent, en_sent in zip(de_sentences, en_sentences):
            # Remove HTML tags
            de_clean = self.remove_html_tags(de_sent)
            en_clean = self.remove_html_tags(en_sent)
            
            # Normalize whitespace
            de_clean = self.normalize_whitespace(de_clean)
            en_clean = self.normalize_whitespace(en_clean)
            
            if de_clean and en_clean:  # Only keep non-empty sentences
                cleaned_de.append(de_clean)
                cleaned_en.append(en_clean)
        
        logger.info(f"After HTML cleaning: {len(cleaned_de)} pairs remaining")
        
        # Step 2: Tokenization and normalization
        logger.info("Step 2: Tokenizing and normalizing sentences...")
        tokenized_de = []
        tokenized_en = []
        
        for i, (de_sent, en_sent) in enumerate(zip(cleaned_de, cleaned_en)):
            if i % 5000 == 0:
                logger.info(f"Processed {i} sentences...")
                
            # Tokenize and normalize
            de_tokens = self.tokenize_and_normalize(de_sent, self.nlp_de)
            en_tokens = self.tokenize_and_normalize(en_sent, self.nlp_en)
            
            tokenized_de.append(de_tokens)
            tokenized_en.append(en_tokens)
        
        # Step 3: Filter by length and length ratio
        logger.info("Step 3: Filtering by length and length ratio...")
        filtered_de = []
        filtered_en = []
        
        for de_tokens, en_tokens in zip(tokenized_de, tokenized_en):
            de_token_list = de_tokens.split()
            en_token_list = en_tokens.split()
            
            if self.is_valid_sentence_pair(de_token_list, en_token_list):
                filtered_de.append(de_tokens)
                filtered_en.append(en_tokens)
            else:
                self.stats['length_filtered'] += 1
        
        logger.info(f"After length filtering: {len(filtered_de)} pairs remaining")
        
        # Step 4: Remove duplicates
        logger.info("Step 4: Removing duplicate sentence pairs...")
        seen_pairs = set()
        final_de = []
        final_en = []
        
        for de_sent, en_sent in zip(filtered_de, filtered_en):
            pair_key = (de_sent, en_sent)
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                final_de.append(de_sent)
                final_en.append(en_sent)
            else:
                self.stats['duplicates_removed'] += 1
        
        self.stats['final_pairs'] = len(final_de)
        
        # Generate output filenames
        if output_prefix is None:
            de_path = Path(de_file)
            output_prefix = de_path.parent / f"{de_path.stem}_cleaned"
        
        de_output = f"{output_prefix}.de"
        en_output = f"{output_prefix}.en"
        
        # Save cleaned data
        self.save_sentences(final_de, de_output)
        self.save_sentences(final_en, en_output)
        
        # Print statistics
        self.print_statistics()
        
        logger.info("Corpus cleaning completed successfully!")
        logger.info(f"Cleaned files saved as: {de_output} and {en_output}")
    
    def print_statistics(self):
        """Print cleaning statistics."""
        logger.info("=== CLEANING STATISTICS ===")
        logger.info(f"Total sentence pairs processed: {self.stats['total_pairs']}")
        logger.info(f"Pairs with HTML tags cleaned: {self.stats['html_cleaned']}")
        logger.info(f"Pairs filtered by length/ratio: {self.stats['length_filtered']}")
        logger.info(f"Duplicate pairs removed: {self.stats['duplicates_removed']}")
        logger.info(f"Final cleaned pairs: {self.stats['final_pairs']}")
        logger.info(f"Data retention rate: {self.stats['final_pairs']/self.stats['total_pairs']*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Clean parallel corpus data for word alignment')
    parser.add_argument('de_file', help='Path to German sentences file')
    parser.add_argument('en_file', help='Path to English sentences file')
    parser.add_argument('--output-prefix', help='Prefix for output filenames')
    parser.add_argument('--min-length', type=int, default=3, 
                       help='Minimum sentence length in tokens (default: 3)')
    parser.add_argument('--max-length', type=int, default=400,
                       help='Maximum sentence length in tokens (default: 200)')
    parser.add_argument('--length-ratio', type=float, default=3.0,
                       help='Maximum allowed length ratio between sentence pairs (default: 3.0)')
    
    args = parser.parse_args()
    
    # Check if input files exist
    if not Path(args.de_file).exists():
        logger.error(f"German file not found: {args.de_file}")
        sys.exit(1)
    
    if not Path(args.en_file).exists():
        logger.error(f"English file not found: {args.en_file}")
        sys.exit(1)
    
    # Initialize cleaner
    cleaner = CorpusDataCleaner(
        min_length=args.min_length,
        max_length=args.max_length,
        length_ratio_threshold=args.length_ratio
    )
    
    # Clean corpus
    cleaner.clean_corpus(args.de_file, args.en_file, args.output_prefix)


if __name__ == "__main__":
    main()
