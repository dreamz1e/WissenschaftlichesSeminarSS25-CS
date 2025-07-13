#!/usr/bin/env python3
"""
Merge ParaCrawl German and English files into a single file with ||| separator.
Equivalent to: paste -d '|' train.de train.en | sed 's/|/ ||| /' > train.de-en.txt
"""

import os
import sys
from pathlib import Path

def merge_paracrawl_files(de_file_path, en_file_path, output_path):
    """
    Merge German and English files line by line with ' ||| ' separator.
    
    Args:
        de_file_path (str): Path to German file
        en_file_path (str): Path to English file  
        output_path (str): Path to output merged file
    """
    
    # Check if input files exist
    if not os.path.exists(de_file_path):
        print(f"Error: German file not found: {de_file_path}")
        return False
        
    if not os.path.exists(en_file_path):
        print(f"Error: English file not found: {en_file_path}")
        return False
    
    print(f"Merging files:")
    print(f"  German: {de_file_path}")
    print(f"  English: {en_file_path}")
    print(f"  Output: {output_path}")
    
    line_count = 0
    
    try:
        with open(de_file_path, 'r', encoding='utf-8') as de_file, \
             open(en_file_path, 'r', encoding='utf-8') as en_file, \
             open(output_path, 'w', encoding='utf-8') as output_file:
            
            while True:
                de_line = de_file.readline()
                en_line = en_file.readline()
                
                # If either file reaches EOF, stop
                if not de_line or not en_line:
                    # Check if files have different lengths
                    if de_line or en_line:
                        print(f"Warning: Files have different lengths. Stopped at line {line_count + 1}")
                    break
                
                # Strip whitespace and combine with ||| separator
                de_line = de_line.strip()
                en_line = en_line.strip()
                
                # Skip empty lines
                if not de_line or not en_line:
                    continue
                
                merged_line = f"{de_line} ||| {en_line}\n"
                output_file.write(merged_line)
                
                line_count += 1
                
                # Progress indicator for large files
                if line_count % 100000 == 0:
                    print(f"Processed {line_count:,} lines...")
    
    except Exception as e:
        print(f"Error during processing: {e}")
        return False
    
    print(f"Successfully merged {line_count:,} lines to {output_path}")
    return True

def main():
    # Set up file paths relative to the script location
    script_dir = Path(__file__).parent
    corpus_dir = script_dir.parent / "Corpus" / "ParaCrawl"
    
    de_file = corpus_dir / "ParaCrawl.de-en.de" 
    en_file = corpus_dir / "ParaCrawl.de-en.en"
    output_file = corpus_dir / "ParaCrawl.de-en.txt"
    
    print("ParaCrawl File Merger")
    print("====================")
    
    # Allow command line arguments to override default paths
    if len(sys.argv) >= 4:
        de_file = Path(sys.argv[1])
        en_file = Path(sys.argv[2])
        output_file = Path(sys.argv[3])
    elif len(sys.argv) > 1:
        print("Usage: python merge_paracrawl.py [de_file] [en_file] [output_file]")
        print("       python merge_paracrawl.py  (uses default ParaCrawl paths)")
        sys.exit(1)
    
    success = merge_paracrawl_files(str(de_file), str(en_file), str(output_file))
    
    if success:
        print("\nMerge completed successfully!")
        print(f"Output file: {output_file}")
    else:
        print("\nMerge failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 