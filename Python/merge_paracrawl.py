#!/usr/bin/env python3
"""
Merge ParaCrawl German and English files into a single file with ||| separator.
Equivalent to: paste -d '|' train.de train.en | sed 's/|/ ||| /' > train.de-en.txt
"""

import os
import sys
import time
from pathlib import Path

def merge_paracrawl_files_simple(de_file_path, en_file_path, output_path, buffer_lines=1000):
    """
    Simple and reliable approach: read a reasonable number of lines at once.
    Optimized for very large files.
    
    Args:
        de_file_path (str): Path to German file
        en_file_path (str): Path to English file  
        output_path (str): Path to output merged file
        buffer_lines (int): Number of lines to process at once (smaller for stability)
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
    print(f"  Buffer size: {buffer_lines:,} lines")
    print("Starting merge process...")
    
    line_count = 0
    start_time = time.time()
    last_progress_time = start_time
    
    try:
        # Use larger file buffers but smaller line buffers
        with open(de_file_path, 'r', encoding='utf-8', buffering=1024*1024) as de_file, \
             open(en_file_path, 'r', encoding='utf-8', buffering=1024*1024) as en_file, \
             open(output_path, 'w', encoding='utf-8', buffering=1024*1024) as output_file:
            
            batch_count = 0
            
            while True:
                batch_count += 1
                current_time = time.time()
                
                # Show progress every few seconds or every batch for the first few batches
                if batch_count <= 10 or (current_time - last_progress_time) >= 5.0:
                    elapsed = current_time - start_time
                    print(f"Batch {batch_count}: Processing lines {line_count+1:,} to {line_count+buffer_lines:,} (Elapsed: {elapsed:.1f}s)")
                    last_progress_time = current_time
                
                # Read lines from both files
                de_lines = []
                en_lines = []
                
                # Read buffer_lines from each file
                for _ in range(buffer_lines):
                    de_line = de_file.readline()
                    en_line = en_file.readline()
                    
                    if not de_line or not en_line:
                        # Check for file length mismatch
                        if de_line or en_line:
                            print(f"Warning: Files have different lengths at line {line_count + len(de_lines) + 1}")
                        break
                        
                    de_lines.append(de_line.strip())
                    en_lines.append(en_line.strip())
                
                # If no lines were read, we're done
                if not de_lines:
                    break
                
                # Process and write this batch
                output_batch = []
                for de_line, en_line in zip(de_lines, en_lines):
                    if de_line and en_line:  # Skip empty lines
                        output_batch.append(f"{de_line} ||| {en_line}\n")
                
                # Write the batch
                if output_batch:
                    output_file.writelines(output_batch)
                    line_count += len(output_batch)
                
                # If we read fewer lines than requested, we're at EOF
                if len(de_lines) < buffer_lines:
                    break
    
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\nMerge completed!")
    print(f"Total lines processed: {line_count:,}")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Average speed: {line_count/total_time:.0f} lines/second")
    print(f"Output written to: {output_path}")
    
    return True

def merge_paracrawl_files(de_file_path, en_file_path, output_path, chunk_size=1024*1024):
    """
    Merge German and English files line by line with ' ||| ' separator using chunk processing.
    
    Args:
        de_file_path (str): Path to German file
        en_file_path (str): Path to English file  
        output_path (str): Path to output merged file
        chunk_size (int): Size of chunks to read in bytes (default 1MB)
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
    print(f"  Chunk size: {chunk_size:,} bytes")
    
    line_count = 0
    
    try:
        with open(de_file_path, 'r', encoding='utf-8', buffering=chunk_size) as de_file, \
             open(en_file_path, 'r', encoding='utf-8', buffering=chunk_size) as en_file, \
             open(output_path, 'w', encoding='utf-8', buffering=chunk_size) as output_file:
            
            # Buffer for incomplete lines
            de_buffer = ""
            en_buffer = ""
            
            while True:
                # Read chunks from both files
                de_chunk = de_file.read(chunk_size)
                en_chunk = en_file.read(chunk_size)
                
                # If both chunks are empty, we're done
                if not de_chunk and not en_chunk:
                    break
                
                # Add chunks to buffers
                de_buffer += de_chunk
                en_buffer += en_chunk
                
                # Process complete lines from buffers
                de_lines = de_buffer.split('\n')
                en_lines = en_buffer.split('\n')
                
                # Keep the last incomplete line in buffer
                de_buffer = de_lines[-1] if de_chunk else ""
                en_buffer = en_lines[-1] if en_chunk else ""
                
                # Remove the incomplete line from processing
                if de_chunk:
                    de_lines = de_lines[:-1]
                if en_chunk:
                    en_lines = en_lines[:-1]
                
                # Process pairs of lines
                min_lines = min(len(de_lines), len(en_lines))
                
                for i in range(min_lines):
                    de_line = de_lines[i].strip()
                    en_line = en_lines[i].strip()
                    
                    # Skip empty lines
                    if not de_line or not en_line:
                        continue
                    
                    merged_line = f"{de_line} ||| {en_line}\n"
                    output_file.write(merged_line)
                    
                    line_count += 1
                    
                    # Progress indicator
                    if line_count % 100000 == 0:
                        print(f"Processed {line_count:,} lines...")
                
                # Handle remaining lines (file length mismatch)
                if len(de_lines) != len(en_lines):
                    print(f"Warning: Chunk contains different number of lines. DE: {len(de_lines)}, EN: {len(en_lines)}")
            
            # Process any remaining buffered lines
            if de_buffer.strip() and en_buffer.strip():
                de_line = de_buffer.strip()
                en_line = en_buffer.strip()
                if de_line and en_line:
                    merged_line = f"{de_line} ||| {en_line}\n"
                    output_file.write(merged_line)
                    line_count += 1
    
    except Exception as e:
        print(f"Error during processing: {e}")
        return False
    
    print(f"Successfully merged {line_count:,} lines to {output_path}")
    return True

def merge_paracrawl_files_optimized(de_file_path, en_file_path, output_path, buffer_lines=10000):
    """
    Alternative optimized approach: read multiple lines at once.
    
    Args:
        de_file_path (str): Path to German file
        en_file_path (str): Path to English file  
        output_path (str): Path to output merged file
        buffer_lines (int): Number of lines to process at once
    """
    
    # Check if input files exist
    if not os.path.exists(de_file_path):
        print(f"Error: German file not found: {de_file_path}")
        return False
        
    if not os.path.exists(en_file_path):
        print(f"Error: English file not found: {en_file_path}")
        return False
    
    print(f"Merging files (optimized approach):")
    print(f"  German: {de_file_path}")
    print(f"  English: {en_file_path}")
    print(f"  Output: {output_path}")
    print(f"  Buffer size: {buffer_lines:,} lines")
    
    line_count = 0
    
    try:
        with open(de_file_path, 'r', encoding='utf-8', buffering=8192*1024) as de_file, \
             open(en_file_path, 'r', encoding='utf-8', buffering=8192*1024) as en_file, \
             open(output_path, 'w', encoding='utf-8', buffering=8192*1024) as output_file:
            
            while True:
                # Read multiple lines at once
                de_lines = []
                en_lines = []
                
                # Read buffer_lines from each file
                for _ in range(buffer_lines):
                    de_line = de_file.readline()
                    en_line = en_file.readline()
                    
                    if not de_line or not en_line:
                        break
                        
                    de_lines.append(de_line.strip())
                    en_lines.append(en_line.strip())
                
                # If no lines were read, we're done
                if not de_lines:
                    break
                
                # Process all lines in this batch
                output_lines = []
                for de_line, en_line in zip(de_lines, en_lines):
                    if de_line and en_line:  # Skip empty lines
                        output_lines.append(f"{de_line} ||| {en_line}\n")
                
                # Write all lines at once
                output_file.writelines(output_lines)
                line_count += len(output_lines)
                
                # Progress indicator
                if line_count % 100000 == 0:
                    print(f"Processed {line_count:,} lines...")
                
                # If we read fewer lines than requested, we're at EOF
                if len(de_lines) < buffer_lines:
                    break
    
    except Exception as e:
        print(f"Error during processing: {e}")
        return False
    
    print(f"Successfully merged {line_count:,} lines to {output_path}")
    return True

def main():
    # Set up file paths - navigate from Python directory to Corpus directory
    script_dir = Path(__file__).parent.absolute()
    
    # Find the workspace root (go up until we find "Corpus" directory)
    workspace_root = script_dir
    while workspace_root.name != "WissenschaftlichesSeminar" and workspace_root.parent != workspace_root:
        workspace_root = workspace_root.parent
    
    corpus_dir = workspace_root / "Corpus" / "GOLD_MANUAL"
    
    de_file = corpus_dir / "gold.de"
    en_file = corpus_dir / "gold.en"
    output_file = corpus_dir / "gold.terminology.txt"
    
    print("ParaCrawl File Merger (Memory Optimized)")
    print("=======================================")
    print(f"Looking for files in: {corpus_dir}")
    
    # Allow command line arguments to override default paths
    if len(sys.argv) >= 4:
        de_file = Path(sys.argv[1])
        en_file = Path(sys.argv[2])
        output_file = Path(sys.argv[3])
    elif len(sys.argv) > 1:
        print("Usage: python merge_paracrawl.py [de_file] [en_file] [output_file]")
        print("       python merge_paracrawl.py  (uses default ParaCrawl paths)")
        sys.exit(1)
    
    # Use the simple approach - most reliable for very large files
    # Uses smaller buffer size but with frequent progress updates
    success = merge_paracrawl_files_simple(str(de_file), str(en_file), str(output_file), buffer_lines=5000)
    
    if success:
        print("\nMerge completed successfully!")
        print(f"Output file: {output_file}")
    else:
        print("\nMerge failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 