# Random Sample Cleaner for Large Corpus Files

This script extracts exactly 10 million random sentence pairs from a large corpus file, cleans them according to the same criteria as `cleanupData.py`, and maintains the exact count by replacing filtered sentences with new random ones.

## Features

- **Memory Efficient**: Processes large files (50GB+) using reservoir sampling without loading everything into memory
- **Random Sampling**: Uses reservoir sampling algorithm to get truly random samples from the entire file
- **Data Cleaning**: Applies the same cleaning criteria as `cleanupData.py`:
  - HTML tag removal
  - Whitespace normalization
  - Tokenization and lowercasing
  - Length filtering (min/max tokens)
  - Length ratio filtering between language pairs
  - Duplicate removal
- **Exact Count Guarantee**: Maintains exactly the target number of pairs by using a replacement reservoir
- **Progress Tracking**: Detailed logging and statistics throughout the process

## Installation

1. Install required dependencies:

```bash
pip install -r requirements.txt
```

2. Download spaCy language models:

```bash
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

## Usage

### Basic Usage

```bash
python randomSampleCleaner.py input_file.txt output_file.txt
```

### With Custom Parameters

```bash
python randomSampleCleaner.py input_file.txt output_file.txt \
    --target-count 5000000 \
    --min-length 5 \
    --max-length 150 \
    --length-ratio 2.5 \
    --seed 42
```

### Parameters

- `input_file`: Path to input corpus file (must be `|||` separated)
- `output_file`: Path to output file
- `--target-count`: Number of sentence pairs to extract (default: 10,000,000)
- `--min-length`: Minimum sentence length in tokens (default: 3)
- `--max-length`: Maximum sentence length in tokens (default: 200)
- `--length-ratio`: Maximum allowed length ratio between sentence pairs (default: 3.0)
- `--chunk-size`: Number of lines to process at once (default: 100,000)
- `--reservoir-size`: Size of replacement reservoir (default: 1,000,000)
- `--seed`: Random seed for reproducibility (optional)

## Input Format

The input file should contain sentence pairs separated by `|||`:

```
German sentence ||| English sentence
Deutscher Satz ||| English sentence
...
```

## Output Format

The output file will contain exactly the target number of cleaned sentence pairs:

```
german tokens ||| english tokens
deutsche tokens ||| englische tokens
...
```

## How It Works

1. **Line Counting**: First counts total lines in the input file
2. **Reservoir Building**: Creates a reservoir of additional clean sentence pairs for replacements
3. **Random Sampling**: Uses reservoir sampling to select random lines from the entire file
4. **Cleaning**: Applies all cleaning criteria to the sampled pairs
5. **Gap Filling**: Replaces filtered pairs with clean pairs from the replacement reservoir
6. **Output**: Writes exactly the target number of clean pairs to the output file

## Memory Usage

The script is designed to handle very large files efficiently:

- **Reservoir sampling**: Only keeps target number of lines in memory at once
- **Streaming processing**: Reads file line by line without loading everything
- **Configurable chunk size**: Adjust processing batch size based on available memory

For a 50GB file targeting 10M pairs, expect:

- Peak memory usage: ~2-3GB
- Processing time: 1-3 hours depending on hardware

## Statistics

The script provides detailed statistics:

- Total lines read from input file
- Valid pairs found during sampling
- Pairs with HTML tags cleaned
- Pairs filtered by length/ratio criteria
- Duplicate pairs removed
- Final pairs in output file
- Data retention rate

## Example

```bash
python randomSampleCleaner.py ../Corpus/large_corpus.txt training_data.txt --seed 42
```

This will:

1. Extract 10 million random sentence pairs from `large_corpus.txt`
2. Clean them according to the default criteria
3. Save exactly 10 million clean pairs to `training_data.txt`
4. Use seed 42 for reproducible results

## Troubleshooting

### Memory Issues

- Reduce `--reservoir-size` if running out of memory
- Reduce `--chunk-size` for very constrained systems

### spaCy Model Issues

- Install models manually: `python -m spacy download en_core_web_sm de_core_news_sm`
- Script will fall back to basic tokenizers if models aren't available

### Performance

- Use SSD storage for better I/O performance
- Adjust `--chunk-size` based on available RAM
- Consider using `--seed` for reproducible results during testing
