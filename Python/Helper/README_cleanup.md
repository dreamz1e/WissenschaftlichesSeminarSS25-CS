# Corpus Data Cleaning Script

This script cleans parallel corpus data for word alignment according to the following criteria:

## Features

1. **Segmentation and Cleanup**: Removes HTML tags, duplicates, and sentence pairs with strong length ratio deviation
2. **Tokenization**: Uses spaCy standardized tokenizer to break sentences into words
3. **Normalization**: Converts all tokens to lowercase to reduce data variance

## Installation

1. Install required dependencies:

```bash
pip install -r requirements.txt
```

2. Install spaCy language models (optional, script will work with basic tokenizers if models are not available):

```bash
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

## Usage

### Basic Usage

```bash
python cleanupData.py path/to/german_file.de path/to/english_file.en
```

### With Custom Parameters

```bash
python cleanupData.py \
  ../../Corpus/TEST_DATA/TEST_DATA-de-en.de \
  ../../Corpus/TEST_DATA/TEST_DATA-de-en.en \
  --output-prefix cleaned_test_data \
  --min-length 3 \
  --max-length 100 \
  --length-ratio 2.5
```

### Parameters

- `de_file`: Path to German sentences file
- `en_file`: Path to English sentences file
- `--output-prefix`: Prefix for output filenames (optional)
- `--min-length`: Minimum sentence length in tokens (default: 3)
- `--max-length`: Maximum sentence length in tokens (default: 200)
- `--length-ratio`: Maximum allowed length ratio between sentence pairs (default: 3.0)

## Output

The script creates two cleaned files:

- `{output_prefix}.de` - Cleaned German sentences
- `{output_prefix}.en` - Cleaned English sentences

## Cleaning Process

1. **HTML Tag Removal**: Removes any HTML tags found in the text
2. **Whitespace Normalization**: Normalizes multiple whitespaces to single spaces
3. **Tokenization**: Uses spaCy to tokenize sentences into individual words
4. **Normalization**: Converts all tokens to lowercase
5. **Length Filtering**: Removes sentence pairs that are too short, too long, or have mismatched lengths
6. **Duplicate Removal**: Removes duplicate sentence pairs

## Statistics

The script provides detailed statistics about the cleaning process:

- Total sentence pairs processed
- Number of pairs with HTML tags cleaned
- Number of pairs filtered by length/ratio criteria
- Number of duplicate pairs removed
- Final number of cleaned pairs
- Data retention rate

## Example

For the TEST_DATA files in your corpus:

```bash
python cleanupData.py ../../Corpus/TEST_DATA/TEST_DATA-de-en.de ../../Corpus/TEST_DATA/TEST_DATA-de-en.en
```

This will create:

- `TEST_DATA-de-en_cleaned.de`
- `TEST_DATA-de-en_cleaned.en`
