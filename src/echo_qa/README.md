# EchoQA

Medical Image Caption Processing and Question Answering using OpenAI's GPT models.

## Overview

EchoQA is a Python package designed to process medical image captions and generate structured Q&A data using OpenAI's GPT models. It's specifically tailored for medical image interpretation tasks, providing automated analysis of imaging modality, anatomical regions, abnormalities, and other medical insights.

## Features

- **Medical Image Caption Processing**: Extract and analyze medical image captions
- **Structured Q&A Generation**: Generate answers to standardized medical questions
- **Batch Processing**: Process large datasets efficiently with configurable batch sizes
- **Error Handling**: Robust error handling and logging
- **Flexible Configuration**: Easy configuration management for different datasets
- **CLI Interface**: Command-line interface for easy integration

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd EchoPrime/src/echo_qa
```

2. Install dependencies:
```bash
# Dependencies are managed in the main pyproject.toml
# Install the entire EchoVLM project
pip install -e .
```

3. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Command Line Interface

The main entry point is the CLI interface:

```bash
python -m echo_qa.cli --data_str train --batch_size 100
```

#### Basic Usage

```bash
# Process training data
python -m echo_qa.cli --data_str train --batch_size 100

# Process validation data with custom settings
python -m echo_qa.cli --data_str val --batch_size 50 --start_idx 1000

# Process test data with limited entries
python -m echo_qa.cli --data_str test --process_all False --num_entries_to_process 10
```

#### Advanced Usage

```bash
# Custom input/output directories
python -m echo_qa.cli --data_str train --input_dir /path/to/data --output_dir /path/to/output

# Custom OpenAI model and settings
python -m echo_qa.cli --data_str train --openai_model gpt-4-turbo --max_tokens 500

# Verbose logging
python -m echo_qa.cli --data_str train --log_level DEBUG --log_file processing.log
```

### Programmatic Usage

```python
from echo_qa import CaptionProcessor, OpenAIClient, Config

# Initialize components
api_client = OpenAIClient(api_key="your-api-key")
processor = CaptionProcessor(api_client, input_dir="data", output_dir="output")

# Process dataset
output_file = processor.process_dataset(
    data_str="train",
    batch_size=100,
    start_idx=0,
    process_all=True
)
```

## Configuration

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data_str` | str | - | Dataset type (train/val/test) |
| `--input_dir` | str | "data" | Input directory |
| `--output_dir` | str | "output_batches" | Output directory |
| `--batch_size` | int | 100 | Entries per batch |
| `--start_idx` | int | 0 | Starting index |
| `--process_all` | bool | True | Process all entries |
| `--num_entries_to_process` | int | 10 | Number of entries if not processing all |
| `--openai_model` | str | "gpt-4" | OpenAI model to use |
| `--max_tokens` | int | 300 | Max tokens for response |
| `--log_level` | str | "INFO" | Logging level |
| `--log_file` | str | None | Log file path |

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)

## Data Format

### Input Format

The package expects JSON files with the following structure:

```json
[
  {
    "conversations": [
      {
        "value": "Medical image caption text here..."
      }
    ],
    "id": "unique_identifier",
    "other_fields": "..."
  }
]
```

### Output Format

Processed data is saved with the following structure:

```json
[
  {
    "id": "unique_identifier",
    "caption": "Medical image caption text here...",
    "answers": [
      "A1: Answer to question 1",
      "A2: Answer to question 2",
      "A3: Answer to question 3",
      "A4: Answer to question 4",
      "A5: Answer to question 5"
    ],
    "other_fields": "..."
  }
]
```

## Medical Questions

The system automatically asks the following medical questions for each caption:

1. **Q1**: What imaging modality is represented in this image?
2. **Q2**: What body region or anatomical area does this image depict?
3. **Q3**: Are there any abnormalities identified in this image?
4. **Q4**: Does this image appear normal, or does it show any irregularities?
5. **Q5**: Does this image contain any label or index that is significant or noteworthy?

## Project Structure

```
echo_qa/
├── __init__.py              # Package initialization
├── cli.py                   # Command-line interface
├── core/                    # Core functionality
│   ├── __init__.py
│   ├── api_client.py        # OpenAI API client
│   └── processor.py         # Main processing logic
├── utils/                   # Utilities
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── io.py               # I/O utilities
│   └── logging.py          # Logging configuration
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Batch Processing

The package processes data in batches to handle large datasets efficiently:

1. **Batch Creation**: Data is split into configurable batch sizes
2. **Parallel Processing**: Each batch is processed independently
3. **Intermediate Storage**: Batch results are saved to temporary files
4. **Final Combination**: All batches are combined into a single output file

## Error Handling

- **API Errors**: Handles OpenAI API rate limits and connection issues
- **File Errors**: Graceful handling of missing or corrupted files
- **Data Errors**: Continues processing even if individual entries fail
- **Logging**: Comprehensive logging for debugging and monitoring

## Logging

The package provides configurable logging with multiple levels:

- **DEBUG**: Detailed processing information
- **INFO**: General processing status
- **WARNING**: Non-critical issues
- **ERROR**: Processing errors
- **CRITICAL**: Critical system errors

## Examples

### SLURM Job Scripts

The package includes SLURM job scripts for cluster processing:

```bash
# Submit training job
sbatch submit_train.sh

# Submit validation job
sbatch submit_val.sh

# Submit test job
sbatch submit_test.sh
```

### Custom Processing

```python
from echo_qa.core.api_client import OpenAIClient

# Custom API client
client = OpenAIClient(
    api_key="your-key",
    model="gpt-4-turbo",
    max_tokens=500
)

# Process single caption
answers = client.process_caption("Echocardiogram showing left ventricular hypertrophy")
print(answers)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the documentation
2. Review existing issues
3. Create a new issue with detailed information
