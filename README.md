# EchoVLM (Echo Vision Language Model)

A comprehensive framework for medical image analysis and question answering using vision-language models, built around the EchoPrime model.

## Overview

EchoVLM is a complete ecosystem for medical image processing, analysis, and question answering. It consists of multiple specialized packages that work together to provide end-to-end solutions for medical image interpretation tasks.

## Project Structure

```
EchoVLM/
├── modules/
│   └── EchoPrime/      # Core EchoPrime model (submodule)
├── src/
│   ├── echo_infer/     # Medical image inference (see README)
│   └── echo_qa/        # Medical image Q&A (see README)
├── configs/            # Configuration files
├── data/               # Data directories
├── tests/              # Project-wide tests
├── pyproject.toml      # Project configuration
├── Makefile            # Development tasks
└── README.md           # This file
```
For package-level details, see [`src/echo_infer/README.md`](src/echo_infer/README.md) and [`src/echo_qa/README.md`](src/echo_qa/README.md).

## Packages

### EchoInfer (`src/echo_infer/`)

**Medical Image Inference Package**

A modern, maintainable Python package for running EchoPrime model inference with minimal glue code and maximum reuse of the EchoPrime submodule.

**Key Features:**
- Thin wrapper around EchoPrime submodule
- CLI interface for inference and batch processing
- YAML-based configuration management
- Structured logging and error handling
- Support for DICOM file processing

**Quick Start:**
```bash
# Run inference on DICOM files
echo-infer run --config configs/default.yaml --input data/raw --output results/

# Batch processing
echo-infer batch manifest.csv --output results/
```

**Documentation:** See [`src/echo_infer/README.md`](src/echo_infer/README.md) for detailed documentation.

### EchoQA (`src/echo_qa/`)

**Medical Image Question Answering Package**

A Python package designed to process medical image captions and generate structured Q&A data using OpenAI's GPT models for medical image interpretation.

**Key Features:**
- Medical image caption processing
- Structured Q&A generation using GPT models
- Batch processing for large datasets
- Comprehensive error handling and logging
- Flexible configuration management

**Quick Start:**
```bash
# Process training data
echo-qa --data_str train --batch_size 100

# Process with custom settings
echo-qa --data_str val --batch_size 50 --start_idx 1000
```

**Documentation:** See [`src/echo_qa/README.md`](src/echo_qa/README.md) for detailed documentation.

## Installation

### Prerequisites

1. **Python 3.8+**
2. **Git submodules**: The EchoPrime model is included as a Git submodule
3. **OpenAI API key** (for EchoQA functionality)

### Setup

```bash
# Clone the repository with submodules
git clone --recursive <repository-url>
cd EchoVLM

# Initialize submodules if needed
git submodule update --init --recursive

# Install the entire project with all dependencies
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"

# Alternative: Use make commands if available
make install
make install-dev
```

### Environment Setup

```bash
# Set OpenAI API key for EchoQA
export OPENAI_API_KEY="your-api-key-here"

# Optional: Set other environment variables
export ECHO_DEVICE=cuda:0
export ECHO_LOG_LEVEL=INFO
```

## Quick Start Examples

### Medical Image Inference

```bash
# Process a directory of DICOM files
echo-infer run --config configs/default.yaml --input data/dicoms --output results/

# Run batch processing with manifest
echo-infer batch manifest.csv --config configs/default.yaml --output results/
```

### Medical Image Q&A

```bash
# Process training dataset
echo-qa --data_str train --batch_size 100

# Process validation dataset with custom settings
echo-qa --data_str val --batch_size 50 --start_idx 1000
```

### SLURM Cluster Processing

```bash
# Submit inference jobs
sbatch src/echo_infer/submit_inference.sh

# Submit Q&A processing jobs
sbatch src/echo_qa/submit_train.sh
sbatch src/echo_qa/submit_val.sh
sbatch src/echo_qa/submit_test.sh
```

## Configuration

### EchoInfer Configuration

EchoInfer uses YAML configuration files. See `configs/default.yaml` for the default configuration:

```yaml
model:
  name: echoprime
  weights_path: modules/EchoPrime/model_data/weights/echo_prime_encoder.pt
  device: cuda:0

data:
  input_dir: data/raw
  pattern: "*.dcm"
  batch_size: 16

output:
  dir: outputs/
```

### EchoQA Configuration

EchoQA uses command-line arguments and environment variables:

```bash
# Basic configuration
echo-qa --data_str train --batch_size 100

# Advanced configuration
echo-qa \
  --data_str train \
  --batch_size 100 \
  --openai_model gpt-4-turbo \
  --max_tokens 500 \
  --log_level DEBUG
```

## Development

### Project Setup

```bash
# Install development dependencies
make install-dev

# Install pre-commit hooks
make pre-commit
```

### Common Development Tasks

```bash
# Format code
make fmt

# Run linting
make lint

# Run tests
make test

# Run all checks
make check

# Clean build artifacts
make clean
```

### Testing

```bash
# Run all tests
pytest

# Run tests with coverage
make test-cov

# Run specific package tests
pytest tests/echo_infer/
pytest tests/echo_qa/
```

## Data Formats

### EchoInfer Input/Output

**Input:** DICOM files organized in directory structure
```
p10/p10002221/s94106955/94106955_0001.dcm
```

**Output:** JSON files with view predictions
```json
{
  "views": ["A2C", "A4C"],
  "video_shape": [3, 16, 224, 224],
  "device": "cuda:0",
  "source_file": "/path/to/original/file.dcm"
}
```

### EchoQA Input/Output

**Input:** JSON files with medical image captions
```json
[
  {
    "conversations": [
      {
        "value": "Echocardiogram showing normal left ventricular function"
      }
    ],
    "id": "unique_identifier"
  }
]
```

**Output:** JSON files with structured Q&A data
```json
[
  {
    "id": "unique_identifier",
    "caption": "Echocardiogram showing normal left ventricular function",
    "answers": [
      "A1: Echocardiogram",
      "A2: Left ventricle",
      "A3: No abnormalities",
      "A4: Normal appearance",
      "A5: No significant labels"
    ]
  }
]
```

## Medical Questions

EchoQA automatically asks the following medical questions for each caption:

1. **Q1**: What imaging modality is represented in this image?
2. **Q2**: What body region or anatomical area does this image depict?
3. **Q3**: Are there any abnormalities identified in this image?
4. **Q4**: Does this image appear normal, or does it show any irregularities?
5. **Q5**: Does this image contain any label or index that is significant or noteworthy?

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure EchoPrime submodule is properly initialized
   ```bash
   git submodule update --init --recursive
   ```

2. **CUDA errors**: Check device configuration
   ```yaml
   model:
     device: cpu  # Use CPU if CUDA not available
   ```

3. **OpenAI API errors**: Verify API key is set
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

4. **Path issues**: Ensure working directory is correct
   ```bash
   # Run from repository root
   cd /path/to/EchoVLM
   ```

### Error Handling

Both packages are designed to handle errors gracefully:

- **Individual file failures**: Processing continues with other files
- **Partial results**: Returns results for successfully processed files
- **Detailed logging**: Shows success/error counts and specific failure reasons
- **Graceful degradation**: Continues processing instead of crashing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `make check`
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- EchoPrime team for the core model and processing pipeline
- OpenAI team for the GPT models used in Q&A generation
- PyTorch team for the deep learning framework
- Typer team for the excellent CLI framework


