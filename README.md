# EchoPrime Inference

A modern, maintainable Python package for running EchoPrime model inference with minimal glue code and maximum reuse of the EchoPrime submodule.

## Features

- **Thin wrapper**: Minimal code that delegates to EchoPrime submodule
- **Modern layout**: Standard `src/` layout with clear separation of concerns
- **CLI interface**: Easy-to-use command-line interface for inference and batch processing
- **Configuration management**: YAML-based configuration with environment variable overrides
- **Structured logging**: JSON-formatted logs for better observability
- **Type hints**: Full type annotations for better IDE support and code quality
- **Testing**: Comprehensive test suite with smoke tests and CLI tests

## Quick Start

### Installation

```bash
# Install in development mode
make install

# Or with development dependencies
make install-dev
```

### Basic Usage

```bash
# Run inference with default config
echo-infer run- --config configs/default.yaml --input data/raw --output results/

# Run with verbose logging
echo-infer run- --config configs/default.yaml --verbose

# Run batch processing
echo-infer batch manifest.csv --output results/

# Show help
echo-infer --help
```

### Output Format

Results are saved as individual JSON files with names based on the directory structure:

- **Input**: `p10/p10002221/s94106955/94106955_0001.dcm`
- **Output**: `results/p10_p10002221_s94106955.json`

Each result file contains:
```json
{
  "views": ["A2C", "A4C"],
  "video_shape": [3, 16, 224, 224],
  "device": "cuda:0",
  "source_file": "/path/to/original/file.dcm"
}
```

### Configuration

The package uses YAML configuration files. See `configs/default.yaml` for the default configuration:

```yaml
model:
  name: echoprime
  weights_path: modules/EchoPrime/model_data/weights/echo_prime_encoder.pt
  device: cuda:0

data:
  input_dir: data/raw
  pattern: "*.dcm"
  batch_size: 16
  num_workers: 4

infer:
  fp16: true
  num_threads: 4
  save_probs: true

output:
  dir: outputs/

logging:
  level: INFO
```

### Environment Variables

You can override configuration values using environment variables:

```bash
export ECHO_DEVICE=cpu
export ECHO_BATCH_SIZE=8
export ECHO_INPUT_DIR=/path/to/data
export ECHO_OUTPUT_DIR=/path/to/results
export ECHO_LOG_LEVEL=DEBUG
```

## Project Structure

```
.
├── modules/
│   └── EchoPrime/              # Git submodule (unchanged)
├── src/
│   └── echo_infer/
│       ├── __init__.py         # Package exports
│       ├── cli.py              # Typer CLI interface
│       ├── config.py           # YAML config loading
│       ├── pipeline.py         # Main inference pipeline
│       ├── adapters/           # Thin wrappers to submodule
│       │   ├── model_adapter.py
│       │   └── dataset_adapter.py
│       └── utils/
│           ├── io.py           # File operations
│           └── logging.py      # Structured logging
├── configs/
│   └── default.yaml           # Default configuration
├── tests/
│   ├── test_smoke.py          # Smoke tests
│   └── test_cli.py            # CLI tests
├── pyproject.toml             # Package configuration
├── Makefile                   # Development tasks
└── README.md                  # This file
```

## Development

### Setup

```bash
# Install development dependencies
make install-dev

# Install pre-commit hooks
make pre-commit
```

### Common Tasks

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

# Run with coverage
make test-cov

# Run specific test file
pytest tests/test_smoke.py
```

## API Reference

### Main Functions

- `echo_infer.run(config)`: Run inference pipeline
- `echo_infer.run_batch(manifest, config)`: Run batch inference
- `echo_infer.load_config(path, overrides)`: Load configuration
- `echo_infer.load_echoprime_model(**kwargs)`: Load EchoPrime model

### CLI Commands

- `echo-infer run-`: Run inference on DICOM files
- `echo-infer batch`: Run batch inference using manifest
- `echo-infer info`: Show package information

## EchoPrime Integration

This package is designed as a thin wrapper around the EchoPrime submodule. It:

1. **Reuses EchoPrime code**: All heavy lifting is delegated to the submodule
2. **Provides clean interfaces**: Simple, well-documented APIs for common tasks
3. **Handles path management**: Automatically manages working directory changes for relative paths
4. **Maintains compatibility**: Doesn't modify EchoPrime internals

### Import Paths

The package automatically handles importing from the EchoPrime submodule:

```python
# Before (broken after submodule move)
from echoprime.dataset import EchoDataset

# After (handled by adapters)
from echo_infer.adapters.model_adapter import load_echoprime_model
```

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure EchoPrime submodule is properly initialized
   ```bash
   git submodule update --init --recursive
   ```

2. **CUDA errors**: Check device configuration in config file
   ```yaml
   model:
     device: cpu  # Use CPU if CUDA not available
   ```

3. **Missing dependencies**: Install development dependencies
   ```bash
   make install-dev
   ```

4. **Path issues**: Ensure working directory is correct for relative paths
   ```bash
   # Run from repository root
   cd /path/to/echoprime-inference
   echo-infer run- --config configs/default.yaml
   ```

### Debug Mode

Enable verbose logging for debugging:

```bash
echo-infer run- --config configs/default.yaml --verbose
```

Or set environment variable:

```bash
export ECHO_LOG_LEVEL=DEBUG
echo-infer run- --config configs/default.yaml
```

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
- PyTorch team for the deep learning framework
- Typer team for the excellent CLI framework
