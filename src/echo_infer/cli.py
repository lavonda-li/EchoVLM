"""Command-line interface for EchoPrime inference."""

import typer
from pathlib import Path
from typing import Optional

from .config import load_config, get_default_config
from .utils.logging import setup_logging

# Lazy imports to avoid EchoPrime submodule issues
def _get_run():
    from .pipeline import run
    return run

def _get_run_batch():
    from .pipeline import run_batch
    return run_batch

app = typer.Typer(help="EchoPrime inference CLI")


@app.command()
def run(
    config: str = typer.Option("configs/default.yaml", "--config", "-c", help="Configuration file path"),
    input: Optional[str] = typer.Option(None, "--input", "-i", help="Input directory (overrides config)"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory (overrides config)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """Run EchoPrime inference on DICOM files."""
    
    # Load configuration
    try:
        if Path(config).exists():
            cfg = load_config(config)
        else:
            typer.echo(f"Config file not found: {config}. Using defaults.")
            cfg = get_default_config()
    except Exception as e:
        typer.echo(f"Error loading config: {e}")
        raise typer.Exit(1)
    
    # Apply CLI overrides
    overrides = {}
    if input:
        overrides["data.input_dir"] = input
    if output:
        overrides["output.dir"] = output
    if verbose:
        overrides["logging.level"] = "DEBUG"
    
    if overrides:
        for key, value in overrides.items():
            keys = key.split('.')
            current = cfg
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
    
    # Setup logging
    setup_logging(
        level=cfg.get('logging', {}).get('level', 'INFO'),
        format="simple" if verbose else "structured"
    )
    
    # Run inference
    try:
        run_func = _get_run()
        results = run_func(cfg)
        typer.echo(f"Successfully processed {len(results)} files")
    except Exception as e:
        typer.echo(f"Error during inference: {e}")
        raise typer.Exit(1)


@app.command()
def batch(
    manifest: str = typer.Argument(..., help="Path to CSV manifest file"),
    config: str = typer.Option("configs/default.yaml", "--config", "-c", help="Configuration file path"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory (overrides config)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """Run batch inference using a manifest file."""
    
    # Load configuration
    try:
        if Path(config).exists():
            cfg = load_config(config)
        else:
            typer.echo(f"Config file not found: {config}. Using defaults.")
            cfg = get_default_config()
    except Exception as e:
        typer.echo(f"Error loading config: {e}")
        raise typer.Exit(1)
    
    # Apply CLI overrides
    if output:
        cfg['output'] = cfg.get('output', {})
        cfg['output']['dir'] = output
    if verbose:
        cfg['logging'] = cfg.get('logging', {})
        cfg['logging']['level'] = "DEBUG"
    
    # Setup logging
    setup_logging(
        level=cfg.get('logging', {}).get('level', 'INFO'),
        format="simple" if verbose else "structured"
    )
    
    # Run batch inference
    try:
        run_batch_func = _get_run_batch()
        results = run_batch_func(manifest, cfg, output)
        typer.echo(f"Batch processing complete. Total files processed: {len(results)}")
    except Exception as e:
        typer.echo(f"Error during batch inference: {e}")
        raise typer.Exit(1)


@app.command()
def info():
    """Show EchoPrime inference information."""
    typer.echo("EchoPrime Inference CLI")
    typer.echo("=====================")
    typer.echo("")
    typer.echo("Available commands:")
    typer.echo("  run     - Run inference on DICOM files")
    typer.echo("  batch   - Run batch inference using manifest")
    typer.echo("  info    - Show this information")
    typer.echo("")
    typer.echo("Examples:")
    typer.echo("  echo-infer run --config configs/default.yaml --input data/raw")
    typer.echo("  echo-infer batch manifest.csv --output results/")
    typer.echo("")
    typer.echo("For more help, use: echo-infer <command> --help")


if __name__ == "__main__":
    app()
