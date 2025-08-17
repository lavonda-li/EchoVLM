"""Tests for CLI functionality."""

import pytest
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_cli_app_creation():
    """Test that CLI app can be created."""
    from echo_infer.cli import app
    
    assert app is not None
    assert hasattr(app, "registered_commands")


def test_cli_commands_exist():
    """Test that expected CLI commands exist."""
    from echo_infer.cli import app
    
    commands = [cmd.callback.__name__ for cmd in app.registered_commands]
    expected_commands = ["run", "batch", "info"]
    
    # Check that we have the right number of commands
    assert len(commands) == 3
    # Check that all expected commands are present (order may vary)
    for cmd in expected_commands:
        assert cmd in commands


def test_cli_info_command():
    """Test info command output."""
    from echo_infer.cli import app
    from typer.testing import CliRunner
    
    runner = CliRunner()
    result = runner.invoke(app, ["info"])
    
    assert result.exit_code == 0
    assert "EchoPrime Inference CLI" in result.stdout


def test_cli_run_command_help():
    """Test run command help."""
    from echo_infer.cli import app
    from typer.testing import CliRunner
    
    runner = CliRunner()
    result = runner.invoke(app, ["run", "--help"])
    
    assert result.exit_code == 0
    assert "Run EchoPrime inference on DICOM files" in result.stdout


def test_cli_batch_command_help():
    """Test batch command help."""
    from echo_infer.cli import app
    from typer.testing import CliRunner
    
    runner = CliRunner()
    result = runner.invoke(app, ["batch", "--help"])
    
    assert result.exit_code == 0
    assert "Run batch inference using a manifest file" in result.stdout


@patch('echo_infer.cli.load_config')
@patch('echo_infer.cli._get_run')
def test_cli_run_command_success(mock_get_run, mock_load_config):
    """Test successful run command execution."""
    from echo_infer.cli import app
    from typer.testing import CliRunner
    
    # Mock return values
    mock_config = {"model": {"name": "echoprime"}, "data": {}, "output": {}}
    mock_load_config.return_value = mock_config
    mock_run_func = MagicMock()
    mock_run_func.return_value = {"file1.dcm": {"views": ["A2C"]}}
    mock_get_run.return_value = mock_run_func
    
    runner = CliRunner()
    result = runner.invoke(app, ["run", "--config", "configs/default.yaml"])
    
    assert result.exit_code == 0
    assert "Successfully processed 1 files" in result.stdout
    mock_get_run.assert_called_once()


@patch('echo_infer.cli.load_config')
def test_cli_run_command_config_not_found(mock_load_config):
    """Test run command with missing config file."""
    from echo_infer.cli import app
    from typer.testing import CliRunner
    
    mock_load_config.side_effect = FileNotFoundError("Config not found")
    
    runner = CliRunner()
    result = runner.invoke(app, ["run", "--config", "nonexistent.yaml"])
    
    assert result.exit_code == 1
    # The error message may vary due to EchoPrime path issues
    assert "Error" in result.stdout


@patch('echo_infer.cli.load_config')
@patch('echo_infer.cli._get_run_batch')
def test_cli_batch_command_success(mock_get_run_batch, mock_load_config):
    """Test successful batch command execution."""
    from echo_infer.cli import app
    from typer.testing import CliRunner
    
    # Mock return values
    mock_config = {"model": {"name": "echoprime"}, "data": {}, "output": {}}
    mock_load_config.return_value = mock_config
    mock_run_batch_func = MagicMock()
    mock_run_batch_func.return_value = {"file1.dcm": {"views": ["A2C"]}}
    mock_get_run_batch.return_value = mock_run_batch_func
    
    # Create a temporary manifest file
    manifest_content = "input,output\n/path/to/input,/path/to/output"
    
    runner = CliRunner()
    with runner.isolated_filesystem():
        with open("manifest.csv", "w") as f:
            f.write(manifest_content)
        
        result = runner.invoke(app, ["batch", "manifest.csv"])
    
    assert result.exit_code == 0
    assert "Batch processing complete" in result.stdout
    mock_get_run_batch.assert_called_once()


def test_cli_batch_command_missing_manifest():
    """Test batch command with missing manifest file."""
    from echo_infer.cli import app
    from typer.testing import CliRunner
    
    runner = CliRunner()
    result = runner.invoke(app, ["batch", "nonexistent.csv"])
    
    assert result.exit_code == 1
    assert "Error during batch inference" in result.stdout
