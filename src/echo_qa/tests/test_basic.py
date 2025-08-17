"""Basic tests for EchoQA package."""

import unittest
from unittest.mock import Mock, patch
import tempfile
import json
from pathlib import Path

from echo_qa.core.api_client import OpenAIClient
from echo_qa.core.processor import CaptionProcessor
from echo_qa.utils.config import Config


class TestConfig(unittest.TestCase):
    """Test configuration management."""
    
    def test_config_creation(self):
        """Test basic config creation."""
        config = Config(data_str="train", batch_size=50)
        self.assertEqual(config.data_str, "train")
        self.assertEqual(config.batch_size, 50)
    
    def test_config_validation(self):
        """Test config validation."""
        config = Config(data_str="invalid")
        with self.assertRaises(ValueError):
            config.validate()


class TestAPIClient(unittest.TestCase):
    """Test OpenAI API client."""
    
    @patch('echo_qa.core.api_client.OpenAI')
    def test_client_initialization(self, mock_openai):
        """Test API client initialization."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        client = OpenAIClient(api_key="test-key")
        self.assertEqual(client.api_key, "test-key")
        self.assertEqual(client.model, "gpt-4")
    
    def test_missing_api_key(self):
        """Test error when API key is missing."""
        with self.assertRaises(ValueError):
            OpenAIClient(api_key=None)


class TestProcessor(unittest.TestCase):
    """Test caption processor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.api_client = Mock()
        self.processor = CaptionProcessor(
            api_client=self.api_client,
            input_dir=self.temp_dir,
            output_dir=self.temp_dir
        )
    
    def test_processor_initialization(self):
        """Test processor initialization."""
        self.assertEqual(self.processor.input_dir, Path(self.temp_dir))
        self.assertEqual(self.processor.output_dir, Path(self.temp_dir))


if __name__ == "__main__":
    unittest.main()
