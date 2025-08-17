"""
EchoQA - Medical Image Caption Processing and Question Answering

A package for processing medical image captions and generating structured Q&A data
using OpenAI's GPT models for medical image interpretation.
"""

__version__ = "0.1.0"
__author__ = "EchoPrime Team"

from .core.processor import CaptionProcessor
from .core.api_client import OpenAIClient
from .utils.config import Config

__all__ = ["CaptionProcessor", "OpenAIClient", "Config"]
