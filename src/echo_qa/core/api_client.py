"""OpenAI API client for medical image caption processing."""

import os
import logging
from typing import List, Dict, Any
from openai import OpenAI
from openai import APIError, RateLimitError, APIConnectionError

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Client for interacting with OpenAI API for medical image caption processing."""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4", max_tokens: int = 300):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key. If None, will try to get from environment
            model: Model to use for processing
            max_tokens: Maximum tokens for response
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.model = model
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=self.api_key)
        
        # Medical questions for caption processing
        self.questions = [
            "Q1: What imaging modality is represented in this image?",
            "Q2: What body region or anatomical area does this image depict?",
            "Q3: Are there any abnormalities identified in this image?",
            "Q4: Does this image appear normal, or does it show any irregularities?",
            "Q5: Does this image contain any label or index that is significant or noteworthy?",
        ]
    
    def process_caption(self, caption: str) -> List[str]:
        """
        Process a medical image caption using OpenAI API.
        
        Args:
            caption: The medical image caption to process
            
        Returns:
            List of answers to the medical questions
            
        Raises:
            APIError: If there's an issue with the OpenAI API
            RateLimitError: If rate limit is exceeded
            APIConnectionError: If there's a connection issue
        """
        try:
            question_str = "\n".join(self.questions)
            messages = [
                {
                    "role": "system", 
                    "content": "You are a medical expert trained to interpret medical image captions."
                },
                {
                    "role": "user", 
                    "content": f"""
                        For the provided caption, answer the following questions strictly based on the caption:
                        Caption: {caption}
                        {question_str}
                        Provide concise answers for each question. For each answer, start with 'A1: ' for answer 1 and so on."""
                }
            ]
            
            logger.debug(f"Processing caption: {caption[:100]}...")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
            )
            
            result = response.choices[0].message.content.strip().split("\n")
            logger.debug(f"Received {len(result)} answers from API")
            
            return result
            
        except RateLimitError as e:
            logger.error(f"Rate limit exceeded: {e}")
            raise
        except APIConnectionError as e:
            logger.error(f"API connection error: {e}")
            raise
        except APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error processing caption: {e}")
            raise
    
    def get_questions(self) -> List[str]:
        """Get the list of medical questions used for processing."""
        return self.questions.copy()
