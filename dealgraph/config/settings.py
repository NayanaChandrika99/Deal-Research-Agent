# ABOUTME: Configuration settings loaded from environment variables.
# ABOUTME: Provides centralized access to API keys, model names, and system parameters.

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    """Centralized configuration for DealGraph Agent."""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    PROMPTS_DIR = PROJECT_ROOT / "prompts"
    MODELS_DIR = PROJECT_ROOT / "models"
    RESULTS_DIR = PROJECT_ROOT / "results"
    
    # Cerebras API (for LLM)
    CEREBRAS_API_KEY: str = os.getenv("CEREBRAS_API_KEY", "")
    CEREBRAS_BASE_URL: str = os.getenv("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1")
    
    # OpenAI API (for embeddings)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # DSPy Configuration
    DSPY_MODEL: str = os.getenv("DSPY_MODEL", "llama3.1-8b")
    DSPY_API_BASE: str = os.getenv("DSPY_API_BASE", "https://api.cerebras.ai/v1")
    DSPY_TEMPERATURE: float = float(os.getenv("DSPY_TEMPERATURE", "0.7"))
    DSPY_MAX_TOKENS: int = int(os.getenv("DSPY_MAX_TOKENS", "2000"))
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
    
    # LLM-as-Judge Configuration
    JUDGE_MODEL: str = os.getenv("JUDGE_MODEL", "llama3.1-8b")
    JUDGE_API_BASE: str = os.getenv("JUDGE_API_BASE", "https://api.cerebras.ai/v1")
    JUDGE_TEMPERATURE: float = float(os.getenv("JUDGE_TEMPERATURE", "0.3"))
    
    # Rate Limiting
    MAX_REQUESTS_PER_MINUTE: int = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "30"))
    MAX_INPUT_TOKENS_PER_MINUTE: int = int(os.getenv("MAX_INPUT_TOKENS_PER_MINUTE", "60000"))
    
    # Context Window Settings
    MAX_CONTEXT_LENGTH: int = int(os.getenv("MAX_CONTEXT_LENGTH", "8192"))
    MAX_OUTPUT_TOKENS: int = int(os.getenv("MAX_OUTPUT_TOKENS", "8192"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def validate(cls) -> None:
        """Validate that required API keys are present."""
        if not cls.CEREBRAS_API_KEY:
            raise ValueError("CEREBRAS_API_KEY not found in environment variables")
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

# Global settings instance
settings = Settings()

