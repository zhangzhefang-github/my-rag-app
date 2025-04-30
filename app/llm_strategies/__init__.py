# app/llm_strategies/__init__.py
import os
import logging
from functools import lru_cache
from dotenv import load_dotenv

from .base import LLMStrategy
from .openai_strategy import OpenAIStrategy
from .ollama_strategy import OllamaStrategy
from .custom_api_strategy import CustomAPIStrategy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
# If .env is not in the same directory as this script, adjust the path
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env') # Assumes .env is in the project root
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    logger.info(f".env file loaded from: {dotenv_path}")
else:
    # Also try loading from the default location just in case
    load_dotenv()
    logger.info("Attempted to load .env from default location.")


# Use lru_cache to cache the strategy instance for performance
@lru_cache(maxsize=1)
def get_llm_strategy() -> LLMStrategy:
    """
    Reads configuration from environment variables and returns the appropriate 
    LLM strategy instance. The instance is cached after the first call.

    Environment Variables Required:
        LLM_PROVIDER: 'openai', 'ollama', or 'custom_api'.
        
        If LLM_PROVIDER='openai':
            OPENAI_API_KEY: Your OpenAI API key.
            OPENAI_API_BASE: (Optional) Custom base URL for OpenAI API.
            OPENAI_MODEL: (Optional) OpenAI model name (default: 'gpt-4o').
            OPENAI_TEMPERATURE: (Optional) Temperature (default: 0.7).
            OPENAI_MAX_TOKENS: (Optional) Max tokens (default: 1024).

        If LLM_PROVIDER='ollama':
            OLLAMA_HOST: Base URL of your Ollama service.
            OLLAMA_MODEL: (Optional) Ollama model name (default: 'qwen3:0.6b').
            OLLAMA_TEMPERATURE: (Optional) Temperature (default: 0.7).
            OLLAMA_NUM_PREDICT: (Optional) Max tokens for Ollama (default: 256).
            
        If LLM_PROVIDER='custom_api':
            CUSTOM_API_KEY: Your API key for the custom endpoint.
            CUSTOM_API_BASE: The base URL for the custom endpoint.
            CUSTOM_API_MODEL: (Optional) Model name for the custom endpoint (default: 'gpt-3.5-turbo').
            CUSTOM_API_TEMPERATURE: (Optional) Temperature (default: 0.7).
            CUSTOM_API_MAX_TOKENS: (Optional) Max tokens (default: 1024).

    Returns:
        An instance of LLMStrategy (OpenAIStrategy, OllamaStrategy, or CustomAPIStrategy).
        
    Raises:
        ValueError: If required environment variables are missing or 
                    LLM_PROVIDER is unsupported.
    """
    llm_provider = os.getenv("LLM_PROVIDER", "custom_api").lower()
    logger.info(f"LLM Provider selected: {llm_provider}")

    if llm_provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required when LLM_PROVIDER is 'openai'.")
        
        base_url = os.getenv("OPENAI_API_BASE")
        model = os.getenv("OPENAI_MODEL", "gpt-4o")
        temperature = float(os.getenv("OPENAI_TEMPERATURE", 0.7))
        # Handle potential ValueError if max_tokens is set but not an integer
        try:
            max_tokens_str = os.getenv("OPENAI_MAX_TOKENS")
            max_tokens = int(max_tokens_str) if max_tokens_str else 1024
        except ValueError:
            logger.warning(f"Invalid value for OPENAI_MAX_TOKENS: '{max_tokens_str}'. Using default 1024.")
            max_tokens = 1024
            
        logger.info(f"Initializing OpenAIStrategy with model: {model}, base_url: {base_url}")
        return OpenAIStrategy(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
    elif llm_provider == "ollama":
        host = os.getenv("OLLAMA_HOST")
        if not host:
            raise ValueError("OLLAMA_HOST environment variable is required when LLM_PROVIDER is 'ollama'.")
            
        model = os.getenv("OLLAMA_MODEL", "qwen3:0.6b")
        temperature = float(os.getenv("OLLAMA_TEMPERATURE", 0.7))
        # Handle potential ValueError for num_predict
        try:
            num_predict_str = os.getenv("OLLAMA_NUM_PREDICT")
            num_predict = int(num_predict_str) if num_predict_str else 256
        except ValueError:
            logger.warning(f"Invalid value for OLLAMA_NUM_PREDICT: '{num_predict_str}'. Using default 256.")
            num_predict = 256

        logger.info(f"Initializing OllamaStrategy with model: {model}, host: {host}")
        return OllamaStrategy(
            host=host,
            model=model,
            temperature=temperature,
            num_predict=num_predict
        )
        
    elif llm_provider == "custom_api":
        api_key = os.getenv("CUSTOM_API_KEY")
        if not api_key:
            raise ValueError("CUSTOM_API_KEY environment variable is required when LLM_PROVIDER is 'custom_api'.")
        
        base_url = os.getenv("CUSTOM_API_BASE")
        if not base_url:
            raise ValueError("CUSTOM_API_BASE environment variable is required when LLM_PROVIDER is 'custom_api'.")
        
        model = os.getenv("CUSTOM_API_MODEL", "gpt-3.5-turbo")
        temperature = float(os.getenv("CUSTOM_API_TEMPERATURE", 0.7))
        # Handle potential ValueError if max_tokens is set but not an integer
        try:
            max_tokens_str = os.getenv("CUSTOM_API_MAX_TOKENS")
            max_tokens = int(max_tokens_str) if max_tokens_str else 1024
        except ValueError:
            logger.warning(f"Invalid value for CUSTOM_API_MAX_TOKENS: '{max_tokens_str}'. Using default 1024.")
            max_tokens = 1024
            
        logger.info(f"Initializing CustomAPIStrategy with model: {model}, base_url: {base_url}")
        return CustomAPIStrategy(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: '{llm_provider}'. Please use 'openai', 'ollama', or 'custom_api'.")

# Expose the factory function for easy import
__all__ = ["get_llm_strategy", "LLMStrategy", "CustomAPIStrategy", "OpenAIStrategy", "OllamaStrategy"] 