# app/chunking_strategies/factory.py

import os
import logging
from .base import BaseChunkingStrategy
from .recursive_character import RecursiveCharacterChunkingStrategy
# Import other strategies here when they are created
# from .semantic import SemanticChunkingStrategy
# from .markdown import MarkdownChunkingStrategy

logger = logging.getLogger(__name__)

def get_chunker() -> BaseChunkingStrategy:
    """
    Factory function to get the configured chunking strategy instance.
    Reads configuration from environment variables.
    """
    # Read strategy name, default to recursive_character
    strategy_name = os.environ.get("CHUNKING_STRATEGY", "recursive_character").lower()
    logger.info(f"Attempting to use chunking strategy: '{strategy_name}'")

    if strategy_name == "recursive_character":
        try:
            # Read chunk size and overlap from environment variables
            chunk_size = int(os.environ.get("CHUNK_SIZE", 1000))
            chunk_overlap = int(os.environ.get("CHUNK_OVERLAP", 100))
            logger.info(f"Using RecursiveCharacterChunkingStrategy with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
            return RecursiveCharacterChunkingStrategy(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        except ValueError:
            logger.warning("Invalid non-integer value for CHUNK_SIZE or CHUNK_OVERLAP env var. Using defaults (1000/100).")
            return RecursiveCharacterChunkingStrategy() # Fallback to defaults

    # Add elif blocks for other strategies here
    # elif strategy_name == "semantic":
    #     # ... logic to initialize SemanticChunkingStrategy ...
    #     pass
    # elif strategy_name == "markdown":
    #     # ... logic to initialize MarkdownChunkingStrategy ...
    #     pass

    else:
        # Instead of falling back, raise an error for unknown strategy
        logger.error(f"Unknown or unsupported chunking strategy specified: '{strategy_name}'. Check CHUNKING_STRATEGY env var.")
        raise ValueError(f"Unknown or unsupported chunking strategy: '{strategy_name}'")

        # Previous fallback logic (removed):
        # logger.warning(f"Unknown chunking strategy specified: '{strategy_name}'. Falling back to default strategy (recursive_character). Please check the CHUNKING_STRATEGY environment variable.")
        # return RecursiveCharacterChunkingStrategy() # Fallback to default 