# app/reranker_strategies/factory.py

import os
import logging
from .base import BaseRerankerStrategy
from .noop import NoOpRerankerStrategy
# Import other reranker strategies here when they are created
# e.g., from .cross_encoder import CrossEncoderRerankerStrategy

logger = logging.getLogger(__name__)

def get_reranker(model_name: str, **kwargs) -> BaseRerankerStrategy:
    """
    Factory function to get the configured reranker strategy instance.

    Args:
        model_name: The name or path of the reranker model (used to select the strategy).
                      For NoOp, this can be anything non-empty or a specific keyword.
        **kwargs: Additional keyword arguments for specific reranker strategies.

    Returns:
        An instance of a BaseRerankerStrategy.

    Raises:
        ValueError: If the specified model or strategy is not supported.
    """
    # Normalize model_name for comparison (optional, but can be helpful)
    normalized_name = model_name.lower()
    logger.info(f"Attempting to initialize reranker strategy for model/name: '{model_name}'")

    # --- Strategy Selection Logic --- #

    # Example: Add logic for actual rerankers first
    # if "cross-encoder" in normalized_name:
    #     logger.info("Selected CrossEncoderRerankerStrategy.")
    #     # You might need additional env vars or kwargs for CrossEncoder
    #     # e.g., batch_size = int(os.environ.get("RERANKER_BATCH_SIZE", 8))
    #     try:
    #         return CrossEncoderRerankerStrategy(model_name=model_name, **kwargs)
    #     except Exception as e:
    #          logger.error(f"Failed to initialize CrossEncoderReranker: {e}", exc_info=True)
    #          raise ValueError(f"Could not initialize CrossEncoder model '{model_name}'") from e

    # Fallback to NoOp Reranker if no specific strategy matches
    # Or you could make NoOp explicit, e.g., if model_name == "noop"
    # For now, let's default to NoOp if nothing else is recognized
    logger.warning(f"No specific reranker strategy recognized for '{model_name}'. Falling back to NoOpRerankerStrategy.")
    return NoOpRerankerStrategy()

    # If you prefer explicit selection:
    # else:
    #     logger.error(f"Unsupported reranker model or type specified: '{model_name}'")
    #     raise ValueError(f"Unsupported reranker model or type: '{model_name}'") 