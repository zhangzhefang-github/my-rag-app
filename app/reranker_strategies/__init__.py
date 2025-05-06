# This file makes the directory a Python package 

import os
import logging
from typing import Optional

from .base import BaseRerankerStrategy
from .cross_encoder import CrossEncoderReranker

logger = logging.getLogger(__name__)

# --- Default Reranker Configuration ---
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"
DEFAULT_RERANKER_DEVICE = "auto"
# In the future, we might add a RERANKER_STRATEGY env var if more types are added.
# For now, we assume CrossEncoder is the only type.
# ---

def get_reranker(
    model_name: Optional[str] = None,
    device: Optional[str] = None
) -> Optional[BaseRerankerStrategy]:
    """
    Factory function to get an instance of a reranker strategy.

    Reads configuration from environment variables if not provided explicitly.
    Currently only supports CrossEncoderReranker. Returns None if reranking
    is implicitly disabled (e.g., model name not specified).

    Args:
        model_name: The name or path of the reranker model. Overrides env var.
        device: The device to use ('cpu', 'cuda', 'auto'). Overrides env var.

    Returns:
        An instance of BaseRerankerStrategy or None if no model name is found.
    """
    _model = model_name or os.environ.get("RERANKER_MODEL", DEFAULT_RERANKER_MODEL)
    _device = device or os.environ.get("RERANKER_DEVICE", DEFAULT_RERANKER_DEVICE)

    if not _model:
        logger.info("No RERANKER_MODEL specified. Reranker strategy cannot be initialized.")
        return None

    # Currently, only CrossEncoder strategy is supported.
    # In the future, could use a RERANKER_STRATEGY env var here.
    try:
        logger.info(f"Attempting to initialize reranker strategy: CrossEncoderReranker")
        strategy = CrossEncoderReranker(model_name=_model, device=_device)
        return strategy
    except Exception as e:
        logger.error(f"Failed to initialize reranker strategy with model '{_model}': {e}", exc_info=True)
        # Return None to indicate failure, allows Retriever to disable reranking
        return None 