import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def _get_int_from_env(env_var: str, default: int) -> int:
    """Helper to get an integer from environment variables with a default."""
    logger.debug(f"Attempting to get env var '{env_var}' as int, default: {default}")
    value = os.environ.get(env_var)
    if value is None:
        logger.debug(f"Env var '{env_var}' not found, returning default: {default}")
        return default
    try:
        int_value = int(value)
        logger.debug(f"Env var '{env_var}' found with value '{value}', returning as int: {int_value}")
        return int_value
    except (ValueError, TypeError):
        logger.warning(f"Invalid value '{value}' for env var {env_var}. Must be an integer. Using default: {default}")
        return default

# Add other environment-related helpers here if needed later 