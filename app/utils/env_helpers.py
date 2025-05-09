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

def _get_bool_from_env(env_var: str, default: bool) -> bool:
    """Helper to get a boolean from environment variables with a default.
    Considers 'true', '1', 'yes', 't' (case-insensitive) as True.
    Considers 'false', '0', 'no', 'f' (case-insensitive) as False.
    Otherwise, or if not set, returns the default.
    """
    logger.debug(f"Attempting to get env var '{env_var}' as bool, default: {default}")
    value = os.environ.get(env_var)
    if value is None:
        logger.debug(f"Env var '{env_var}' not found, returning default: {default}")
        return default
    
    # Normalize to lowercase for case-insensitive comparison
    value_lower = value.strip().lower()
    
    if value_lower in ["true", "1", "yes", "t"]:
        logger.debug(f"Env var '{env_var}' found with value '{value}', returning as bool: True")
        return True
    elif value_lower in ["false", "0", "no", "f"]:
        logger.debug(f"Env var '{env_var}' found with value '{value}', returning as bool: False")
        return False
    else:
        logger.warning(f"Invalid value '{value}' for env var '{env_var}' for boolean conversion. "
                       f"Expected common boolean strings (e.g., 'true', 'false', '1', '0'). Using default: {default}")
        return default

# Add other environment-related helpers here if needed later 