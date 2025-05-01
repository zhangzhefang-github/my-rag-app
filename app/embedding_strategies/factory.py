import logging
# from functools import lru_cache # Removed

from .base import EmbeddingStrategy
from .hf_embed import HFEmbedder
from .openai_embed import OpenAIEmbedder
from .ollama_embed import OllamaEmbedder
# Import the config models
from .config import EmbeddingConfig, OpenAIConfig, HuggingFaceConfig, OllamaConfig

logger = logging.getLogger(__name__)

# Using lru_cache to reuse embedder instances for the same config
# This is useful if get_embedder is called multiple times with identical settings.
# Adjust maxsize as needed.
# @lru_cache(maxsize=16) # Removed due to unhashable Pydantic model
def get_embedder(config_wrapper: EmbeddingConfig) -> EmbeddingStrategy:
    """Factory function to create and return an embedding strategy instance based on config."""
    # The config_wrapper has a single 'config' attribute holding the specific provider config
    config = config_wrapper.config
    provider = config.provider

    logger.info(f"Creating embedder for provider: {provider} with config: {config.model_dump(exclude={'api_key'})}") # Exclude sensitive info

    try:
        if isinstance(config, OpenAIConfig):
            return OpenAIEmbedder(
                model=config.model,
                api_key=config.api_key,
                base_url=str(config.base_url) if config.base_url else None,
                **config.client_kwargs
            )
        elif isinstance(config, HuggingFaceConfig):
            # Note: Caching within HFEmbedder now handles model reuse for same path
            # and checks device consistency. The lru_cache here prevents recreating 
            # the HFEmbedder instance itself unnecessarily for identical config objects.
            return HFEmbedder(
                model_path=config.model_path,
                # device=config.device, # Removed
                use_gpu=config.use_gpu, # Added
                local_model_dir=config.local_model_dir, # Added
                **config.model_kwargs
            )
        elif isinstance(config, OllamaConfig):
            return OllamaEmbedder(
                model=config.model,
                base_url=str(config.base_url),
                request_timeout=config.request_timeout
            )
        else:
            # This path should theoretically not be reached due to Pydantic validation
            # but added for robustness.
            raise ValueError(f"Unsupported provider type encountered in factory: {provider}")

    except Exception as e:
        logger.error(f"Failed to create embedder for provider {provider}: {e}", exc_info=True)
        # Re-raise the exception to signal failure to the caller
        raise RuntimeError(f"Failed to initialize embedding strategy for {provider}: {e}") from e

# --- Example Usage (within factory.py or in another file) ---
# if __name__ == '__main__':
#     import os
#     from dotenv import load_dotenv
#     from .config import EmbeddingConfig
#
#     load_dotenv() # Load environment variables (e.g., OPENAI_API_KEY)
#
#     # Example Configs (Load from YAML/JSON/Dict in real application)
#     openai_conf_data = {
#         "config": {
#             "provider": "openai",
#             "model": "text-embedding-3-small"
#             # API key loaded from env
#         }
#     }
#
#     hf_conf_data = {
#         "config": {
#             "provider": "huggingface",
#             "model_path": "sentence-transformers/all-MiniLM-L6-v2"
#         }
#     }
#
#     ollama_conf_data = {
#         "config": {
#             "provider": "ollama",
#             "model": "nomic-embed-text",
#             "base_url": "http://localhost:11434"
#         }
#     }
#
#     try:
#         # Validate and parse configs
#         openai_config = EmbeddingConfig(**openai_conf_data)
#         hf_config = EmbeddingConfig(**hf_conf_data)
#         ollama_config = EmbeddingConfig(**ollama_conf_data)
#
#         # Get embedders using the factory (instances might be cached)
#         openai_embedder = get_embedder(openai_config)
#         hf_embedder = get_embedder(hf_config)
#         ollama_embedder = get_embedder(ollama_config)
#
#         print("Embedders created successfully!")
#
#         # --- Test Sync Embedding ---
#         texts_to_embed = ["hello world", "embedding test sentence"]
#
#         print("\n--- Testing OpenAI Sync ---")
#         openai_vectors = openai_embedder.embed(texts_to_embed)
#         print(f"OpenAI vectors shape: ({len(openai_vectors)}, {len(openai_vectors[0])})")
#
#         print("\n--- Testing HuggingFace Sync ---")
#         hf_vectors = hf_embedder.embed(texts_to_embed)
#         print(f"HF vectors shape: ({len(hf_vectors)}, {len(hf_vectors[0])})")
#
#         print("\n--- Testing Ollama Sync ---")
#         try:
#             ollama_vectors = ollama_embedder.embed(texts_to_embed)
#             print(f"Ollama vectors shape: ({len(ollama_vectors)}, {len(ollama_vectors[0])})")
#         except Exception as e:
#             print(f"Could not run Ollama sync test: {e}") # Ollama might not be running
#
#         # --- Test Async Embedding ---
#         import asyncio
#
#         async def run_async_tests():
#             print("\n--- Testing OpenAI Async ---")
#             openai_vectors_async = await openai_embedder.aembed(texts_to_embed)
#             print(f"OpenAI async vectors shape: ({len(openai_vectors_async)}, {len(openai_vectors_async[0])})")
#
#             print("\n--- Testing HuggingFace Async ---")
#             hf_vectors_async = await hf_embedder.aembed(texts_to_embed)
#             print(f"HF async vectors shape: ({len(hf_vectors_async)}, {len(hf_vectors_async[0])})")
#
#             print("\n--- Testing Ollama Async ---")
#             try:
#                 ollama_vectors_async = await ollama_embedder.aembed(texts_to_embed)
#                 print(f"Ollama async vectors shape: ({len(ollama_vectors_async)}, {len(ollama_vectors_async[0])})")
#             except Exception as e:
#                 print(f"Could not run Ollama async test: {e}")
#
#         asyncio.run(run_async_tests())
#
#     except Exception as e:
#         print(f"An error occurred during setup or testing: {e}") 