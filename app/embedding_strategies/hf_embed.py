import logging
import os
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

from .base import EmbeddingStrategy
# Assuming utils are available in the python path
from utils.gpu_manager import GPUManager
from utils.model_utils import get_model_path, ensure_dir_exists # Import helpers

logger = logging.getLogger(__name__)

# Cache for loaded models to avoid reloading within the same embedder instance
# The factory's lru_cache handles caching embedder instances themselves.
_model_cache: Dict[str, SentenceTransformer] = {}

# Function to clear the internal model cache (for testing)
def clear_hf_embedder_cache():
    global _model_cache
    _model_cache.clear()
    logger.info("Cleared HFEmbedder internal model cache (_model_cache).")

class HFEmbedder(EmbeddingStrategy):
    """Embedding strategy using local HuggingFace SentenceTransformer models,
       with local persistence and GPU management."""

    def __init__(self, 
                 model_path: str, 
                 use_gpu: bool = True,
                 local_model_dir: str = "models", 
                 **model_kwargs: Any):
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.local_model_dir = local_model_dir
        self.model_kwargs = model_kwargs
        self.gpu_manager = GPUManager() # Instantiate GPU manager
        self.model: SentenceTransformer | None = None # Initialize model attribute
        self._load_model()

    def _load_model(self):
        if self.model_path in _model_cache:
            self.model = _model_cache[self.model_path]
            logger.info(f"Reusing cached SentenceTransformer model (in-memory): {self.model_path}")
            
            # Simplified check: Just ensure the model is loaded.
            # Complex device consistency checks between cache and request removed for now.
            # The actual device placement happens below if not cached or during initial load.
            if self.model is not None:
                return # Use cached model
            else: # Should not happen if key exists, but defensive
                 logger.warning(f"Cache key '{self.model_path}' exists but model is None. Reloading.")
                 del _model_cache[self.model_path]

        # Determine the actual device using GPUManager
        device = self.gpu_manager.get_device(
            use_gpu=self.use_gpu,
            min_memory_gb=1.0, # Example minimum memory, adjust as needed
            task_name=f"Loading HF model {self.model_path}"
        )
        logger.info(f"Determined device for {self.model_path}: {device}")

        # Determine local path and ensure directory exists
        target_local_path = get_model_path(self.model_path, self.local_model_dir)
        ensure_dir_exists(self.local_model_dir)

        try:
            # Check if model exists in the target local directory
            if os.path.exists(target_local_path):
                logger.info(f"Loading SentenceTransformer model from local path: {target_local_path}")
                self.model = SentenceTransformer(target_local_path, device=device, **self.model_kwargs)
            else:
                logger.info(f"Local model not found at {target_local_path}. Downloading/loading {self.model_path} from source.")
                # Load from source (Hugging Face Hub or original path), let ST handle HF cache
                self.model = SentenceTransformer(self.model_path, device=device, **self.model_kwargs)
                
                # Save to the target local directory for future use
                try:
                    logger.info(f"Saving model to target local directory: {target_local_path}")
                    self.model.save(target_local_path)
                except Exception as save_e:
                    logger.warning(f"Failed to save model to {target_local_path}: {save_e}", exc_info=True)
                    # Continue even if saving fails, model is still loaded in memory
            
            _model_cache[self.model_path] = self.model
            logger.info(f"SentenceTransformer model '{self.model_path}' loaded successfully onto device '{device}'.")
            
            # Log embedding dimension
            try:
                embedding_dim = self.model.get_sentence_embedding_dimension()
                logger.info(f"Model '{self.model_path}' embedding dimension: {embedding_dim}")
            except Exception as dim_e:
                 logger.warning(f"Could not determine embedding dimension for '{self.model_path}': {dim_e}")

        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model '{self.model_path}': {e}", exc_info=True)
            raise ValueError(f"Could not load SentenceTransformer model '{self.model_path}': {e}") from e

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not self.model:
             raise RuntimeError(f"SentenceTransformer model '{self.model_path}' not loaded properly.")
        try:
            logger.debug(f"Generating embeddings for {len(texts)} texts using {self.model_path} on device {self.model.device}.")
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error during SentenceTransformer encoding for '{self.model_path}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to generate embeddings with '{self.model_path}': {e}") from e

    async def aembed(self, texts: List[str]) -> List[List[float]]:
        """Runs the synchronous embedding in an executor thread."""
        if not self.model:
             raise RuntimeError(f"SentenceTransformer model '{self.model_path}' not loaded properly before async call.")
             
        logger.debug(f"Running async HF embed for {len(texts)} texts using {self.model_path} in executor.")
        # SentenceTransformer encoding can be CPU/GPU intensive.
        # Running it in a thread pool executor prevents blocking the main async loop.
        return await self._run_sync_in_async(texts) 

    def get_embedding_dimension(self) -> int:
        """Returns the embedding dimension of the loaded model."""
        if self.model:
            try:
                # For SentenceTransformer models, this is the standard way
                dim = self.model.get_sentence_embedding_dimension()
                if dim is not None:
                    # logger.info(f"Reported embedding dimension: {dim} for model {self.model_path}")
                    return dim
                
                # Fallback for some other types of Hugging Face models if the above doesn't work
                # (though SentenceTransformer should have the above method)
                if hasattr(self.model, 'config') and hasattr(self.model.config, 'hidden_size'):
                    dim = self.model.config.hidden_size
                    # logger.info(f"Reported embedding dimension from config.hidden_size: {dim} for model {self.model_path}")
                    return dim

                # Fallback for SentenceTransformer models where the embedding layer is the first module (e.g. model[0])
                # and it's a Transformer model. model[0].auto_model.config.hidden_size
                if isinstance(self.model, torch.nn.Module) and hasattr(self.model, '0') and \
                   hasattr(self.model[0], 'auto_model') and \
                   hasattr(self.model[0].auto_model, 'config') and \
                   hasattr(self.model[0].auto_model.config, 'hidden_size'):
                    dim = self.model[0].auto_model.config.hidden_size
                    # logger.info(f"Reported embedding dimension from model[0].auto_model.config.hidden_size: {dim} for model {self.model_path}")
                    return dim

                logger.warning(f"Could not reliably determine embedding dimension for model '{self.model_path}'.")
                # You might want to raise an error or return a default/sentinel value
                # For moka-ai/m3e-base, the dimension is 768. If all else fails, and you know it,
                # you could hardcode it here, but it's preferable to get it from the model.
                # if self.model_path == "moka-ai/m3e-base": return 768
                return 0 # Or raise ValueError("Dimension not found")
            except Exception as e:
                logger.error(f"Error getting embedding dimension for '{self.model_path}': {e}", exc_info=True)
                # Depending on desired strictness, either raise or return a default
                raise ValueError(f"Could not get embedding dimension for '{self.model_path}': {e}") from e
        else:
            logger.error(f"Model '{self.model_path}' not loaded, cannot determine embedding dimension.")
            # raise RuntimeError(f"Model '{self.model_path}' not loaded.")
            return 0 # Or raise an error 