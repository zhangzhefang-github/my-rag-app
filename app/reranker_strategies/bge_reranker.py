import logging
import os # Import os
import torch
from typing import List, Dict, Any, Optional
from sentence_transformers.cross_encoder import CrossEncoder
from .base import BaseRerankerStrategy

logger = logging.getLogger(__name__)

class BGERerankerStrategy(BaseRerankerStrategy):
    """Reranking strategy using BAAI BGE Reranker models."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: Optional[str] = None, # Auto-detect or use configured device
        **kwargs: Any
    ):
        """
        Initializes the BGE Reranker strategy.

        Args:
            model_name: The name of the BGE reranker model on Hugging Face.
            device: The device to run the model on ('cpu', 'cuda', 'gpu', None for auto).
                      If 'gpu' is specified, it checks CUDA availability and uses 'cuda' if possible, otherwise 'cpu'.
            **kwargs: Additional arguments passed to CrossEncoder constructor.
        """
        # --- Resolve device name ---
        resolved_device = device
        if device == "gpu":
            if torch.cuda.is_available():
                resolved_device = "cuda"
                logger.info("Device 'gpu' specified, CUDA is available. Using 'cuda'.")
            else:
                resolved_device = "cpu"
                logger.warning("Device 'gpu' specified, but CUDA is not available. Falling back to 'cpu'.")
        elif device is None:
             # Let CrossEncoder handle auto-detection if device is None
             logger.info("Device not specified, letting CrossEncoder auto-detect.")
             resolved_device = None # Ensure it's None for CrossEncoder's auto-detection
        else:
             # Use the provided device string directly if it's not 'gpu' or None
             logger.info(f"Using specified device: '{device}'")
             resolved_device = device # Keep the original value ('cpu', 'cuda', etc.)

        logger.info(f"Initializing BGE Reranker model '{model_name}' on device '{resolved_device or 'auto'}'")
        try:
            # device=None lets sentence-transformers handle auto-detection if not specified
            self.model = CrossEncoder(model_name, device=resolved_device, **kwargs)
            logger.info(f"BGE Reranker model '{model_name}' loaded successfully on device '{self.model.device}'.")
        except Exception as e:
            logger.error(f"Failed to load BGE Reranker model '{model_name}': {e}", exc_info=True)
            # Depending on desired behavior, could raise e or set self.model to None
            # For now, let's raise to make the failure explicit during startup
            raise

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Reranks documents using the BGE Cross-Encoder model.
        """
        if not documents:
            logger.debug("No documents provided for reranking.")
            return []
        
        if not hasattr(self, 'model') or self.model is None:
             logger.error("Reranker model was not loaded successfully. Cannot rerank.")
             # Return original documents or empty list? Let's return empty to signal failure.
             return []

        try:
            doc_texts = [doc.get("text", "") for doc in documents]
            pairs = [(query, doc_text) for doc_text in doc_texts]
            
            # Use _name_or_path which usually holds the model identifier
            model_identifier = getattr(self.model.config, '_name_or_path', 'unknown_model') 
            logger.debug(f"Reranking {len(pairs)} pairs for query: '{query[:50]}...' using {model_identifier}")
            scores = self.model.predict(pairs, show_progress_bar=False) # Disable progress bar for cleaner logs
            logger.debug(f"Reranking scores calculated.")

            # Add scores to documents and sort
            for idx, doc in enumerate(documents):
                doc['rerank_score'] = scores[idx]

            reranked_docs = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
            
            logger.debug(f"Returning top {min(top_n, len(reranked_docs))} reranked documents.")
            return reranked_docs[:top_n]
        
        except Exception as e:
            logger.error(f"Error during BGE reranking: {e}", exc_info=True)
            # Gracefully fallback: return original top_n documents or empty?
            # Let's return empty list to indicate reranking failed.
            return [] 