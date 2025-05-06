import logging
import time
from typing import List, Tuple
from sentence_transformers.cross_encoder import CrossEncoder
from app.reranker_strategies.base import BaseRerankerStrategy
from utils.gpu_manager import GPUManager

logger = logging.getLogger(__name__)

class CrossEncoderReranker(BaseRerankerStrategy):
    \"\"\"Reranker strategy using Sentence Transformers CrossEncoder models.\"\"\"

    def __init__(self, model_name: str = \"BAAI/bge-reranker-base\", device: str = \"auto\"):
        \"\"\"
        Initializes the CrossEncoder reranker.

        Args:
            model_name: The name or path of the CrossEncoder model.
            device: The device to run the model on ('cpu', 'cuda', 'auto').
        \"\"\"
        self.gpu_manager = GPUManager()
        self._model_name = model_name
        # Determine the actual device using GPUManager
        self.torch_device = self.gpu_manager.get_torch_device_for_model(requested_device=device)
        logger.info(f\"Initializing CrossEncoderReranker with model: '{self._model_name}' on device: {self.torch_device}\")
        try:
            self.model = CrossEncoder(
                self._model_name,
                max_length=512, # Can be adjusted if needed
                device=self.torch_device,
                # automodel_args={'trust_remote_code': True} # May be needed for some models
            )
            logger.info(f\"CrossEncoder model '{self._model_name}' loaded successfully on {self.torch_device}.\")
        except Exception as e:
            logger.error(f\"Failed to load CrossEncoder model '{self._model_name}': {e}\", exc_info=True)
            raise RuntimeError(f\"Could not initialize CrossEncoder model: {e}\") from e

    def rerank(self, query: str, documents: List[str]) -> List[Tuple[int, float]]:
        \"\"\"
        Reranks documents using the loaded CrossEncoder model.

        Args:
            query: The query string.
            documents: A list of document texts.

        Returns:
            A list of (original_index, score) tuples, sorted by score descending.
        \"\"\"
        if not documents:
            return []

        logger.debug(f\"Reranking {len(documents)} documents for query: '{query[:50]}...'\" )
        start_time = time.time()

        # Prepare input pairs for the CrossEncoder
        sentence_pairs = [[query, doc] for doc in documents]

        # Predict scores
        try:
            # Use batch prediction
            scores = self.model.predict(sentence_pairs, show_progress_bar=False, convert_to_numpy=True)
            logger.debug(f\"Raw reranker scores: {scores}\")
        except Exception as e:
            logger.error(f\"CrossEncoder prediction failed: {e}\", exc_info=True)
            # Return empty list or raise error? Returning empty for now.
            return []

        # Combine original indices with scores
        # We need the index of the document in the *original* list
        indexed_scores = list(enumerate(scores)) # Creates [(0, score0), (1, score1), ...]

        # Sort by score in descending order
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        end_time = time.time()
        logger.debug(f\"Reranking took {end_time - start_time:.4f} seconds.\")
        # Log top results for debugging
        for i, (original_index, score) in enumerate(indexed_scores[:5]): # Log top 5
             logger.debug(f\"  Reranked {i+1}: Original Index={original_index}, Score={score:.4f}\")


        return indexed_scores # Return list of (original_index, score) sorted by score

    @property
    def model_name(self) -> str:
        \"\"\"Returns the name of the CrossEncoder model.\"\"\"
        return self._model_name 