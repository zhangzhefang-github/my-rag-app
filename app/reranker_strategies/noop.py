# app/reranker_strategies/noop.py

from typing import List, Union, Dict
from .base import BaseRerankerStrategy
import logging

logger = logging.getLogger(__name__)

class NoOpRerankerStrategy(BaseRerankerStrategy):
    """
    A reranker strategy that does nothing (no-operation).
    It returns scores indicating no change in relevance (e.g., all zeros or decreasing numbers).
    Useful for disabling reranking gracefully or as a placeholder.
    """

    def __init__(self):
        logger.info("Initialized NoOpRerankerStrategy. Reranking will have no effect.")

    def rerank(self, query: str, passages: List[Union[str, Dict[str, str]]]) -> List[float]:
        """
        Returns a list of scores that effectively keep the original order.
        Using descending scores ensures that if sorting happens, the original order is preserved.

        Args:
            query: The query string (unused).
            passages: The list of passages (structure inspected but content ignored).

        Returns:
            A list of dummy scores (e.g., [N, N-1, ..., 1, 0]).
        """
        num_passages = len(passages)
        # Return descending scores to maintain original order if sorted descending
        # Or simply return zeros if sorting isn't guaranteed
        # Let's return zeros for simplicity, as the Retriever handles the top_n cutoff.
        scores = [0.0] * num_passages
        logger.debug(f"NoOpReranker called for {num_passages} passages. Returning zero scores.")
        return scores 