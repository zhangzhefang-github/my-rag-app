from abc import ABC, abstractmethod
from typing import List, Tuple

class BaseRerankerStrategy(ABC):
    """
    Abstract base class for different document reranking strategies.
    """

    @abstractmethod
    def rerank(self, query: str, documents: List[str]) -> List[Tuple[int, float]]:
        """
        Reranks a list of documents based on a query.

        Args:
            query: The query string.
            documents: A list of document texts to be reranked.

        Returns:
            A list of tuples, where each tuple contains the original index
            of the document in the input list and its relevance score,
            sorted by relevance score in descending order.
            Example: [(2, 0.95), (0, 0.85), (1, 0.75)]
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Returns the name or identifier of the reranker model being used.
        """
        pass 