from abc import ABC, abstractmethod
from typing import List, Union, Dict

class BaseRerankerStrategy(ABC):
    """
    Abstract base class for different document reranking strategies.
    """

    @abstractmethod
    def rerank(self, query: str, passages: List[Union[str, Dict[str, str]]]) -> List[float]:
        """
        Reranks a list of passages based on their relevance to the query.

        Args:
            query: The original query string.
            passages: A list of passage texts (str) or dictionaries containing text
                      (e.g., {'text': 'passage content'}).

        Returns:
            A list of relevance scores (float), corresponding to the input passages.
            Higher scores indicate higher relevance.
            The order of scores should match the order of the input passages.
        """
        pass 