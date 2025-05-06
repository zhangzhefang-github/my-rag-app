# app/chunking_strategies/base.py

from abc import ABC, abstractmethod
from typing import List
# Import the core data models
from app.models.document import ParsedDocument, Chunk

class BaseChunkingStrategy(ABC):
    """
    Abstract base class for different document chunking strategies.
    """

    @abstractmethod
    def chunk(self, document: ParsedDocument) -> List[Chunk]:
        """
        Splits a ParsedDocument into a list of Chunks.

        Args:
            document: The ParsedDocument object containing the text and metadata.

        Returns:
            A list of Chunk objects, each representing a portion of the original document,
            complete with populated metadata (doc_id, chunk_index, offset, etc.).
        """
        pass

    # You might add other common methods or properties here if needed later,
    # e.g., methods to calculate estimated token counts, etc. 