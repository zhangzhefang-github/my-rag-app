from abc import ABC, abstractmethod
from typing import List
import asyncio
import logging

logger = logging.getLogger(__name__)

class EmbeddingStrategy(ABC):
    """Abstract base class for different embedding strategies."""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a list of texts synchronously."""
        pass

    @abstractmethod
    async def aembed(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a list of texts asynchronously."""
        pass

    async def _run_sync_in_async(self, texts: List[str]) -> List[List[float]]:
        """Helper method to run synchronous embed in an async context."""
        try:
            loop = asyncio.get_running_loop()
            # Use default executor (ThreadPoolExecutor)
            # Consider ProcessPoolExecutor for heavy CPU-bound tasks if necessary
            result = await loop.run_in_executor(None, self.embed, texts)
            return result
        except Exception as e:
            logger.error(f"Error running sync embed in async executor: {e}", exc_info=True)
            raise RuntimeError(f"Failed to run sync embed in async executor: {e}") from e 