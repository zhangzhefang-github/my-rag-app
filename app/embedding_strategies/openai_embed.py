import logging
from typing import List
from openai import OpenAI, AsyncOpenAI, OpenAIError

from .base import EmbeddingStrategy

logger = logging.getLogger(__name__)

class OpenAIEmbedder(EmbeddingStrategy):
    """Embedding strategy using the OpenAI API."""

    def __init__(self, model: str, api_key: str, base_url: str | None = None, **kwargs):
        self.model = model
        try:
            self.client = OpenAI(api_key=api_key, base_url=base_url, **kwargs)
            self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url, **kwargs)
            # You might want to add a quick check, like listing models, if feasible
            logger.info(f"OpenAIEmbedder initialized with model: {model}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI clients: {e}", exc_info=True)
            raise ValueError(f"Could not initialize OpenAI clients: {e}") from e

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not self.client:
            raise RuntimeError("OpenAI client not initialized.")
        # Replace newlines for OpenAI requirement, see:
        # https://community.openai.com/t/api-error-input-string-cannot-contain-newlines/341951/2
        processed_texts = [text.replace("\n", " ") for text in texts]
        try:
            response = self.client.embeddings.create(model=self.model, input=processed_texts)
            return [d.embedding for d in response.data]
        except OpenAIError as e:
            logger.error(f"OpenAI API error during embed: {e}", exc_info=True)
            raise RuntimeError(f"OpenAI API call failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during OpenAI embed: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error during OpenAI embedding: {e}") from e

    async def aembed(self, texts: List[str]) -> List[List[float]]:
        if not self.async_client:
            raise RuntimeError("OpenAI async client not initialized.")
        processed_texts = [text.replace("\n", " ") for text in texts]
        try:
            response = await self.async_client.embeddings.create(model=self.model, input=processed_texts)
            return [d.embedding for d in response.data]
        except OpenAIError as e:
            logger.error(f"OpenAI API error during async embed: {e}", exc_info=True)
            raise RuntimeError(f"Async OpenAI API call failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during async OpenAI embed: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error during async OpenAI embedding: {e}") from e 