import logging
import json
from typing import List
import requests
import aiohttp
import asyncio

from .base import EmbeddingStrategy

logger = logging.getLogger(__name__)

class OllamaEmbedder(EmbeddingStrategy):
    """Embedding strategy using a locally running Ollama instance."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434", request_timeout: int = 60):
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/api/embeddings"
        self.request_timeout = request_timeout
        self._check_connection()

    def _check_connection(self):
        """Checks if the Ollama server is reachable."""
        try:
            # Check if the base URL is reachable
            response = requests.get(self.base_url, timeout=5) # Short timeout for check
            response.raise_for_status()
            # Optionally, check if the specific model is available
            # res_models = requests.get(f"{self.base_url}/api/tags")
            # if self.model not in [m['name'] for m in res_models.json().get('models', [])]:
            #    logger.warning(f"Ollama model '{self.model}' not found at {self.base_url}. It might be pulled on first use.")
            logger.info(f"OllamaEmbedder initialized for model '{self.model}' at {self.base_url}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Could not connect to Ollama at {self.base_url}: {e}")
            # Decide if this should be a hard error or just a warning
            # raise ConnectionError(f"Could not connect to Ollama at {self.base_url}") from e

    def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        try:
            with requests.Session() as session:
                for text in texts:
                    payload = {"model": self.model, "prompt": text}
                    response = session.post(self.api_url, json=payload, timeout=self.request_timeout)
                    response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                    result = response.json()
                    if 'embedding' not in result:
                        raise ValueError("Ollama API response missing 'embedding' key.")
                    embeddings.append(result['embedding'])
            if len(embeddings) != len(texts):
                 logger.warning(f"Ollama embed mismatch: requested {len(texts)}, got {len(embeddings)}")
            return embeddings
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API request failed: {e}", exc_info=True)
            raise RuntimeError(f"Ollama API request failed: {e}") from e
        except (json.JSONDecodeError, ValueError, KeyError) as e:
             logger.error(f"Error processing Ollama response: {e}", exc_info=True)
             raise ValueError(f"Error processing Ollama response: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during Ollama embed: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error during Ollama embedding: {e}") from e

    async def _aembed_single(self, text: str, session: aiohttp.ClientSession) -> List[float]:
        """Helper to embed a single text asynchronously."""
        payload = {"model": self.model, "prompt": text}
        try:
            async with session.post(self.api_url, json=payload, timeout=self.request_timeout) as response:
                response.raise_for_status()
                result = await response.json()
                if 'embedding' not in result:
                    raise ValueError("Ollama API response missing 'embedding' key.")
                return result['embedding']
        except aiohttp.ClientError as e:
            logger.error(f"Ollama async API request failed for text chunk: {e}")
            # Depending on requirements, you might want to return None or a zero vector
            raise RuntimeError(f"Ollama async API request failed: {e}") from e
        except (json.JSONDecodeError, ValueError, KeyError) as e:
             logger.error(f"Error processing Ollama async response: {e}")
             raise ValueError(f"Error processing Ollama async response: {e}") from e


    async def aembed(self, texts: List[str]) -> List[List[float]]:
        # Ollama API typically doesn't support batching for embeddings in one request
        # So we send requests concurrently.
        conn = aiohttp.TCPConnector(limit_per_host=10) # Limit concurrency
        timeout = aiohttp.ClientTimeout(total=self.request_timeout)
        async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
            tasks = [self._aembed_single(text, session) for text in texts]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results, handling potential errors from gather
        final_embeddings = []
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                logger.error(f"Failed to get embedding for text index {i}: {res}")
                # Option 1: Raise the first exception encountered
                # raise res
                # Option 2: Return partial results with placeholders (e.g., None or zeros)
                # final_embeddings.append(None) # Or a list of zeros of expected dim
                # Option 3: Raise a summary error (current choice)
                raise RuntimeError(f"Failed to get embedding for one or more texts. First error: {res}") from res
            else:
                final_embeddings.append(res)

        if len(final_embeddings) != len(texts):
            logger.warning(f"Ollama async embed mismatch: requested {len(texts)}, got {len(final_embeddings)}")
            # Handle mismatch if necessary

        return final_embeddings 