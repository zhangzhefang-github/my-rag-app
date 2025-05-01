# app/llm_strategies/ollama_strategy.py
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, AIMessageChunk
from .base import LLMStrategy
import logging
from typing import AsyncGenerator

logger = logging.getLogger(__name__)

class OllamaStrategy(LLMStrategy):
    """LLM strategy implementation using a local Ollama service via LangChain."""

    def __init__(self,
                 host: str,
                 model: str = "qwen3:0.6b",
                 temperature: float = 0.7,
                 num_predict: int | None = 512):
        """
        Initializes the Ollama strategy.

        Args:
            host: The base URL of the Ollama service (e.g., 'http://localhost:11500').
            model: The specific Ollama model to use.
            temperature: Sampling temperature for generation.
            num_predict: Optional maximum number of tokens to predict (Ollama specific).
        """
        logger.info(f"Initializing OllamaStrategy with host: {host}, model: {model}")
        self.llm = ChatOllama(
            base_url=host,
            model=model,
            temperature=temperature,
            num_predict=num_predict,
            # You might want to add other parameters like request_timeout if needed
        )

    def invoke(self, messages: list[BaseMessage] | list[tuple[str, str]]) -> str:
        response = self.llm.invoke(messages)
        return response.content

    async def astream(self, messages: list[BaseMessage] | list[tuple[str, str]]) -> AsyncGenerator[str, None]:
        """Streams the response from the Ollama model.

        Args:
            messages: A list of messages in LangChain format.

        Yields:
            String chunks of the response content.
        """
        try:
            # Use the stream method provided by ChatOllama
            async for chunk in self.llm.astream(messages):
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"Error during Ollama stream: {e}", exc_info=True)
            # Yield an error message chunk or raise?
            yield f"\n[Error during Ollama generation: {e}]"
            # Or re-raise: raise RuntimeError(f"Ollama stream failed: {e}") from e

    # Optional: Implement asynchronous invocation if needed
    # async def ainvoke(self, messages: list[BaseMessage] | list[tuple[str, str]]) -> str:
    #    response = await self.llm.ainvoke(messages)
    #    return response.content 