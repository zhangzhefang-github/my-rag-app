# app/llm_strategies/custom_api_strategy.py
import logging
from typing import AsyncGenerator
from openai import OpenAI, OpenAIError, AsyncOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from .base import LLMStrategy

logger = logging.getLogger(__name__)

class CustomAPIStrategy(LLMStrategy):
    """LLM strategy implementation using a custom API endpoint via the OpenAI library."""

    def __init__(self,
                 api_key: str,
                 base_url: str,
                 model: str = "gpt-3.5-turbo", # Default model, can be overridden by env var
                 temperature: float = 0.7,
                 max_tokens: int | None = 1024):
        """
        Initializes the custom API strategy.

        Args:
            api_key: Your API key for the custom endpoint.
            base_url: The base URL of the custom API endpoint.
            model: The model to use at the custom endpoint.
            temperature: Sampling temperature for generation.
            max_tokens: Optional maximum number of tokens to generate.
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        try:
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            self.async_client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url
            )
            # Optional: You could add a quick check here, e.g., list models
            # self.client.models.list() 
            logger.info(f"CustomAPIStrategy initialized for model '{self.model}' at {base_url}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI clients for CustomAPIStrategy: {e}", exc_info=True)
            raise ValueError(f"Could not initialize Custom API clients: {e}") from e
            
    def _convert_messages(self, messages: list[BaseMessage] | list[tuple[str, str]]) -> list[dict]:
        """Converts LangChain messages to the format expected by openai.chat.completions.create."""
        output_messages = []
        for msg in messages:
            role = ""
            content = ""
            if isinstance(msg, BaseMessage):
                content = msg.content
                if isinstance(msg, HumanMessage):
                    role = "user"
                elif isinstance(msg, AIMessage):
                    role = "assistant"
                elif isinstance(msg, SystemMessage):
                    role = "system"
                else: # Generic BaseMessage, try to infer role or skip
                    logger.warning(f"Unsupported message type: {type(msg)}, attempting to treat as user.")
                    role = "user" # Default assumption or skip?
            elif isinstance(msg, tuple) and len(msg) == 2:
                role_str, content = msg
                role_str = role_str.lower()
                if role_str in ["user", "human"]:
                    role = "user"
                elif role_str in ["assistant", "ai"]:
                    role = "assistant"
                elif role_str == "system":
                    role = "system"
                else:
                     logger.warning(f"Unsupported role in tuple: {role_str}")
                     continue # Skip unsupported roles in tuples
            else:
                logger.warning(f"Skipping unsupported message format: {msg}")
                continue
                
            if role and content:
                 output_messages.append({"role": role, "content": content})
            
        return output_messages

    def invoke(self, messages: list[BaseMessage] | list[tuple[str, str]]) -> str:
        if not self.client:
             raise RuntimeError("Custom API client was not initialized successfully.")
             
        openai_messages = self._convert_messages(messages)
        if not openai_messages:
            logger.error("Could not convert any messages into the required format.")
            return "Error: Could not process input messages."
        
        try:
            logger.debug(f"Calling custom API chat completion with model: {self.model}")
            response = self.client.chat.completions.create(
                messages=openai_messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=False # Invoke is non-streaming
            )
            content = response.choices[0].message.content
            logger.debug("Custom API chat completion successful.")
            return content
        except OpenAIError as e:
            logger.error(f"Custom API Error during invoke: {e.status_code} - {e.body}", exc_info=True)
            # Reraise or return a specific error message
            raise RuntimeError(f"Custom API call failed: {e.body}") from e
        except Exception as e:
             logger.error(f"Unexpected error during Custom API invoke: {e}", exc_info=True)
             raise RuntimeError(f"Unexpected error during Custom API call: {e}") from e

    async def astream(self, messages: list[BaseMessage] | list[tuple[str, str]]) -> AsyncGenerator[str, None]:
        """Streams the response from the custom API endpoint.
        
        Args:
            messages: A list of messages in LangChain format.

        Yields:
            String chunks of the response content delta.
        """
        if not self.async_client:
            raise RuntimeError("Custom API async client was not initialized successfully.")

        openai_messages = self._convert_messages(messages)
        if not openai_messages:
            logger.error("Could not convert messages for streaming.")
            yield "[Error: Could not process input messages for streaming.]"
            return
            
        try:
            logger.debug(f"Calling custom API stream completion with model: {self.model}")
            stream = await self.async_client.chat.completions.create(
                messages=openai_messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content is not None:
                    yield content
            logger.debug("Custom API stream completion finished.")
        except OpenAIError as e:
            logger.error(f"Custom API Error during stream: {e.status_code} - {e.body}", exc_info=True)
            yield f"\n[Error during Custom API generation: {e.body}]"
        except Exception as e:
            logger.error(f"Unexpected error during Custom API stream: {e}", exc_info=True)
            yield f"\n[Unexpected error during Custom API generation: {e}]"

    # Note: Implementing streaming would require a different method 
    # that calls client.chat.completions.create(stream=True) and yields results. 