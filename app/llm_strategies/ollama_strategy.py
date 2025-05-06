# app/llm_strategies/ollama_strategy.py
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, AIMessageChunk
from .base import LLMStrategy
import logging
from typing import AsyncGenerator, List, Optional

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

    # Implement the stream_generate method required by the pipeline
    async def stream_generate(self, context_str: str, query: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Streams the response using the existing astream method, formatting the prompt internally."""
        
        # --- SPRINT 2: Revised Prompt Template (More Natural Tone) ---
        prompt_template = (
            "你是一个友好且乐于助人的问答机器人。请严格按照以下格式回答问题：\n"
            "1. **直接回答问题**，使用提供的上下文块。请在严格基于上下文回答的同时，保持友好和稍带对话感的语气。不要包含任何思考过程、前言或与问题无关的评论。\n"
            "2. 在你的回答中，使用 `[索引]` 标注信息来源的上下文块编号，例如：'天空是蓝色的 [0]。'\n"
            "3. 如果上下文中没有答案，请明确说明'抱歉，根据我所了解的信息，还无法回答您这个问题呢。'\n" # More friendly refusal
            "4. 回答完毕后，必须另起一行并只包含 `<<<CITATIONS>>>` 这个标记。\n"
            "5. 在标记的下一行，提供一个JSON列表，包含你答案中引用的具体文本和来源索引。严格遵循此格式: `[{{\"text_quote\": \"引用的文本...\", \"source_chunk_indices\": [0, 1]}}, ...]`\n"
            "6. **请务必只输出答案、标记和 JSON 列表，不要有任何其他多余内容。** 用中文回答。\n\n"
            "--- 上下文块 ---\n{context}\n\n"
            "--- 问题 ---\n{query}\n\n"
            "--- 回答 ---"
        )
        prompt = prompt_template.format(context=context_str, query=query)
        # --- End Revised Prompt Definition ---
        
        # Format for LangChain ChatOllama
        messages = [("human", prompt)]
        
        logger.debug(f"Streaming generation for Ollama with revised prompt (natural tone): {prompt[:300]}...")
        async for chunk in self.astream(messages):
            yield chunk

    # Optional: Implement asynchronous invocation if needed
    # async def ainvoke(self, messages: list[BaseMessage] | list[tuple[str, str]]) -> str:
    #    response = await self.llm.ainvoke(messages)
    #    return response.content 