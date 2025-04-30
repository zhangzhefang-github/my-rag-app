from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from .base import LLMStrategy

class OpenAIStrategy(LLMStrategy):
    """LLM strategy implementation using OpenAI API via LangChain."""

    def __init__(self,
                 api_key: str,
                 base_url: str | None = None,
                 model: str = "gpt-4o",
                 temperature: float = 0.7,
                 max_tokens: int | None = 1024):
        """
        Initializes the OpenAI strategy.

        Args:
            api_key: Your OpenAI API key.
            base_url: Optional custom base URL for the OpenAI API.
            model: The specific OpenAI model to use.
            temperature: Sampling temperature for generation.
            max_tokens: Optional maximum number of tokens to generate.
        """
        self.llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            openai_api_base=base_url,  # Use openai_api_base for custom URLs
            temperature=temperature,
            max_tokens=max_tokens,
            # You might want to add other parameters like request_timeout if needed
        )

    def invoke(self, messages: list[BaseMessage] | list[tuple[str, str]]) -> str:
        response = self.llm.invoke(messages)
        return response.content

    # Optional: Implement asynchronous invocation
    # async def ainvoke(self, messages: list[BaseMessage] | list[tuple[str, str]]) -> str:
    #    response = await self.llm.ainvoke(messages)
    #    return response.content 