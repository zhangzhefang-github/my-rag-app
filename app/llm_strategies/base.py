from abc import ABC, abstractmethod
from langchain_core.messages import BaseMessage

class LLMStrategy(ABC):
    """Abstract base class for Large Language Model interaction strategies."""

    @abstractmethod
    def invoke(self, messages: list[BaseMessage] | list[tuple[str, str]]) -> str:
        """
        Invokes the LLM with the given messages and returns the content of the response.

        Args:
            messages: A list of messages in LangChain format (either BaseMessage objects or tuples).

        Returns:
            The string content of the LLM's response.
        """
        raise NotImplementedError

    # Optional: Define interface for asynchronous invocation if needed later
    # async def ainvoke(self, messages: list[BaseMessage] | list[tuple[str, str]]) -> str:
    #    raise NotImplementedError 