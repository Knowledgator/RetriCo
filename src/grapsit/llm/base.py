"""Base class for LLM clients and built-in graph query tools."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

# Re-export tool definitions and Cypher translation from tools module
from .tools import (  # noqa: F401
    GRAPH_TOOLS,
    PROPERTY_FILTER_SCHEMA,
    tool_call_to_cypher,
    register_tool_translator,
    build_graph_schema_prompt,
)


class BaseLLMClient(ABC):
    """Abstract base for LLM API clients.

    All LLM clients must implement ``complete()`` which takes a list of
    chat messages and returns the assistant's text response.
    """

    @abstractmethod
    def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_completion_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> str:
        """Send a chat completion request.

        Args:
            messages: List of ``{"role": ..., "content": ...}`` dicts.
            temperature: Sampling temperature override.
            max_completion_tokens: Max tokens override.
            response_format: E.g. ``{"type": "json_object"}``.

        Returns:
            The assistant message text.
        """
        ...

    def complete_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        temperature: Optional[float] = None,
        max_completion_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Send a chat completion request with tool/function calling.

        When *tools* is ``None``, the built-in ``GRAPH_TOOLS`` are used.
        Pass a list to supply custom tools, or combine with the built-ins::

            client.complete_with_tools(msgs, tools=GRAPH_TOOLS + my_tools)

        Args:
            messages: Chat messages (may include tool results).
            tools: Tool definitions in OpenAI function-calling format.
                Defaults to ``GRAPH_TOOLS``.
            temperature: Sampling temperature override.
            max_completion_tokens: Max tokens override.

        Returns:
            ``{"content": str | None,
              "tool_calls": [{"id": str, "name": str, "arguments": dict}, ...]}``
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support tool/function calling."
        )
