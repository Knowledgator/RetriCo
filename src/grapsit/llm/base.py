"""Base class for LLM clients."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


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
