"""OpenAI-compatible LLM client with lazy SDK loading."""

from typing import Any, Dict, List, Optional
import logging

from .base import BaseLLMClient

logger = logging.getLogger(__name__)


class OpenAIClient(BaseLLMClient):
    """OpenAI-compatible chat completion client.

    Supports any OpenAI-compatible API (OpenAI, vLLM, Ollama, etc.)
    via the ``base_url`` parameter.

    The ``openai`` package is lazily imported on first use.

    Args:
        api_key: API key (or "dummy" for local servers).
        base_url: API base URL. Defaults to OpenAI's API.
        model: Model name (default: "gpt-4o-mini").
        temperature: Default sampling temperature.
        max_completion_tokens: Default max tokens.
        timeout: Request timeout in seconds.
        max_retries: Max retry attempts.
    """

    def __init__(
        self,
        *,
        api_key: str = None,
        base_url: str = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_completion_tokens: int = 4096,
        timeout: float = 60.0,
        max_retries: int = 2,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.default_temperature = temperature
        self.default_max_completion_tokens = max_completion_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = None

    def _ensure_client(self):
        """Lazily create the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openai package required for LLM processors. "
                    "Install with: pip install openai"
                )

            kwargs: Dict[str, Any] = {
                "timeout": self.timeout,
                "max_retries": self.max_retries,
            }
            if self.api_key is not None:
                kwargs["api_key"] = self.api_key
            if self.base_url is not None:
                kwargs["base_url"] = self.base_url

            self._client = OpenAI(**kwargs)
            logger.info(f"OpenAI client initialized (model={self.model})")

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

        Returns:
            The assistant message content as a string.
        """
        self._ensure_client()

        api_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.default_temperature,
            "max_completion_tokens": max_completion_tokens if max_completion_tokens is not None else self.default_max_completion_tokens,
        }
        if response_format is not None:
            api_kwargs["response_format"] = response_format
        api_kwargs.update(kwargs)

        response = self._client.chat.completions.create(**api_kwargs)
        return response.choices[0].message.content
