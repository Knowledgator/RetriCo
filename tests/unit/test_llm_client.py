"""Tests for OpenAI LLM client."""

import pytest
from unittest.mock import MagicMock, patch

from grapsit.llm.base import BaseLLMClient
from grapsit.llm.openai_client import OpenAIClient


class TestBaseLLMClient:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            BaseLLMClient()


class TestOpenAIClient:
    def test_lazy_loading(self):
        """Client is not created until first use."""
        client = OpenAIClient(api_key="test-key", model="gpt-4o-mini")
        assert client._client is None

    @patch("grapsit.llm.openai_client.OpenAIClient._ensure_client")
    def test_complete(self, mock_ensure):
        """complete() calls the underlying OpenAI API."""
        client = OpenAIClient(api_key="test-key", model="gpt-4o-mini")

        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"entities": []}'
        mock_openai.chat.completions.create.return_value = mock_response
        client._client = mock_openai

        result = client.complete(
            [{"role": "user", "content": "test"}],
            response_format={"type": "json_object"},
        )

        assert result == '{"entities": []}'
        mock_openai.chat.completions.create.assert_called_once()
        call_kwargs = mock_openai.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs["response_format"] == {"type": "json_object"}

    @patch("grapsit.llm.openai_client.OpenAIClient._ensure_client")
    def test_temperature_override(self, mock_ensure):
        """Temperature can be overridden per call."""
        client = OpenAIClient(api_key="test-key", temperature=0.1)
        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"
        mock_openai.chat.completions.create.return_value = mock_response
        client._client = mock_openai

        client.complete(
            [{"role": "user", "content": "test"}],
            temperature=0.7,
        )

        call_kwargs = mock_openai.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.7

    def test_config_stored(self):
        """Config values are stored correctly."""
        client = OpenAIClient(
            api_key="key123",
            base_url="http://localhost:8000/v1",
            model="llama3",
            temperature=0.5,
            max_completion_tokens=2048,
            timeout=30.0,
            max_retries=3,
        )
        assert client.api_key == "key123"
        assert client.base_url == "http://localhost:8000/v1"
        assert client.model == "llama3"
        assert client.default_temperature == 0.5
        assert client.default_max_completion_tokens == 2048
        assert client.timeout == 30.0
        assert client.max_retries == 3
