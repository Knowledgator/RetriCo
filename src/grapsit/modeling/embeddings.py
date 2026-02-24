"""Embedding model abstractions with lazy loading."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import logging

logger = logging.getLogger(__name__)


class BaseEmbeddingModel(ABC):
    """Abstract base for text embedding models."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...

    @abstractmethod
    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts into embedding vectors.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (one per input text).
        """
        ...


class SentenceTransformerEmbedding(BaseEmbeddingModel):
    """Sentence-transformers embedding model with lazy loading.

    Args:
        model_name: HuggingFace model name or path.
        device: Device to load model on (e.g. "cpu", "cuda").
        batch_size: Batch size for encoding.
    """

    def __init__(
        self,
        *,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._dimension: Optional[int] = None

    def _ensure_model(self):
        """Lazily load the sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers package required. "
                    "Install with: pip install sentence-transformers"
                )

            kwargs: Dict[str, Any] = {}
            if self.device is not None:
                kwargs["device"] = self.device

            self._model = SentenceTransformer(self.model_name, **kwargs)
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(
                f"SentenceTransformer loaded: {self.model_name} "
                f"(dim={self._dimension})"
            )

    @property
    def dimension(self) -> int:
        self._ensure_model()
        return self._dimension

    def encode(self, texts: List[str]) -> List[List[float]]:
        self._ensure_model()
        embeddings = self._model.encode(
            texts, batch_size=self.batch_size, show_progress_bar=False
        )
        return [emb.tolist() for emb in embeddings]


class OpenAIEmbedding(BaseEmbeddingModel):
    """OpenAI embedding model with lazy SDK loading.

    Supports any OpenAI-compatible embedding API via ``base_url``.

    Args:
        api_key: API key.
        base_url: API base URL (defaults to OpenAI).
        model: Embedding model name.
        dimensions: Output embedding dimensions (if supported by model).
        timeout: Request timeout in seconds.
        max_retries: Max retry attempts.
    """

    # Default dimensions for known models
    _KNOWN_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "text-embedding-3-small",
        dimensions: Optional[int] = None,
        timeout: float = 60.0,
        max_retries: int = 2,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self._requested_dimensions = dimensions
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
                    "openai package required for OpenAI embeddings. "
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
            logger.info(f"OpenAI embedding client initialized (model={self.model})")

    @property
    def dimension(self) -> int:
        if self._requested_dimensions is not None:
            return self._requested_dimensions
        if self.model in self._KNOWN_DIMENSIONS:
            return self._KNOWN_DIMENSIONS[self.model]
        raise ValueError(
            f"Unknown dimension for model {self.model!r}. "
            f"Pass dimensions= explicitly."
        )

    def encode(self, texts: List[str]) -> List[List[float]]:
        self._ensure_client()

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "input": texts,
        }
        if self._requested_dimensions is not None:
            kwargs["dimensions"] = self._requested_dimensions

        response = self._client.embeddings.create(**kwargs)
        return [item.embedding for item in response.data]


def create_embedding_model(config: dict) -> BaseEmbeddingModel:
    """Factory to create an embedding model from config.

    The ``embedding_method`` key selects the backend:
    - ``"sentence_transformer"`` (default): SentenceTransformerEmbedding
    - ``"openai"``: OpenAIEmbedding

    Remaining keys are passed as constructor kwargs.
    """
    config = dict(config)
    method = config.pop("embedding_method", "sentence_transformer")

    if method == "sentence_transformer":
        return SentenceTransformerEmbedding(**config)
    elif method == "openai":
        return OpenAIEmbedding(**config)
    else:
        raise ValueError(
            f"Unknown embedding_method: {method!r}. "
            f"Expected 'sentence_transformer' or 'openai'."
        )