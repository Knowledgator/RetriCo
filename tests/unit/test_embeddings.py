"""Unit tests for embedding model abstractions."""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np


class TestSentenceTransformerEmbedding:
    """Tests for SentenceTransformerEmbedding with mocked sentence_transformers."""

    def _make_mock_st(self, dim=384):
        """Create a mock SentenceTransformer class."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = dim
        mock_model.encode.side_effect = lambda texts, **kw: np.random.rand(
            len(texts), dim
        ).astype(np.float32)

        mock_cls = MagicMock(return_value=mock_model)
        return mock_cls, mock_model

    def test_lazy_loading(self):
        """Model is not loaded until first use."""
        mock_cls, mock_model = self._make_mock_st()

        with patch.dict("sys.modules", {"sentence_transformers": MagicMock()}):
            with patch(
                "grapsit.modeling.embeddings.SentenceTransformerEmbedding._ensure_model"
            ) as mock_ensure:
                from grapsit.modeling.embeddings import SentenceTransformerEmbedding

                emb = SentenceTransformerEmbedding(model_name="test-model")
                assert emb._model is None
                mock_ensure.assert_not_called()

    def test_encode_returns_list_of_lists(self):
        """encode() returns List[List[float]]."""
        mock_cls, mock_model = self._make_mock_st(dim=128)

        mock_st_module = MagicMock()
        mock_st_module.SentenceTransformer = mock_cls

        with patch.dict("sys.modules", {"sentence_transformers": mock_st_module}):
            from grapsit.modeling.embeddings import SentenceTransformerEmbedding

            emb = SentenceTransformerEmbedding(model_name="test-model")
            result = emb.encode(["hello", "world"])

            assert len(result) == 2
            assert all(isinstance(v, list) for v in result)
            assert all(len(v) == 128 for v in result)
            assert all(isinstance(v[0], float) for v in result)

    def test_dimension_property(self):
        """dimension property returns correct value after loading."""
        mock_cls, mock_model = self._make_mock_st(dim=256)

        mock_st_module = MagicMock()
        mock_st_module.SentenceTransformer = mock_cls

        with patch.dict("sys.modules", {"sentence_transformers": mock_st_module}):
            from grapsit.modeling.embeddings import SentenceTransformerEmbedding

            emb = SentenceTransformerEmbedding(model_name="test-model")
            assert emb.dimension == 256

    def test_device_passed(self):
        """Device kwarg is forwarded to SentenceTransformer."""
        mock_cls, mock_model = self._make_mock_st()

        mock_st_module = MagicMock()
        mock_st_module.SentenceTransformer = mock_cls

        with patch.dict("sys.modules", {"sentence_transformers": mock_st_module}):
            from grapsit.modeling.embeddings import SentenceTransformerEmbedding

            emb = SentenceTransformerEmbedding(model_name="test-model", device="cpu")
            emb.encode(["test"])
            mock_cls.assert_called_once_with("test-model", device="cpu")

    def test_import_error(self):
        """ImportError raised when sentence_transformers not installed."""
        import sys

        # Temporarily remove the module if it exists
        saved = sys.modules.pop("sentence_transformers", None)
        try:
            with patch.dict("sys.modules", {"sentence_transformers": None}):
                from grapsit.modeling.embeddings import SentenceTransformerEmbedding

                emb = SentenceTransformerEmbedding()
                with pytest.raises(ImportError, match="sentence-transformers"):
                    emb.encode(["test"])
        finally:
            if saved is not None:
                sys.modules["sentence_transformers"] = saved


class TestOpenAIEmbedding:
    """Tests for OpenAIEmbedding with mocked openai SDK."""

    def _make_mock_response(self, texts, dim=1536):
        """Create a mock embeddings.create response."""
        mock_response = MagicMock()
        mock_data = []
        for i in range(len(texts)):
            item = MagicMock()
            item.embedding = np.random.rand(dim).tolist()
            mock_data.append(item)
        mock_response.data = mock_data
        return mock_response

    def test_lazy_loading(self):
        """Client is not created until first encode()."""
        from grapsit.modeling.embeddings import OpenAIEmbedding

        emb = OpenAIEmbedding(api_key="test-key")
        assert emb._client is None

    def test_encode_returns_embeddings(self):
        """encode() returns correct shape from mocked API."""
        mock_openai = MagicMock()
        mock_client = MagicMock()

        texts = ["hello", "world"]
        mock_response = self._make_mock_response(texts, dim=1536)
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai}):
            from grapsit.modeling.embeddings import OpenAIEmbedding

            emb = OpenAIEmbedding(api_key="test-key")
            result = emb.encode(texts)

            assert len(result) == 2
            assert all(isinstance(v, list) for v in result)
            assert all(len(v) == 1536 for v in result)

    def test_dimension_known_model(self):
        """dimension returns known value without API call."""
        from grapsit.modeling.embeddings import OpenAIEmbedding

        emb = OpenAIEmbedding(model="text-embedding-3-small")
        assert emb.dimension == 1536

        emb2 = OpenAIEmbedding(model="text-embedding-3-large")
        assert emb2.dimension == 3072

    def test_dimension_custom(self):
        """Explicit dimensions override defaults."""
        from grapsit.modeling.embeddings import OpenAIEmbedding

        emb = OpenAIEmbedding(model="text-embedding-3-small", dimensions=512)
        assert emb.dimension == 512

    def test_dimension_unknown_model_no_explicit(self):
        """ValueError for unknown model without explicit dimensions."""
        from grapsit.modeling.embeddings import OpenAIEmbedding

        emb = OpenAIEmbedding(model="some-custom-model")
        with pytest.raises(ValueError, match="Unknown dimension"):
            _ = emb.dimension

    def test_dimensions_passed_to_api(self):
        """dimensions kwarg is forwarded to API call."""
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = self._make_mock_response(
            ["test"], dim=256
        )
        mock_openai.OpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai}):
            from grapsit.modeling.embeddings import OpenAIEmbedding

            emb = OpenAIEmbedding(api_key="key", dimensions=256)
            emb.encode(["test"])

            call_kwargs = mock_client.embeddings.create.call_args
            assert call_kwargs[1]["dimensions"] == 256

    def test_import_error(self):
        """ImportError raised when openai not installed."""
        import sys

        saved = sys.modules.pop("openai", None)
        try:
            with patch.dict("sys.modules", {"openai": None}):
                from grapsit.modeling.embeddings import OpenAIEmbedding

                emb = OpenAIEmbedding(api_key="key")
                with pytest.raises(ImportError, match="openai"):
                    emb.encode(["test"])
        finally:
            if saved is not None:
                sys.modules["openai"] = saved


class TestCreateEmbeddingModel:
    """Tests for the create_embedding_model factory."""

    def test_default_sentence_transformer(self):
        from grapsit.modeling.embeddings import (
            SentenceTransformerEmbedding,
            create_embedding_model,
        )

        model = create_embedding_model({})
        assert isinstance(model, SentenceTransformerEmbedding)

    def test_sentence_transformer_explicit(self):
        from grapsit.modeling.embeddings import (
            SentenceTransformerEmbedding,
            create_embedding_model,
        )

        model = create_embedding_model(
            {"embedding_method": "sentence_transformer", "model_name": "custom-model"}
        )
        assert isinstance(model, SentenceTransformerEmbedding)
        assert model.model_name == "custom-model"

    def test_openai(self):
        from grapsit.modeling.embeddings import OpenAIEmbedding, create_embedding_model

        model = create_embedding_model(
            {"embedding_method": "openai", "api_key": "test", "model": "text-embedding-3-small"}
        )
        assert isinstance(model, OpenAIEmbedding)
        assert model.api_key == "test"

    def test_unknown_method(self):
        from grapsit.modeling.embeddings import create_embedding_model

        with pytest.raises(ValueError, match="Unknown embedding_method"):
            create_embedding_model({"embedding_method": "cohere"})
