"""Tests for ChunkEmbedderProcessor."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_store():
    return MagicMock()


@pytest.fixture
def mock_embedding_model():
    model = MagicMock()
    type(model).dimension = PropertyMock(return_value=384)
    model.encode.return_value = [[0.1] * 384, [0.2] * 384]
    return model


@pytest.fixture
def mock_vector_store():
    return MagicMock()


@pytest.fixture
def sample_chunks():
    """Chunk-like objects with .id and .text."""
    c1 = MagicMock()
    c1.id = "chunk-1"
    c1.text = "Einstein was born in Ulm."
    c2 = MagicMock()
    c2.id = "chunk-2"
    c2.text = "He developed the theory of relativity."
    return [c1, c2]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestChunkEmbedder:

    def _make_processor(self, config=None):
        from grapsit.construct.chunk_embedder import ChunkEmbedderProcessor
        return ChunkEmbedderProcessor(config or {})

    def test_basic_embedding(self, mock_store, mock_embedding_model, mock_vector_store, sample_chunks):
        proc = self._make_processor({"vector_index_name": "chunk_embeddings"})
        proc._store = mock_store
        proc._embedding_model = mock_embedding_model
        proc._vector_store = mock_vector_store

        result = proc(chunks=sample_chunks)

        assert result["embedded_count"] == 2
        assert result["dimension"] == 384
        assert result["index_name"] == "chunk_embeddings"

        mock_vector_store.create_index.assert_called_once_with("chunk_embeddings", 384)
        mock_embedding_model.encode.assert_called_once_with(
            ["Einstein was born in Ulm.", "He developed the theory of relativity."]
        )
        mock_vector_store.store_embeddings.assert_called_once()
        assert mock_store.update_chunk_embedding.call_count == 2

    def test_empty_chunks(self, mock_store, mock_embedding_model, mock_vector_store):
        proc = self._make_processor()
        proc._store = mock_store
        proc._embedding_model = mock_embedding_model
        proc._vector_store = mock_vector_store

        result = proc(chunks=[])
        assert result["embedded_count"] == 0

        result = proc(chunks=None)
        assert result["embedded_count"] == 0

    def test_config_defaults(self):
        proc = self._make_processor({})
        assert proc.config_dict.get("vector_index_name") is None  # uses default in __call__

    def test_custom_index_name(self, mock_store, mock_embedding_model, mock_vector_store, sample_chunks):
        proc = self._make_processor({"vector_index_name": "my_chunks"})
        proc._store = mock_store
        proc._embedding_model = mock_embedding_model
        proc._vector_store = mock_vector_store

        result = proc(chunks=sample_chunks)
        assert result["index_name"] == "my_chunks"
        mock_vector_store.create_index.assert_called_once_with("my_chunks", 384)

    def test_store_persistence_failure_does_not_raise(
        self, mock_store, mock_embedding_model, mock_vector_store, sample_chunks,
    ):
        mock_store.update_chunk_embedding.side_effect = NotImplementedError("not supported")
        proc = self._make_processor()
        proc._store = mock_store
        proc._embedding_model = mock_embedding_model
        proc._vector_store = mock_vector_store

        # Should not raise
        result = proc(chunks=sample_chunks)
        assert result["embedded_count"] == 2

    @patch("grapsit.construct.chunk_embedder.create_store")
    @patch("grapsit.construct.chunk_embedder.create_embedding_model")
    @patch("grapsit.construct.chunk_embedder.create_vector_store")
    def test_lazy_initialization(self, mock_cvs, mock_cem, mock_cs, sample_chunks):
        mock_em = MagicMock()
        type(mock_em).dimension = PropertyMock(return_value=128)
        mock_em.encode.return_value = [[0.5] * 128, [0.6] * 128]
        mock_cem.return_value = mock_em
        mock_vs = MagicMock()
        mock_cvs.return_value = mock_vs
        mock_cs.return_value = MagicMock()

        proc = self._make_processor({
            "embedding_method": "sentence_transformer",
            "model_name": "all-MiniLM-L6-v2",
            "vector_store_type": "in_memory",
            "store_type": "neo4j",
        })

        result = proc(chunks=sample_chunks)
        assert result["embedded_count"] == 2
        mock_cs.assert_called_once()
        mock_cem.assert_called_once()
        mock_cvs.assert_called_once()

    def test_processor_registration(self):
        from grapsit.core.registry import processor_registry
        assert "chunk_embedder" in processor_registry._factories


class TestChunkEmbedderBuilder:
    """Test BuildConfigBuilder integration with chunk_embedder."""

    def test_builder_adds_chunk_embedder_node(self):
        from grapsit.core.builders import BuildConfigBuilder

        builder = BuildConfigBuilder(name="test")
        builder.ner_gliner(labels=["person"])
        builder.graph_writer()
        builder.chunk_embedder()

        config = builder.get_config()
        node_ids = [n["id"] for n in config["nodes"]]
        assert "chunk_embedder" in node_ids

        embedder_node = [n for n in config["nodes"] if n["id"] == "chunk_embedder"][0]
        assert embedder_node["requires"] == ["graph_writer"]
        assert "chunks" in embedder_node["inputs"]
        assert embedder_node["inputs"]["chunks"]["source"] == "chunker_result"

    def test_builder_without_embedder_has_no_node(self):
        from grapsit.core.builders import BuildConfigBuilder

        builder = BuildConfigBuilder(name="test")
        builder.ner_gliner(labels=["person"])
        builder.graph_writer()

        config = builder.get_config()
        node_ids = [n["id"] for n in config["nodes"]]
        assert "chunk_embedder" not in node_ids

    def test_embedder_inherits_store_params(self):
        from grapsit.core.builders import BuildConfigBuilder

        builder = BuildConfigBuilder(name="test")
        builder.ner_gliner(labels=["person"])
        builder.graph_writer(neo4j_uri="bolt://custom:7687", store_type="neo4j")
        builder.chunk_embedder(embedding_method="openai")

        config = builder.get_config()
        embedder_node = [n for n in config["nodes"] if n["id"] == "chunk_embedder"][0]
        assert embedder_node["config"]["store_type"] == "neo4j"
        assert embedder_node["config"]["neo4j_uri"] == "bolt://custom:7687"
        assert embedder_node["config"]["embedding_method"] == "openai"
