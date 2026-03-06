"""Tests for EntityEmbedderProcessor."""

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
    model.encode.return_value = [[0.1] * 384, [0.2] * 384, [0.3] * 384]
    return model


@pytest.fixture
def mock_vector_store():
    return MagicMock()


@pytest.fixture
def sample_entity_map():
    """Entity-like objects with .id and .label."""
    e1 = MagicMock()
    e1.id = "ent-1"
    e1.label = "Einstein"
    e2 = MagicMock()
    e2.id = "ent-2"
    e2.label = "Ulm"
    e3 = MagicMock()
    e3.id = "ent-3"
    e3.label = "Physics"
    return {"einstein": e1, "ulm": e2, "physics": e3}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEntityEmbedder:

    def _make_processor(self, config=None):
        from retrico.construct.entity_embedder import EntityEmbedderProcessor
        return EntityEmbedderProcessor(config or {})

    def test_basic_embedding(self, mock_store, mock_embedding_model, mock_vector_store, sample_entity_map):
        proc = self._make_processor({"vector_index_name": "entity_embeddings"})
        proc._store = mock_store
        proc._embedding_model = mock_embedding_model
        proc._vector_store = mock_vector_store

        result = proc(entity_map=sample_entity_map)

        assert result["embedded_count"] == 3
        assert result["dimension"] == 384
        assert result["index_name"] == "entity_embeddings"

        mock_vector_store.create_index.assert_called_once_with("entity_embeddings", 384)
        mock_embedding_model.encode.assert_called_once()
        # Verify all labels were encoded
        encoded_texts = mock_embedding_model.encode.call_args[0][0]
        assert set(encoded_texts) == {"Einstein", "Ulm", "Physics"}

        mock_vector_store.store_embeddings.assert_called_once()
        assert mock_store.update_entity_embedding.call_count == 3

    def test_empty_entity_map(self, mock_store, mock_embedding_model, mock_vector_store):
        proc = self._make_processor()
        proc._store = mock_store
        proc._embedding_model = mock_embedding_model
        proc._vector_store = mock_vector_store

        result = proc(entity_map={})
        assert result["embedded_count"] == 0

        result = proc(entity_map=None)
        assert result["embedded_count"] == 0

    def test_custom_index_name(self, mock_store, mock_embedding_model, mock_vector_store, sample_entity_map):
        proc = self._make_processor({"vector_index_name": "my_entities"})
        proc._store = mock_store
        proc._embedding_model = mock_embedding_model
        proc._vector_store = mock_vector_store

        result = proc(entity_map=sample_entity_map)
        assert result["index_name"] == "my_entities"
        mock_vector_store.create_index.assert_called_once_with("my_entities", 384)

    def test_store_persistence_failure_does_not_raise(
        self, mock_store, mock_embedding_model, mock_vector_store, sample_entity_map,
    ):
        mock_store.update_entity_embedding.side_effect = NotImplementedError("not supported")
        proc = self._make_processor()
        proc._store = mock_store
        proc._embedding_model = mock_embedding_model
        proc._vector_store = mock_vector_store

        # Should not raise
        result = proc(entity_map=sample_entity_map)
        assert result["embedded_count"] == 3

    @patch("retrico.construct.entity_embedder.resolve_from_pool_or_create")
    @patch("retrico.construct.entity_embedder.create_embedding_model")
    def test_lazy_initialization(self, mock_cem, mock_resolve, sample_entity_map):
        mock_em = MagicMock()
        type(mock_em).dimension = PropertyMock(return_value=128)
        mock_em.encode.return_value = [[0.5] * 128, [0.6] * 128, [0.7] * 128]
        mock_cem.return_value = mock_em
        mock_graph = MagicMock()
        mock_vs = MagicMock()
        mock_resolve.side_effect = lambda cfg, cat: mock_graph if cat == "graph" else mock_vs

        proc = self._make_processor({
            "embedding_method": "sentence_transformer",
            "model_name": "all-MiniLM-L6-v2",
            "vector_store_type": "in_memory",
            "store_type": "neo4j",
        })

        result = proc(entity_map=sample_entity_map)
        assert result["embedded_count"] == 3
        assert mock_resolve.call_count == 2  # graph + vector
        mock_cem.assert_called_once()

    def test_processor_registration(self):
        from retrico.core.registry import processor_registry
        assert "entity_embedder" in processor_registry._factories


class TestEntityEmbedderBuilder:
    """Test RetriCoBuilder integration with entity_embedder."""

    def test_builder_adds_entity_embedder_node(self):
        from retrico.core.builders import RetriCoBuilder

        builder = RetriCoBuilder(name="test")
        builder.ner_gliner(labels=["person"])
        builder.graph_writer()
        builder.entity_embedder()

        config = builder.get_config()
        node_ids = [n["id"] for n in config["nodes"]]
        assert "entity_embedder" in node_ids

        embedder_node = [n for n in config["nodes"] if n["id"] == "entity_embedder"][0]
        assert embedder_node["requires"] == ["graph_writer"]
        assert "entity_map" in embedder_node["inputs"]
        assert embedder_node["inputs"]["entity_map"]["source"] == "writer_result"

    def test_builder_both_embedders(self):
        from retrico.core.builders import RetriCoBuilder

        builder = RetriCoBuilder(name="test")
        builder.ner_gliner(labels=["person"])
        builder.graph_writer()
        builder.chunk_embedder()
        builder.entity_embedder()

        config = builder.get_config()
        node_ids = [n["id"] for n in config["nodes"]]
        assert "chunk_embedder" in node_ids
        assert "entity_embedder" in node_ids

        # Both should require graph_writer
        for nid in ("chunk_embedder", "entity_embedder"):
            node = [n for n in config["nodes"] if n["id"] == nid][0]
            assert "graph_writer" in node["requires"]

    def test_embedder_inherits_store_params(self):
        from retrico.core.builders import RetriCoBuilder

        builder = RetriCoBuilder(name="test")
        builder.ner_gliner(labels=["person"])
        builder.graph_writer(neo4j_uri="bolt://custom:7687", store_type="neo4j")
        builder.entity_embedder(embedding_method="openai")

        config = builder.get_config()
        embedder_node = [n for n in config["nodes"] if n["id"] == "entity_embedder"][0]
        assert embedder_node["config"]["store_type"] == "neo4j"
        assert embedder_node["config"]["neo4j_uri"] == "bolt://custom:7687"
        assert embedder_node["config"]["embedding_method"] == "openai"
