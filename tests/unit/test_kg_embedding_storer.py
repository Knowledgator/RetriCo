"""Tests for KG embedding storer processor."""

import json
import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeTensor:
    """Fake tensor that supports .detach().cpu().numpy()."""

    def __init__(self, data):
        self._data = np.array(data, dtype=np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._data


class FakeRepresentation:
    """Simulates a PyKEEN representation that returns embeddings."""

    def __init__(self, num_embeddings, dim):
        self._data = np.random.randn(num_embeddings, dim).astype(np.float32)

    def __call__(self, indices=None):
        return FakeTensor(self._data)


def make_fake_model(num_entities=3, num_relations=2, dim=8):
    """Create a fake PyKEEN model with entity and relation representations."""
    model = MagicMock()
    model.entity_representations = [FakeRepresentation(num_entities, dim)]
    model.relation_representations = [FakeRepresentation(num_relations, dim)]
    model.state_dict.return_value = {"fake": "state"}
    return model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def storer_config(tmp_path):
    return {
        "model_path": str(tmp_path / "kg_model"),
        "entity_index_name": "test_entity_emb",
        "relation_index_name": "test_relation_emb",
        "vector_store_type": "in_memory",
        "store_to_graph": False,
    }


@pytest.fixture
def entity_to_id():
    return {"Einstein": 0, "Ulm": 1, "Physics": 2}


@pytest.fixture
def relation_to_id():
    return {"BORN_IN": 0, "FIELD": 1}


@pytest.fixture
def fake_model():
    return make_fake_model()


@pytest.fixture
def mock_vector_store():
    store = MagicMock()
    return store


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestKGEmbeddingStorer:
    def test_basic_storage(self, storer_config, fake_model, entity_to_id, relation_to_id, mock_vector_store):
        from retrico.modeling.kg_embedding_storer import KGEmbeddingStorerProcessor

        proc = KGEmbeddingStorerProcessor(storer_config)
        proc._vector_store = mock_vector_store

        result = proc(
            model=fake_model,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
        )

        assert result["entity_embeddings_shape"] == [3, 8]
        assert result["relation_embeddings_shape"] == [2, 8]
        assert result["model_path"] == storer_config["model_path"]

        # Check vector store was called
        assert mock_vector_store.create_index.call_count == 2
        assert mock_vector_store.store_embeddings.call_count == 2

    def test_disk_persistence(self, storer_config, fake_model, entity_to_id, relation_to_id, mock_vector_store):
        from retrico.modeling.kg_embedding_storer import KGEmbeddingStorerProcessor

        proc = KGEmbeddingStorerProcessor(storer_config)
        proc._vector_store = mock_vector_store

        proc(
            model=fake_model,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
        )

        model_dir = storer_config["model_path"]
        assert os.path.isdir(model_dir)

        # Check JSON files were written
        with open(os.path.join(model_dir, "entity_to_id.json")) as f:
            saved_e2id = json.load(f)
        assert saved_e2id == entity_to_id

        with open(os.path.join(model_dir, "relation_to_id.json")) as f:
            saved_r2id = json.load(f)
        assert saved_r2id == relation_to_id

        with open(os.path.join(model_dir, "metadata.json")) as f:
            metadata = json.load(f)
        assert metadata["entity_dim"] == 8
        assert metadata["num_entities"] == 3

    def test_store_to_graph(self, storer_config, fake_model, entity_to_id, relation_to_id, mock_vector_store):
        from retrico.modeling.kg_embedding_storer import KGEmbeddingStorerProcessor

        storer_config["store_to_graph"] = True
        proc = KGEmbeddingStorerProcessor(storer_config)
        proc._vector_store = mock_vector_store

        mock_store = MagicMock()
        mock_store.get_entity_by_label.return_value = {"id": "e1"}
        proc._store = mock_store

        proc(
            model=fake_model,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
        )

        assert mock_store.get_entity_by_label.call_count == 3
        assert mock_store.update_entity_embedding.call_count == 3

    def test_missing_model_raises(self, storer_config):
        from retrico.modeling.kg_embedding_storer import KGEmbeddingStorerProcessor

        proc = KGEmbeddingStorerProcessor(storer_config)
        with pytest.raises(ValueError, match="model"):
            proc()

    def test_default_config(self):
        from retrico.modeling.kg_embedding_storer import KGEmbeddingStorerProcessor

        proc = KGEmbeddingStorerProcessor({})
        assert proc.model_path == "kg_model"
        assert proc.entity_index_name == "kg_entity_embeddings"
        assert proc.relation_index_name == "kg_relation_embeddings"
        assert proc.store_to_graph is False

    def test_vector_index_names(self, storer_config, fake_model, entity_to_id, relation_to_id, mock_vector_store):
        from retrico.modeling.kg_embedding_storer import KGEmbeddingStorerProcessor

        proc = KGEmbeddingStorerProcessor(storer_config)
        proc._vector_store = mock_vector_store

        proc(
            model=fake_model,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
        )

        # Verify index names used
        create_calls = mock_vector_store.create_index.call_args_list
        assert create_calls[0][0][0] == "test_entity_emb"
        assert create_calls[1][0][0] == "test_relation_emb"

    def test_processor_registration(self):
        from retrico.core.registry import processor_registry
        assert "kg_embedding_storer" in processor_registry._factories
