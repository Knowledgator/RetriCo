"""Tests for KG triple reader processor."""

import sys
import pytest
import numpy as np
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Mock pykeen before importing the processor
# ---------------------------------------------------------------------------

mock_pykeen = MagicMock()
mock_triples_module = MagicMock()


class FakeTriplesFactory:
    """Minimal fake TriplesFactory for testing."""

    def __init__(self, triples=None, entity_to_id=None, relation_to_id=None, num_triples=10):
        self.entity_to_id = entity_to_id or {"Einstein": 0, "Ulm": 1, "Physics": 2}
        self.relation_to_id = relation_to_id or {"BORN_IN": 0, "FIELD": 1}
        self.num_triples = num_triples

    @classmethod
    def from_labeled_triples(cls, triples, **kwargs):
        return cls(triples=triples, num_triples=len(triples))

    @classmethod
    def from_path(cls, path, **kwargs):
        return cls(num_triples=5)

    def split(self, ratios, random_state=None):
        train = FakeTriplesFactory(num_triples=int(self.num_triples * ratios[0]))
        train.entity_to_id = self.entity_to_id
        train.relation_to_id = self.relation_to_id
        val = FakeTriplesFactory(num_triples=int(self.num_triples * ratios[1]))
        test = FakeTriplesFactory(num_triples=int(self.num_triples * ratios[2]))
        return train, val, test


mock_triples_module.TriplesFactory = FakeTriplesFactory
mock_pykeen.triples = mock_triples_module
sys.modules.setdefault("pykeen", mock_pykeen)
sys.modules.setdefault("pykeen.triples", mock_triples_module)

from grapsit.modeling.kg_triple_reader import KGTripleReaderProcessor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_store():
    store = MagicMock()
    store.get_all_triples.return_value = [
        ("Einstein", "BORN_IN", "Ulm"),
        ("Einstein", "FIELD", "Physics"),
        ("Bohr", "BORN_IN", "Copenhagen"),
    ]
    return store


@pytest.fixture
def reader_config():
    return {
        "source": "graph_store",
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "random_seed": 42,
        "store_type": "neo4j",
        "neo4j_uri": "bolt://localhost:7687",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestKGTripleReader:
    def test_read_from_graph_store(self, reader_config, mock_store):
        proc = KGTripleReaderProcessor(reader_config)
        proc._store = mock_store

        result = proc()

        assert "triples_factory" in result
        assert "training" in result
        assert "validation" in result
        assert "testing" in result
        assert "entity_to_id" in result
        assert "relation_to_id" in result
        assert result["triple_count"] == 3
        mock_store.get_all_triples.assert_called_once()

    def test_read_from_tsv(self):
        config = {"source": "tsv", "tsv_path": "/tmp/triples.tsv"}
        proc = KGTripleReaderProcessor(config)

        result = proc()

        assert result["triple_count"] == 5
        assert "training" in result

    def test_tsv_missing_path_raises(self):
        config = {"source": "tsv"}
        proc = KGTripleReaderProcessor(config)

        with pytest.raises(ValueError, match="tsv_path required"):
            proc()

    def test_empty_graph_raises(self, reader_config):
        store = MagicMock()
        store.get_all_triples.return_value = []
        proc = KGTripleReaderProcessor(reader_config)
        proc._store = store

        with pytest.raises(ValueError, match="No triples found"):
            proc()

    def test_split_ratios(self, reader_config, mock_store):
        reader_config["train_ratio"] = 0.6
        reader_config["val_ratio"] = 0.2
        reader_config["test_ratio"] = 0.2
        proc = KGTripleReaderProcessor(reader_config)
        proc._store = mock_store

        result = proc()

        assert result["training"].num_triples == 1  # 3 * 0.6 = 1.8 -> int = 1
        assert result["triple_count"] == 3

    def test_entity_to_id_mapping(self, reader_config, mock_store):
        proc = KGTripleReaderProcessor(reader_config)
        proc._store = mock_store

        result = proc()

        assert isinstance(result["entity_to_id"], dict)
        assert isinstance(result["relation_to_id"], dict)

    def test_default_config_values(self):
        proc = KGTripleReaderProcessor({})
        assert proc.source == "graph_store"
        assert proc.train_ratio == 0.8
        assert proc.val_ratio == 0.1
        assert proc.test_ratio == 0.1
        assert proc.random_seed == 42

    def test_processor_registration(self):
        from grapsit.core.registry import processor_registry
        assert "kg_triple_reader" in processor_registry._factories
