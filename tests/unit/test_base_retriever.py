"""Tests for BaseRetriever shared helpers."""

import pytest
from unittest.mock import MagicMock

from retrico.query.base_retriever import BaseRetriever
from retrico.models.entity import EntityMention


class ConcreteRetriever(BaseRetriever):
    """Concrete subclass for testing abstract base."""

    def __call__(self, **kwargs):
        return {"subgraph": None}


class TestBaseRetriever:
    def _make_proc(self, **config_overrides):
        config = {
            "neo4j_uri": "bolt://localhost:7687",
            "neo4j_user": "neo4j",
            "neo4j_password": "password",
        }
        config.update(config_overrides)
        proc = ConcreteRetriever(config)
        proc._store = MagicMock()
        return proc

    def test_lookup_entity_by_linked_id(self):
        proc = self._make_proc()
        proc._store.get_entity_by_id.return_value = {"id": "e1", "label": "Einstein"}

        mention = EntityMention(text="Einstein", label="person", linked_entity_id="e1")
        result = proc._lookup_entity(mention)

        assert result["id"] == "e1"
        proc._store.get_entity_by_id.assert_called_once_with("e1")
        proc._store.get_entity_by_label.assert_not_called()

    def test_lookup_entity_by_label(self):
        proc = self._make_proc()
        proc._store.get_entity_by_label.return_value = {"id": "e1", "label": "Einstein"}

        mention = EntityMention(text="Einstein", label="person")
        result = proc._lookup_entity(mention)

        assert result["id"] == "e1"
        proc._store.get_entity_by_label.assert_called_once_with("Einstein")

    def test_lookup_entity_not_found(self):
        proc = self._make_proc()
        proc._store.get_entity_by_label.return_value = None

        mention = EntityMention(text="Unknown", label="person")
        result = proc._lookup_entity(mention)

        assert result is None

    def test_raw_to_subgraph(self):
        proc = self._make_proc()
        raw = {
            "entities": [
                {"id": "e1", "label": "Einstein", "entity_type": "person"},
                {"id": "e2", "label": "Ulm", "entity_type": "location"},
            ],
            "relations": [
                {"head": "e1", "tail": "e2", "type": "BORN_IN", "score": 0.8},
            ],
        }

        sg = proc._raw_to_subgraph(raw)

        assert len(sg.entities) == 2
        assert len(sg.relations) == 1
        assert sg.entities[0].label == "Einstein"
        assert sg.relations[0].relation_type == "BORN_IN"

    def test_raw_to_subgraph_skips_none(self):
        proc = self._make_proc()
        raw = {
            "entities": [None, {"id": "e1", "label": "X", "entity_type": ""}],
            "relations": [None, {"head": "e1", "tail": "e2", "type": None}],
        }

        sg = proc._raw_to_subgraph(raw)
        assert len(sg.entities) == 1
        assert len(sg.relations) == 0

    def test_raw_to_subgraph_empty(self):
        proc = self._make_proc()
        sg = proc._raw_to_subgraph({"entities": [], "relations": []})
        assert len(sg.entities) == 0
        assert len(sg.relations) == 0

    def test_ensure_store_lazy(self):
        proc = ConcreteRetriever({"neo4j_uri": "bolt://localhost:7687"})
        assert proc._store is None
        # Once set, it should not be re-created
        proc._store = MagicMock()
        original = proc._store
        proc._ensure_store()
        assert proc._store is original
