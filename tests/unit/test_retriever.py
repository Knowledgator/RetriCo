"""Tests for retriever processor."""

import pytest
from unittest.mock import MagicMock, patch

from grapsit.query.retriever import RetrieverProcessor
from grapsit.models.entity import EntityMention


class TestRetrieverProcessor:
    def _make_proc(self, **config_overrides):
        config = {
            "neo4j_uri": "bolt://localhost:7687",
            "neo4j_user": "neo4j",
            "neo4j_password": "password",
            "max_hops": 2,
        }
        config.update(config_overrides)
        proc = RetrieverProcessor(config)
        proc._store = MagicMock()
        return proc

    def test_basic_retrieval(self):
        proc = self._make_proc()
        proc._store.get_entity_by_label.side_effect = [
            {"id": "e1", "label": "Einstein", "entity_type": "person"},
        ]
        proc._store.get_subgraph.return_value = {
            "entities": [
                {"id": "e1", "label": "Einstein", "entity_type": "person"},
                {"id": "e2", "label": "Ulm", "entity_type": "location"},
            ],
            "relations": [
                {"head": "e1", "tail": "e2", "type": "BORN_IN", "score": 0.8},
            ],
        }

        entities = [EntityMention(text="Einstein", label="person")]
        result = proc(entities=entities)

        assert "subgraph" in result
        sg = result["subgraph"]
        assert len(sg.entities) == 2
        assert len(sg.relations) == 1
        assert sg.relations[0].relation_type == "BORN_IN"

    def test_no_entities_found_in_graph(self):
        proc = self._make_proc()
        proc._store.get_entity_by_label.return_value = None

        entities = [EntityMention(text="Unknown", label="person")]
        result = proc(entities=entities)

        sg = result["subgraph"]
        assert len(sg.entities) == 0
        assert len(sg.relations) == 0

    def test_multiple_entities_with_overlap(self):
        proc = self._make_proc()
        proc._store.get_entity_by_label.side_effect = [
            {"id": "e1", "label": "Einstein", "entity_type": "person"},
            {"id": "e2", "label": "Ulm", "entity_type": "location"},
        ]
        proc._store.get_subgraph.return_value = {
            "entities": [
                {"id": "e1", "label": "Einstein", "entity_type": "person"},
                {"id": "e2", "label": "Ulm", "entity_type": "location"},
                {"id": "e3", "label": "Germany", "entity_type": "location"},
            ],
            "relations": [
                {"head": "e1", "tail": "e2", "type": "BORN_IN", "score": 0.8},
                {"head": "e2", "tail": "e3", "type": "LOCATED_IN", "score": 0.9},
            ],
        }

        entities = [
            EntityMention(text="Einstein", label="person"),
            EntityMention(text="Ulm", label="location"),
        ]
        result = proc(entities=entities)

        sg = result["subgraph"]
        assert len(sg.entities) == 3
        assert len(sg.relations) == 2

    def test_skips_unmatched_entities(self):
        proc = self._make_proc()
        proc._store.get_entity_by_label.side_effect = [
            {"id": "e1", "label": "Einstein", "entity_type": "person"},
            None,  # Unknown not found
        ]
        proc._store.get_subgraph.return_value = {
            "entities": [
                {"id": "e1", "label": "Einstein", "entity_type": "person"},
            ],
            "relations": [],
        }

        entities = [
            EntityMention(text="Einstein", label="person"),
            EntityMention(text="Unknown", label="person"),
        ]
        result = proc(entities=entities)

        sg = result["subgraph"]
        assert len(sg.entities) == 1
        proc._store.get_subgraph.assert_called_once_with(["e1"], max_hops=2)

    def test_null_relation_type_filtered(self):
        proc = self._make_proc()
        proc._store.get_entity_by_label.return_value = {"id": "e1", "label": "X", "entity_type": ""}
        proc._store.get_subgraph.return_value = {
            "entities": [{"id": "e1", "label": "X", "entity_type": ""}],
            "relations": [
                {"head": "e1", "tail": "e2", "type": None, "score": None},
            ],
        }

        result = proc(entities=[EntityMention(text="X", label="")])
        assert len(result["subgraph"].relations) == 0
