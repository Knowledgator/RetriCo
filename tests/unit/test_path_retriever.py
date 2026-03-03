"""Tests for path retriever processor."""

import pytest
from unittest.mock import MagicMock

from grapsit.query.path_retriever import PathRetrieverProcessor
from grapsit.models.entity import EntityMention


class TestPathRetrieverProcessor:
    def _make_proc(self, **config_overrides):
        config = {
            "neo4j_uri": "bolt://localhost:7687",
            "max_path_length": 5,
            "max_pairs": 10,
        }
        config.update(config_overrides)
        proc = PathRetrieverProcessor(config)
        proc._store = MagicMock()
        return proc

    def test_basic_path_retrieval(self):
        proc = self._make_proc()
        proc._store.get_entity_by_label.side_effect = [
            {"id": "e1", "label": "Einstein", "entity_type": "person"},
            {"id": "e2", "label": "Ulm", "entity_type": "location"},
        ]
        proc._store.get_shortest_paths.return_value = [
            {
                "nodes": [
                    {"id": "e1", "label": "Einstein", "entity_type": "person"},
                    {"id": "e2", "label": "Ulm", "entity_type": "location"},
                ],
                "rels": [
                    {"type": "BORN_IN", "score": 0.8},
                ],
            }
        ]

        entities = [
            EntityMention(text="Einstein", label="person"),
            EntityMention(text="Ulm", label="location"),
        ]
        result = proc(entities=entities)

        sg = result["subgraph"]
        assert len(sg.entities) == 2
        assert len(sg.relations) == 1
        assert sg.relations[0].relation_type == "BORN_IN"
        assert sg.relations[0].head_text == "e1"
        assert sg.relations[0].tail_text == "e2"

    def test_empty_entities(self):
        proc = self._make_proc()
        result = proc(entities=[])
        assert len(result["subgraph"].entities) == 0

    def test_single_entity_fallback(self):
        """With only one matched entity, falls back to subgraph expansion."""
        proc = self._make_proc()
        proc._store.get_entity_by_label.side_effect = [
            {"id": "e1", "label": "Einstein", "entity_type": "person"},
            None,  # second entity not found
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
        proc._store.get_subgraph.assert_called_once_with(["e1"], max_hops=1)

    def test_no_entities_found(self):
        proc = self._make_proc()
        proc._store.get_entity_by_label.return_value = None

        entities = [
            EntityMention(text="Unknown1", label=""),
            EntityMention(text="Unknown2", label=""),
        ]
        result = proc(entities=entities)
        assert len(result["subgraph"].entities) == 0

    def test_multiple_pairs(self):
        proc = self._make_proc()
        proc._store.get_entity_by_label.side_effect = [
            {"id": "e1", "label": "A", "entity_type": ""},
            {"id": "e2", "label": "B", "entity_type": ""},
            {"id": "e3", "label": "C", "entity_type": ""},
        ]
        proc._store.get_shortest_paths.side_effect = [
            [{  # A → B
                "nodes": [
                    {"id": "e1", "label": "A", "entity_type": ""},
                    {"id": "e2", "label": "B", "entity_type": ""},
                ],
                "rels": [{"type": "REL_1", "score": 0.9}],
            }],
            [{  # A → C
                "nodes": [
                    {"id": "e1", "label": "A", "entity_type": ""},
                    {"id": "e3", "label": "C", "entity_type": ""},
                ],
                "rels": [{"type": "REL_2", "score": 0.8}],
            }],
            [{  # B → C
                "nodes": [
                    {"id": "e2", "label": "B", "entity_type": ""},
                    {"id": "e3", "label": "C", "entity_type": ""},
                ],
                "rels": [{"type": "REL_3", "score": 0.7}],
            }],
        ]

        entities = [
            EntityMention(text="A", label=""),
            EntityMention(text="B", label=""),
            EntityMention(text="C", label=""),
        ]
        result = proc(entities=entities)

        sg = result["subgraph"]
        # 3 unique entities, 3 relations from 3 paths
        assert len(sg.entities) == 3
        assert len(sg.relations) == 3

    def test_max_pairs_limit(self):
        proc = self._make_proc(max_pairs=1)
        proc._store.get_entity_by_label.side_effect = [
            {"id": "e1", "label": "A", "entity_type": ""},
            {"id": "e2", "label": "B", "entity_type": ""},
            {"id": "e3", "label": "C", "entity_type": ""},
        ]
        proc._store.get_shortest_paths.return_value = [{
            "nodes": [
                {"id": "e1", "label": "A", "entity_type": ""},
                {"id": "e2", "label": "B", "entity_type": ""},
            ],
            "rels": [{"type": "REL", "score": 0.5}],
        }]

        entities = [
            EntityMention(text="A", label=""),
            EntityMention(text="B", label=""),
            EntityMention(text="C", label=""),
        ]
        result = proc(entities=entities)

        # Only 1 pair processed due to max_pairs=1
        assert proc._store.get_shortest_paths.call_count == 1

    def test_no_paths_found(self):
        proc = self._make_proc()
        proc._store.get_entity_by_label.side_effect = [
            {"id": "e1", "label": "A", "entity_type": ""},
            {"id": "e2", "label": "B", "entity_type": ""},
        ]
        proc._store.get_shortest_paths.return_value = []

        entities = [
            EntityMention(text="A", label=""),
            EntityMention(text="B", label=""),
        ]
        result = proc(entities=entities)

        sg = result["subgraph"]
        assert len(sg.entities) == 0
        assert len(sg.relations) == 0

    def test_store_not_implemented(self):
        proc = self._make_proc()
        proc._store.get_entity_by_label.side_effect = [
            {"id": "e1", "label": "A", "entity_type": ""},
            {"id": "e2", "label": "B", "entity_type": ""},
        ]
        proc._store.get_shortest_paths.side_effect = NotImplementedError

        entities = [
            EntityMention(text="A", label=""),
            EntityMention(text="B", label=""),
        ]
        result = proc(entities=entities)
        assert len(result["subgraph"].entities) == 0

    def test_config_defaults(self):
        proc = PathRetrieverProcessor({})
        assert proc.max_path_length == 5
        assert proc.max_pairs == 10
