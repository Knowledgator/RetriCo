"""Tests for chunk_id List[str] migration across the pipeline."""

import json
import pytest
from unittest.mock import MagicMock, patch

from retrico.models.relation import Relation
from retrico.models.graph import KGTriple, Subgraph
from retrico.models.entity import EntityMention, Entity
from retrico.models.document import Chunk, Document


# ---------------------------------------------------------------------------
# 1. Model backward compatibility
# ---------------------------------------------------------------------------


class TestRelationChunkIdMigration:
    """Relation.chunk_id: str -> List[str] with backward compat."""

    def test_default_is_empty_list(self):
        r = Relation(head_text="A", tail_text="B", relation_type="REL")
        assert r.chunk_id == []

    def test_string_migrates_to_list(self):
        r = Relation(head_text="A", tail_text="B", relation_type="REL", chunk_id="c1")
        assert r.chunk_id == ["c1"]

    def test_empty_string_migrates_to_empty_list(self):
        r = Relation(head_text="A", tail_text="B", relation_type="REL", chunk_id="")
        assert r.chunk_id == []

    def test_list_stays_as_list(self):
        r = Relation(head_text="A", tail_text="B", relation_type="REL", chunk_id=["c1", "c2"])
        assert r.chunk_id == ["c1", "c2"]

    def test_empty_list(self):
        r = Relation(head_text="A", tail_text="B", relation_type="REL", chunk_id=[])
        assert r.chunk_id == []

    def test_dict_construction(self):
        """Constructing from a dict (e.g. JSON deserialization) works."""
        data = {"head_text": "A", "tail_text": "B", "relation_type": "REL", "chunk_id": "c3"}
        r = Relation(**data)
        assert r.chunk_id == ["c3"]

    def test_dict_construction_with_list(self):
        data = {"head_text": "A", "tail_text": "B", "relation_type": "REL", "chunk_id": ["c3", "c4"]}
        r = Relation(**data)
        assert r.chunk_id == ["c3", "c4"]


class TestKGTripleChunkIdMigration:
    """KGTriple.chunk_id: str -> List[str] with backward compat."""

    def test_default_is_empty_list(self):
        t = KGTriple(head="A", relation="REL", tail="B")
        assert t.chunk_id == []

    def test_string_migrates_to_list(self):
        t = KGTriple(head="A", relation="REL", tail="B", chunk_id="c1")
        assert t.chunk_id == ["c1"]

    def test_empty_string_migrates_to_empty_list(self):
        t = KGTriple(head="A", relation="REL", tail="B", chunk_id="")
        assert t.chunk_id == []

    def test_list_stays_as_list(self):
        t = KGTriple(head="A", relation="REL", tail="B", chunk_id=["c1", "c2"])
        assert t.chunk_id == ["c1", "c2"]


# ---------------------------------------------------------------------------
# 2. Graph writer — relation dedup accumulates chunk_ids
# ---------------------------------------------------------------------------


class TestGraphWriterChunkIdAccumulation:
    """Graph writer deduplicates relations and accumulates chunk_ids."""

    @patch("retrico.construct.graph_writer.resolve_from_pool_or_create")
    def test_same_relation_in_two_chunks_accumulates(self, mock_resolve):
        mock_store = MagicMock()
        mock_store.setup_indexes = MagicMock()
        mock_resolve.return_value = mock_store

        from retrico.construct.graph_writer import GraphWriterProcessor

        writer = GraphWriterProcessor({"store_type": "neo4j", "setup_indexes": False})

        chunks = [
            Chunk(id="c1", text="Einstein was born in Ulm.", index=0),
            Chunk(id="c2", text="Ulm is where Einstein was born.", index=1),
        ]
        entities = [
            [
                EntityMention(text="Einstein", label="person", chunk_id="c1"),
                EntityMention(text="Ulm", label="location", chunk_id="c1"),
            ],
            [
                EntityMention(text="Einstein", label="person", chunk_id="c2"),
                EntityMention(text="Ulm", label="location", chunk_id="c2"),
            ],
        ]
        relations = [
            [Relation(head_text="Einstein", tail_text="Ulm", relation_type="born in", chunk_id=["c1"])],
            [Relation(head_text="Einstein", tail_text="Ulm", relation_type="born in", chunk_id=["c2"])],
        ]

        result = writer(chunks=chunks, entities=entities, relations=relations)

        # Should write only 1 deduplicated relation
        assert result["relation_count"] == 1
        assert mock_store.write_relation.call_count == 1

        # The written relation should have both chunk_ids
        written_rel = mock_store.write_relation.call_args[0][0]
        assert sorted(written_rel.chunk_id) == ["c1", "c2"]

    @patch("retrico.construct.graph_writer.resolve_from_pool_or_create")
    def test_different_relations_not_merged(self, mock_resolve):
        mock_store = MagicMock()
        mock_store.setup_indexes = MagicMock()
        mock_resolve.return_value = mock_store

        from retrico.construct.graph_writer import GraphWriterProcessor

        writer = GraphWriterProcessor({"store_type": "neo4j", "setup_indexes": False})

        entities = [
            [
                EntityMention(text="Einstein", label="person", chunk_id="c1"),
                EntityMention(text="Ulm", label="location", chunk_id="c1"),
            ],
        ]
        relations = [
            [
                Relation(head_text="Einstein", tail_text="Ulm", relation_type="born in", chunk_id=["c1"]),
                Relation(head_text="Einstein", tail_text="Ulm", relation_type="visited", chunk_id=["c1"]),
            ],
        ]

        result = writer(entities=entities, relations=relations)
        assert result["relation_count"] == 2
        assert mock_store.write_relation.call_count == 2

    @patch("retrico.construct.graph_writer.resolve_from_pool_or_create")
    def test_reversed_relations_use_list_chunk_id(self, mock_resolve):
        mock_store = MagicMock()
        mock_store.setup_indexes = MagicMock()
        mock_resolve.return_value = mock_store

        from retrico.construct.graph_writer import GraphWriterProcessor

        writer = GraphWriterProcessor({
            "store_type": "neo4j",
            "setup_indexes": False,
            "write_reversed_relations": True,
        })

        entities = [
            [
                EntityMention(text="Einstein", label="person", chunk_id="c1"),
                EntityMention(text="Ulm", label="location", chunk_id="c1"),
            ],
        ]
        relations = [
            [Relation(head_text="Einstein", tail_text="Ulm", relation_type="born in", chunk_id=["c1", "c2"])],
        ]

        writer(entities=entities, relations=relations)

        # Should write forward + reverse relation
        assert mock_store.write_relation.call_count == 2
        rev_rel = mock_store.write_relation.call_args_list[1][0][0]
        assert rev_rel.chunk_id == ["c1", "c2"]
        assert rev_rel.relation_type.startswith("REV_")


# ---------------------------------------------------------------------------
# 3. Data ingest — all chunk IDs set on relations
# ---------------------------------------------------------------------------


class TestIngestChunkIdList:
    """Data ingest sets all chunk_ids on relations."""

    def test_relation_gets_all_chunk_ids(self):
        from retrico.construct.ingest import DataIngestProcessor

        proc = DataIngestProcessor({})

        data = [
            {
                "entities": [
                    {"text": "Einstein", "label": "person"},
                    {"text": "Ulm", "label": "location"},
                ],
                "relations": [
                    {"head": "Einstein", "tail": "Ulm", "type": "born_in"},
                ],
                "text": "Einstein was born in Ulm. He lived in Ulm for years.",
            }
        ]
        result = proc(data=data)

        rels = result["relations"][0]
        assert len(rels) == 1
        # Should have chunk_ids for all chunks from the text
        assert len(rels[0].chunk_id) >= 1
        assert isinstance(rels[0].chunk_id, list)

    def test_relation_without_text_gets_empty_chunk_ids(self):
        from retrico.construct.ingest import DataIngestProcessor

        proc = DataIngestProcessor({})

        data = [
            {
                "entities": [
                    {"text": "Einstein", "label": "person"},
                    {"text": "Ulm", "label": "location"},
                ],
                "relations": [
                    {"head": "Einstein", "tail": "Ulm", "type": "born_in"},
                ],
            }
        ]
        result = proc(data=data)

        rels = result["relations"][0]
        assert rels[0].chunk_id == []


# ---------------------------------------------------------------------------
# 4. BaseRetriever — _raw_to_subgraph passes chunk_id through
# ---------------------------------------------------------------------------


def _make_retriever(config=None):
    """Create a concrete BaseRetriever subclass for testing."""
    from retrico.query.base_retriever import BaseRetriever

    class _TestRetriever(BaseRetriever):
        def __call__(self, **kwargs):
            return {}

    return _TestRetriever(config or {})


class TestBaseRetrieverChunkId:
    """BaseRetriever._raw_to_subgraph passes chunk_id list through."""

    def test_raw_to_subgraph_with_chunk_id_list(self):
        proc = _make_retriever()
        raw = {
            "entities": [
                {"id": "e1", "label": "Einstein", "entity_type": "person"},
                {"id": "e2", "label": "Ulm", "entity_type": "location"},
            ],
            "relations": [
                {"head": "e1", "tail": "e2", "type": "BORN_IN", "score": 0.8,
                 "chunk_id": ["c1", "c2"]},
            ],
        }
        sg = proc._raw_to_subgraph(raw)
        assert sg.relations[0].chunk_id == ["c1", "c2"]

    def test_raw_to_subgraph_with_string_chunk_id(self):
        """Backward compat: string chunk_id from old DB data."""
        proc = _make_retriever()
        raw = {
            "entities": [
                {"id": "e1", "label": "A", "entity_type": ""},
                {"id": "e2", "label": "B", "entity_type": ""},
            ],
            "relations": [
                {"head": "e1", "tail": "e2", "type": "REL", "score": 0.5,
                 "chunk_id": "c1"},
            ],
        }
        sg = proc._raw_to_subgraph(raw)
        assert sg.relations[0].chunk_id == ["c1"]

    def test_raw_to_subgraph_without_chunk_id(self):
        proc = _make_retriever()
        raw = {
            "entities": [
                {"id": "e1", "label": "A", "entity_type": ""},
                {"id": "e2", "label": "B", "entity_type": ""},
            ],
            "relations": [
                {"head": "e1", "tail": "e2", "type": "REL", "score": 0.5},
            ],
        }
        sg = proc._raw_to_subgraph(raw)
        assert sg.relations[0].chunk_id == []

    def test_get_chunks_from_relations(self):
        proc = _make_retriever()
        proc._store = MagicMock()
        proc._store.get_chunks_by_ids.return_value = [
            {"id": "c1", "text": "Chunk 1", "document_id": "d1", "index": 0, "start_char": 0, "end_char": 10},
            {"id": "c2", "text": "Chunk 2", "document_id": "d1", "index": 1, "start_char": 10, "end_char": 20},
        ]

        rels = [
            Relation(head_text="A", tail_text="B", relation_type="REL", chunk_id=["c1", "c2"]),
            Relation(head_text="A", tail_text="C", relation_type="REL2", chunk_id=["c2", "c3"]),
        ]

        chunks = proc._get_chunks_from_relations(rels)
        # Should batch-fetch unique chunk_ids
        proc._store.get_chunks_by_ids.assert_called_once_with(["c1", "c2", "c3"])
        assert len(chunks) == 2  # only 2 returned from store

    def test_get_chunks_from_relations_empty(self):
        proc = _make_retriever()
        proc._store = MagicMock()

        rels = [
            Relation(head_text="A", tail_text="B", relation_type="REL", chunk_id=[]),
        ]
        chunks = proc._get_chunks_from_relations(rels)
        assert chunks == []
        proc._store.get_chunks_by_ids.assert_not_called()


# ---------------------------------------------------------------------------
# 5. Path retriever — chunk_source config
# ---------------------------------------------------------------------------


class TestPathRetrieverChunkSource:
    """Path retriever fetches relation-derived chunks when chunk_source is set."""

    def _make_proc(self, **overrides):
        from retrico.query.path_retriever import PathRetrieverProcessor

        config = {"neo4j_uri": "bolt://localhost:7687", "max_path_length": 5, "top_k": 3}
        config.update(overrides)
        proc = PathRetrieverProcessor(config)
        proc._store = MagicMock()
        return proc

    def test_default_chunk_source_is_entity(self):
        proc = self._make_proc()
        assert proc.chunk_source == "entity"

    def test_chunk_source_relation_fetches_chunks(self):
        proc = self._make_proc(chunk_source="relation")
        proc._store.get_entity_by_label.side_effect = [
            {"id": "e1", "label": "A", "entity_type": ""},
            {"id": "e2", "label": "B", "entity_type": ""},
        ]
        proc._store.get_top_shortest_paths.return_value = [
            {
                "nodes": [
                    {"id": "e1", "label": "A", "entity_type": ""},
                    {"id": "e2", "label": "B", "entity_type": ""},
                ],
                "rels": [{"type": "REL", "score": 0.9, "chunk_id": ["c1", "c2"]}],
            }
        ]
        proc._store.get_chunks_by_ids.return_value = [
            {"id": "c1", "text": "Chunk 1", "document_id": "d1", "index": 0, "start_char": 0, "end_char": 10},
        ]

        entities = [
            EntityMention(text="A", label=""),
            EntityMention(text="B", label=""),
        ]
        result = proc(entities=entities)

        sg = result["subgraph"]
        assert len(sg.chunks) == 1
        assert sg.chunks[0].id == "c1"
        proc._store.get_chunks_by_ids.assert_called_once()

    def test_chunk_source_entity_no_chunk_fetch(self):
        proc = self._make_proc(chunk_source="entity")
        proc._store.get_entity_by_label.side_effect = [
            {"id": "e1", "label": "A", "entity_type": ""},
            {"id": "e2", "label": "B", "entity_type": ""},
        ]
        proc._store.get_top_shortest_paths.return_value = [
            {
                "nodes": [
                    {"id": "e1", "label": "A", "entity_type": ""},
                    {"id": "e2", "label": "B", "entity_type": ""},
                ],
                "rels": [{"type": "REL", "score": 0.9, "chunk_id": ["c1"]}],
            }
        ]

        entities = [
            EntityMention(text="A", label=""),
            EntityMention(text="B", label=""),
        ]
        result = proc(entities=entities)

        sg = result["subgraph"]
        assert len(sg.chunks) == 0
        proc._store.get_chunks_by_ids.assert_not_called()

    def test_chunk_source_both(self):
        proc = self._make_proc(chunk_source="both")
        proc._store.get_entity_by_label.side_effect = [
            {"id": "e1", "label": "A", "entity_type": ""},
            {"id": "e2", "label": "B", "entity_type": ""},
        ]
        proc._store.get_top_shortest_paths.return_value = [
            {
                "nodes": [
                    {"id": "e1", "label": "A", "entity_type": ""},
                    {"id": "e2", "label": "B", "entity_type": ""},
                ],
                "rels": [{"type": "REL", "score": 0.9, "chunk_id": ["c1"]}],
            }
        ]
        proc._store.get_chunks_by_ids.return_value = [
            {"id": "c1", "text": "Chunk 1", "document_id": "d1", "index": 0, "start_char": 0, "end_char": 10},
        ]

        entities = [
            EntityMention(text="A", label=""),
            EntityMention(text="B", label=""),
        ]
        result = proc(entities=entities)

        sg = result["subgraph"]
        assert len(sg.chunks) == 1


# ---------------------------------------------------------------------------
# 6. Tool retriever — chunk_source config
# ---------------------------------------------------------------------------


class TestToolRetrieverChunkSource:
    """Tool retriever merges relation-derived chunks when chunk_source is set."""

    def _make_proc(self, **overrides):
        from retrico.query.tool_retriever import ToolRetrieverProcessor

        config = {
            "neo4j_uri": "bolt://localhost:7687",
            "api_key": "test",
            "model": "test-model",
            "entity_types": [],
            "relation_types": [],
        }
        config.update(overrides)
        proc = ToolRetrieverProcessor(config)
        proc._store = MagicMock()
        proc._llm = MagicMock()
        return proc

    def test_default_chunk_source_is_entity(self):
        proc = self._make_proc()
        assert proc.chunk_source == "entity"

    def test_chunk_source_relation(self):
        proc = self._make_proc(chunk_source="relation")
        proc._llm.complete_with_tools.side_effect = [
            {
                "content": None,
                "tool_calls": [{
                    "id": "tc1",
                    "name": "get_entity_relations",
                    "arguments": {"entity_id": "e1"},
                }],
            },
            {"content": "Done", "tool_calls": []},
        ]
        proc._store.run_cypher.return_value = [
            {
                "entity_id": "e1",
                "relation_type": "BORN_IN",
                "target_id": "e2",
                "score": 0.8,
                "chunk_id": ["c1", "c2"],
            },
        ]
        proc._store.get_chunks_by_ids.return_value = [
            {"id": "c1", "text": "Chunk 1", "document_id": "d1", "index": 0, "start_char": 0, "end_char": 10},
        ]

        result = proc(query="Where was Einstein born?")
        sg = result["subgraph"]
        assert len(sg.relations) == 1
        assert sg.relations[0].chunk_id == ["c1", "c2"]
        assert len(sg.chunks) == 1
        proc._store.get_chunks_by_ids.assert_called_once()


# ---------------------------------------------------------------------------
# 7. LLM tools — get_chunks_for_relation
# ---------------------------------------------------------------------------


class TestGetChunksForRelationTool:
    """New LLM tool: get_chunks_for_relation."""

    def test_tool_definition_exists(self):
        from retrico.llm.tools import GRAPH_TOOLS

        names = [t["function"]["name"] for t in GRAPH_TOOLS]
        assert "get_chunks_for_relation" in names

    def test_translator_registered(self):
        from retrico.llm.tools import tool_call_to_cypher

        cypher, params = tool_call_to_cypher("get_chunks_for_relation", {
            "head_entity_id": "e1",
            "tail_entity_id": "e2",
        })
        assert "r.chunk_id" in cypher
        assert params["head_id"] == "e1"
        assert params["tail_id"] == "e2"

    def test_translator_with_relation_type(self):
        from retrico.llm.tools import tool_call_to_cypher

        cypher, params = tool_call_to_cypher("get_chunks_for_relation", {
            "head_entity_id": "e1",
            "tail_entity_id": "e2",
            "relation_type": "BORN_IN",
        })
        assert "BORN_IN" in cypher


# ---------------------------------------------------------------------------
# 8. Builder — chunk_source parameter
# ---------------------------------------------------------------------------


class TestBuilderChunkSource:
    """Builders pass chunk_source to retriever configs."""

    def test_path_retriever_chunk_source(self):
        from retrico.core.builders import RetriCoSearch

        builder = RetriCoSearch(name="test")
        builder.query_parser(method="gliner", labels=["person"])
        builder.path_retriever(chunk_source="relation")

        config = builder.get_config()
        retriever_node = next(
            n for n in config["nodes"] if n["processor"] == "path_retriever"
        )
        assert retriever_node["config"]["chunk_source"] == "relation"

    def test_tool_retriever_chunk_source(self):
        from retrico.core.builders import RetriCoSearch

        builder = RetriCoSearch(name="test")
        builder.query_parser(method="gliner", labels=["person"])
        builder.tool_retriever(api_key="test", chunk_source="both")

        config = builder.get_config()
        retriever_node = next(
            n for n in config["nodes"] if n["processor"] == "tool_retriever"
        )
        assert retriever_node["config"]["chunk_source"] == "both"

    def test_path_retriever_default_chunk_source(self):
        from retrico.core.builders import RetriCoSearch

        builder = RetriCoSearch(name="test")
        builder.query_parser(method="gliner", labels=["person"])
        builder.path_retriever()

        config = builder.get_config()
        retriever_node = next(
            n for n in config["nodes"] if n["processor"] == "path_retriever"
        )
        assert retriever_node["config"]["chunk_source"] == "entity"


# ---------------------------------------------------------------------------
# 9. Store — get_chunks_by_ids
# ---------------------------------------------------------------------------


class TestStoreGetChunksByIds:
    """BaseGraphStore.get_chunks_by_ids default implementation."""

    def test_default_implementation(self):
        from retrico.store.graph.base import BaseGraphStore

        # Create a concrete subclass with get_chunk_by_id implemented
        class TestStore(BaseGraphStore):
            def setup_indexes(self): pass
            def close(self): pass
            def write_document(self, doc): pass
            def write_chunk(self, chunk): pass
            def write_chunk_document_link(self, chunk_id, document_id): pass
            def write_entity(self, entity): pass
            def write_mention_link(self, entity_id, chunk_id, mention): pass
            def write_relation(self, relation, head_entity_id, tail_entity_id): pass
            def get_entity_by_id(self, entity_id): pass
            def get_all_entities(self): pass
            def get_entity_by_label(self, label): pass
            def get_entity_neighbors(self, entity_id, max_hops=1): pass
            def get_entity_relations(self, entity_id, *, active_after=None, active_before=None): pass
            def get_chunks_for_entity(self, entity_id): pass
            def get_subgraph(self, entity_ids, max_hops=1, *, active_after=None, active_before=None): pass
            def clear_all(self): pass

            def get_chunk_by_id(self, chunk_id):
                if chunk_id == "c1":
                    return {"id": "c1", "text": "Chunk 1"}
                if chunk_id == "c2":
                    return {"id": "c2", "text": "Chunk 2"}
                return None

        store = TestStore()
        results = store.get_chunks_by_ids(["c1", "c2", "c3", "c1"])  # c1 deduped, c3 missing
        assert len(results) == 2
        assert {r["id"] for r in results} == {"c1", "c2"}

    def test_empty_ids(self):
        from retrico.store.graph.base import BaseGraphStore

        class TestStore(BaseGraphStore):
            def setup_indexes(self): pass
            def close(self): pass
            def write_document(self, doc): pass
            def write_chunk(self, chunk): pass
            def write_chunk_document_link(self, chunk_id, document_id): pass
            def write_entity(self, entity): pass
            def write_mention_link(self, entity_id, chunk_id, mention): pass
            def write_relation(self, relation, head_entity_id, tail_entity_id): pass
            def get_entity_by_id(self, entity_id): pass
            def get_all_entities(self): pass
            def get_entity_by_label(self, label): pass
            def get_entity_neighbors(self, entity_id, max_hops=1): pass
            def get_entity_relations(self, entity_id, *, active_after=None, active_before=None): pass
            def get_chunks_for_entity(self, entity_id): pass
            def get_subgraph(self, entity_ids, max_hops=1, *, active_after=None, active_before=None): pass
            def clear_all(self): pass
            def get_chunk_by_id(self, chunk_id): pass

        store = TestStore()
        assert store.get_chunks_by_ids([]) == []
