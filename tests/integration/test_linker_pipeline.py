"""Integration tests — linker in full build and query pipelines."""

import json
import pytest
from unittest.mock import MagicMock, patch

from grapsit.core.builders import BuildConfigBuilder, QueryConfigBuilder
from grapsit.models.entity import EntityMention
from grapsit.models.graph import QueryResult


def _make_mock_glinker_executor(linked_entities=None):
    """Create a mock GLinker executor."""
    executor = MagicMock()
    l0_result = MagicMock()

    mock_entities = []
    for ent in (linked_entities or []):
        mock_ent = MagicMock()
        mock_ent.mention_text = ent["mention_text"]
        mock_ent.text = ent["mention_text"]
        mock_ent.label = ent.get("label", "")
        mock_ent.start = ent.get("start", 0)
        mock_ent.end = ent.get("end", 0)
        mock_ent.score = ent.get("score", 0.9)
        mock_ent.text_idx = ent.get("text_idx", 0)

        if ent.get("entity_id"):
            linked = MagicMock()
            linked.entity_id = ent["entity_id"]
            linked.label = ent.get("kb_label", ent["mention_text"])
            linked.score = ent.get("score", 0.9)
            mock_ent.linked_entity = linked
        else:
            mock_ent.linked_entity = None

        mock_entities.append(mock_ent)

    l0_result.entities = mock_entities
    executor.execute.return_value = {"l0_result": l0_result}
    return executor


class TestBuildPipelineWithLinker:
    @patch("grapsit.construct.graph_writer.create_store")
    @patch("grapsit.construct.ner_gliner.NERGLiNERProcessor._load_model")
    def test_ner_plus_linker_pipeline(self, mock_ner_load, mock_create_store):
        """Test chunker -> NER -> linker -> graph_writer."""
        mock_store = MagicMock()
        mock_create_store.return_value = mock_store

        glinker_executor = _make_mock_glinker_executor([
            {"mention_text": "Einstein", "entity_id": "Q937"},
            {"mention_text": "Ulm", "entity_id": "Q3012"},
        ])

        builder = BuildConfigBuilder(name="linked_pipeline")
        builder.chunker(method="sentence")
        builder.ner_gliner(model="test-ner", labels=["person", "location"])
        builder.linker(executor=glinker_executor)
        builder.graph_writer()

        executor = builder.build()

        # Mock NER
        ner_proc = executor.processors["ner"]
        ner_proc._model = MagicMock()
        ner_proc._model.inference.return_value = [
            [
                {"text": "Einstein", "label": "person", "start": 0, "end": 8, "score": 0.95},
                {"text": "Ulm", "label": "location", "start": 21, "end": 24, "score": 0.85},
            ],
        ]

        result = executor.execute({"texts": ["Einstein was born in Ulm."]})

        assert result.has("ner_result")
        assert result.has("linker_result")
        assert result.has("writer_result")

        # Check linker output has linked_entity_id
        linker_result = result.get("linker_result")
        linked_ents = linker_result["entities"][0]
        assert linked_ents[0].linked_entity_id == "Q937"
        assert linked_ents[1].linked_entity_id == "Q3012"

        # Check graph_writer used linked IDs for dedup
        writer_result = result.get("writer_result")
        assert writer_result["entity_count"] == 2
        entity_map = writer_result["entity_map"]
        assert "Q937" in entity_map
        assert "Q3012" in entity_map

    @patch("grapsit.construct.graph_writer.create_store")
    def test_linker_only_pipeline(self, mock_create_store):
        """Test chunker -> linker -> graph_writer (no NER, end-to-end)."""
        mock_store = MagicMock()
        mock_create_store.return_value = mock_store

        glinker_executor = _make_mock_glinker_executor([
            {"mention_text": "Einstein", "label": "person", "entity_id": "Q937",
             "start": 0, "end": 8, "text_idx": 0},
            {"mention_text": "Ulm", "label": "location", "entity_id": "Q3012",
             "start": 21, "end": 24, "text_idx": 0},
        ])

        builder = BuildConfigBuilder(name="linker_only")
        builder.chunker(method="sentence")
        builder.linker(executor=glinker_executor)
        builder.graph_writer()

        executor = builder.build()
        result = executor.execute({"texts": ["Einstein was born in Ulm."]})

        assert result.has("linker_result")
        assert result.has("writer_result")

        linker_result = result.get("linker_result")
        assert len(linker_result["entities"][0]) == 2
        assert linker_result["entities"][0][0].linked_entity_id == "Q937"

        writer_result = result.get("writer_result")
        assert writer_result["entity_count"] == 2

    @patch("grapsit.construct.graph_writer.create_store")
    @patch("grapsit.construct.ner_gliner.NERGLiNERProcessor._load_model")
    def test_ner_linker_relex_pipeline(self, mock_ner_load, mock_create_store):
        """Test chunker -> NER -> linker -> relex -> graph_writer."""
        mock_store = MagicMock()
        mock_create_store.return_value = mock_store

        glinker_executor = _make_mock_glinker_executor([
            {"mention_text": "Einstein", "entity_id": "Q937"},
            {"mention_text": "Ulm", "entity_id": "Q3012"},
        ])

        builder = BuildConfigBuilder(name="full_linked")
        builder.chunker(method="sentence")
        builder.ner_gliner(model="test-ner", labels=["person", "location"])
        builder.linker(executor=glinker_executor)
        builder.relex_llm(
            api_key="test", model="test-model",
            entity_labels=["person", "location"],
            relation_labels=["born in"],
        )
        builder.graph_writer()

        executor = builder.build()

        # Mock NER
        ner_proc = executor.processors["ner"]
        ner_proc._model = MagicMock()
        ner_proc._model.inference.return_value = [
            [
                {"text": "Einstein", "label": "person", "start": 0, "end": 8, "score": 0.95},
                {"text": "Ulm", "label": "location", "start": 21, "end": 24, "score": 0.85},
            ],
        ]

        # Mock relex LLM
        relex_proc = executor.processors["relex"]
        relex_proc._client = MagicMock()
        relex_proc._client.complete.return_value = json.dumps({"relations": [
            {"head": "Einstein", "tail": "Ulm", "relation": "born in"},
        ]})

        result = executor.execute({"texts": ["Einstein was born in Ulm."]})

        assert result.has("linker_result")
        assert result.has("relex_result")
        assert result.has("writer_result")

        writer_result = result.get("writer_result")
        assert writer_result["entity_count"] >= 2
        assert writer_result["relation_count"] >= 1


class TestQueryPipelineWithLinker:
    def _mock_store(self):
        store = MagicMock()
        store.get_entity_by_label.side_effect = lambda label: {
            "einstein": {"id": "Q937", "label": "Einstein", "entity_type": "person"},
            "ulm": {"id": "Q3012", "label": "Ulm", "entity_type": "location"},
        }.get(label.strip().lower())

        store.get_entity_by_id.side_effect = lambda eid: {
            "Q937": {"id": "Q937", "label": "Einstein", "entity_type": "person"},
            "Q3012": {"id": "Q3012", "label": "Ulm", "entity_type": "location"},
        }.get(eid)

        store.get_subgraph.return_value = {
            "entities": [
                {"id": "Q937", "label": "Einstein", "entity_type": "person"},
                {"id": "Q3012", "label": "Ulm", "entity_type": "location"},
            ],
            "relations": [
                {"head": "Q937", "tail": "Q3012", "type": "BORN_IN", "score": 0.8},
            ],
        }
        store.get_chunks_for_entity.return_value = [
            {"id": "c1", "document_id": "d1", "text": "Einstein was born in Ulm.", "index": 0},
        ]
        return store

    @patch("grapsit.query.chunk_retriever.ChunkRetrieverProcessor._ensure_store")
    @patch("grapsit.query.retriever.RetrieverProcessor._ensure_store")
    @patch("grapsit.query.parser.QueryParserProcessor._load_gliner")
    def test_parser_plus_linker_pipeline(self, mock_load, mock_ret_store, mock_chunk_store):
        """Test query_parser -> linker -> retriever -> chunk_retriever."""
        glinker_executor = _make_mock_glinker_executor([
            {"mention_text": "Einstein", "entity_id": "Q937"},
        ])

        builder = QueryConfigBuilder(name="query_linked")
        builder.query_parser(method="gliner", labels=["person", "location"])
        builder.linker(executor=glinker_executor)
        builder.retriever(neo4j_uri="bolt://localhost:7687")
        builder.chunk_retriever()

        executor = builder.build()

        # Mock parser
        parser_proc = executor.processors["query_parser"]
        parser_proc._model = MagicMock()
        parser_proc._model.inference.return_value = [[
            {"text": "Einstein", "label": "person", "start": 10, "end": 18, "score": 0.95},
        ]]

        # Mock store
        mock_store = self._mock_store()
        executor.processors["retriever"]._store = mock_store
        executor.processors["chunk_retriever"]._store = mock_store

        ctx = executor.execute({"query": "Where was Einstein born?"})

        assert ctx.has("parser_result")
        assert ctx.has("linker_result")
        assert ctx.has("retriever_result")

        # Retriever should use linked entity ID
        linker_result = ctx.get("linker_result")
        assert linker_result["entities"][0].linked_entity_id == "Q937"

        # Verify retriever looked up by ID
        mock_store.get_entity_by_id.assert_called_with("Q937")
