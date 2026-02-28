"""Unit tests for the entity linker processor."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from grapsit.models.document import Chunk
from grapsit.models.entity import EntityMention
from grapsit.construct.linker import (
    EntityLinkerProcessor,
    _is_flat_entity_list,
    _mentions_to_glinker_spans,
    _build_linked_map,
    _apply_links,
)


def _make_mock_executor(linked_entities=None):
    """Create a mock GLinker executor with l0_result."""
    executor = MagicMock()
    l0_result = MagicMock()

    if linked_entities is None:
        linked_entities = []

    mock_entities = []
    for ent in linked_entities:
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


class TestHelpers:
    def test_is_flat_entity_list_empty(self):
        assert _is_flat_entity_list([]) is True

    def test_is_flat_entity_list_flat(self):
        mentions = [EntityMention(text="A", label="person")]
        assert _is_flat_entity_list(mentions) is True

    def test_is_flat_entity_list_nested(self):
        mentions = [[EntityMention(text="A", label="person")]]
        assert _is_flat_entity_list(mentions) is False

    def test_mentions_to_glinker_spans_from_mentions(self):
        mentions = [
            EntityMention(text="Einstein", label="person", start=0, end=8),
            EntityMention(text="Ulm", label="location", start=21, end=24),
        ]
        spans = _mentions_to_glinker_spans(mentions)
        assert len(spans) == 2
        assert spans[0] == {"text": "Einstein", "label": "person", "start": 0, "end": 8}
        assert spans[1] == {"text": "Ulm", "label": "location", "start": 21, "end": 24}

    def test_mentions_to_glinker_spans_from_dicts(self):
        mentions = [{"text": "Einstein", "label": "person", "start": 0, "end": 8}]
        spans = _mentions_to_glinker_spans(mentions)
        assert spans[0]["text"] == "Einstein"

    def test_build_linked_map(self):
        l0_result = MagicMock()
        ent1 = MagicMock()
        ent1.mention_text = "Einstein"
        ent1.linked_entity = MagicMock()
        ent1.linked_entity.entity_id = "Q937"
        ent2 = MagicMock()
        ent2.mention_text = "Ulm"
        ent2.linked_entity = None
        l0_result.entities = [ent1, ent2]

        linked_map = _build_linked_map(l0_result)
        assert linked_map == {"Einstein": "Q937"}

    def test_apply_links(self):
        mentions = [
            EntityMention(text="Einstein", label="person"),
            EntityMention(text="Ulm", label="location"),
        ]
        linked_map = {"Einstein": "Q937"}
        result = _apply_links(mentions, linked_map)
        assert result[0].linked_entity_id == "Q937"
        assert result[1].linked_entity_id is None


class TestEntityLinkerProcessor:
    def test_with_prebuilt_executor(self):
        executor = _make_mock_executor([
            {"mention_text": "Einstein", "entity_id": "Q937"},
        ])
        proc = EntityLinkerProcessor({"executor": executor})

        chunks = [Chunk(id="c1", text="Einstein was a physicist.", index=0)]
        entities = [[EntityMention(text="Einstein", label="person", start=0, end=8, chunk_id="c1")]]

        result = proc(entities=entities, chunks=chunks)

        assert result["entities"][0][0].linked_entity_id == "Q937"
        assert result["chunks"] == chunks
        executor.execute.assert_called_once()

    def test_lazy_loading(self):
        """Executor should be created on first call, not at init."""
        proc = EntityLinkerProcessor({"model": "test-model"})
        assert proc._executor is None

    @patch("grapsit.construct.linker.EntityLinkerProcessor._ensure_executor")
    def test_build_mode_with_entities(self, mock_ensure):
        """Build mode: list-of-lists entities from NER."""
        executor = _make_mock_executor([
            {"mention_text": "Einstein", "entity_id": "Q937"},
            {"mention_text": "Ulm", "entity_id": "Q3012"},
        ])
        proc = EntityLinkerProcessor({"executor": executor})

        chunks = [Chunk(id="c1", text="Einstein was born in Ulm.", index=0)]
        entities = [[
            EntityMention(text="Einstein", label="person", start=0, end=8, chunk_id="c1"),
            EntityMention(text="Ulm", label="location", start=21, end=24, chunk_id="c1"),
        ]]

        result = proc(entities=entities, chunks=chunks)

        assert len(result["entities"]) == 1
        assert result["entities"][0][0].linked_entity_id == "Q937"
        assert result["entities"][0][1].linked_entity_id == "Q3012"

    @patch("grapsit.construct.linker.EntityLinkerProcessor._ensure_executor")
    def test_build_mode_without_entities_end_to_end(self, mock_ensure):
        """Build mode: no upstream NER, GLinker does end-to-end."""
        executor = _make_mock_executor([
            {"mention_text": "Einstein", "label": "person", "entity_id": "Q937",
             "start": 0, "end": 8, "text_idx": 0},
        ])
        proc = EntityLinkerProcessor({"executor": executor})

        chunks = [Chunk(id="c1", text="Einstein was a physicist.", index=0)]

        result = proc(chunks=chunks)

        assert len(result["entities"]) == 1
        assert len(result["entities"][0]) == 1
        assert result["entities"][0][0].text == "Einstein"
        assert result["entities"][0][0].linked_entity_id == "Q937"

    @patch("grapsit.construct.linker.EntityLinkerProcessor._ensure_executor")
    def test_query_mode_with_flat_entities(self, mock_ensure):
        """Query mode: flat entity list from parser."""
        executor = _make_mock_executor([
            {"mention_text": "Einstein", "entity_id": "Q937"},
        ])
        proc = EntityLinkerProcessor({"executor": executor})

        entities = [
            EntityMention(text="Einstein", label="person", start=10, end=18),
        ]

        result = proc(entities=entities, query="Where was Einstein born?")

        assert result["query"] == "Where was Einstein born?"
        assert result["entities"][0].linked_entity_id == "Q937"

    @patch("grapsit.construct.linker.EntityLinkerProcessor._ensure_executor")
    def test_query_mode_without_entities(self, mock_ensure):
        """Query mode: no parser, GLinker does end-to-end on query."""
        executor = _make_mock_executor([
            {"mention_text": "Einstein", "label": "person", "entity_id": "Q937",
             "start": 10, "end": 18, "text_idx": 0},
        ])
        proc = EntityLinkerProcessor({"executor": executor})

        result = proc(query="Where was Einstein born?")

        assert result["query"] == "Where was Einstein born?"
        assert len(result["entities"]) >= 1
        assert result["entities"][0].text == "Einstein"
        assert result["entities"][0].linked_entity_id == "Q937"

    @patch("grapsit.construct.linker.EntityLinkerProcessor._ensure_executor")
    def test_unlinked_entities_pass_through(self, mock_ensure):
        """Entities not found in KB should have linked_entity_id=None."""
        executor = _make_mock_executor([
            {"mention_text": "Einstein", "entity_id": "Q937"},
            # Ulm is not linked
        ])
        proc = EntityLinkerProcessor({"executor": executor})

        chunks = [Chunk(id="c1", text="Einstein was born in Ulm.", index=0)]
        entities = [[
            EntityMention(text="Einstein", label="person", start=0, end=8, chunk_id="c1"),
            EntityMention(text="Ulm", label="location", start=21, end=24, chunk_id="c1"),
        ]]

        result = proc(entities=entities, chunks=chunks)

        assert result["entities"][0][0].linked_entity_id == "Q937"
        assert result["entities"][0][1].linked_entity_id is None

    @patch("grapsit.construct.linker.EntityLinkerProcessor._ensure_executor")
    def test_empty_entities(self, mock_ensure):
        """Empty entity list returns empty."""
        executor = _make_mock_executor([])
        proc = EntityLinkerProcessor({"executor": executor})

        result = proc(entities=[], chunks=[])
        assert result["entities"] == []
        assert result["chunks"] == []

    @patch("grapsit.construct.linker.EntityLinkerProcessor._ensure_executor")
    def test_no_inputs_returns_empty(self, mock_ensure):
        """No inputs at all returns empty."""
        proc = EntityLinkerProcessor({})
        result = proc()
        assert result["entities"] == []
        assert result["chunks"] == []

    def test_from_parameters_initialization(self):
        """Test creating executor from model/threshold parameters."""
        import sys
        mock_glinker = MagicMock()
        sys.modules["glinker"] = mock_glinker

        try:
            mock_executor = MagicMock()
            mock_glinker.ProcessorFactory.create_simple.return_value = mock_executor

            proc = EntityLinkerProcessor({
                "model": "test-model",
                "threshold": 0.7,
                "entities": [{"entity_id": "Q1", "label": "Test"}],
            })
            proc._ensure_executor(external_entities=True)

            mock_glinker.ProcessorFactory.create_simple.assert_called_once_with(
                model_name="test-model",
                threshold=0.7,
                external_entities=True,
            )
            mock_executor.load_entities.assert_called_once_with(
                [{"entity_id": "Q1", "label": "Test"}]
            )
        finally:
            del sys.modules["glinker"]

    def test_from_parameters_file_path(self):
        """Test loading KB from file path."""
        import sys
        mock_glinker = MagicMock()
        sys.modules["glinker"] = mock_glinker

        try:
            mock_executor = MagicMock()
            mock_glinker.ProcessorFactory.create_simple.return_value = mock_executor

            proc = EntityLinkerProcessor({
                "model": "test-model",
                "entities": "/data/entities.jsonl",
            })
            proc._ensure_executor(external_entities=False)

            mock_executor.load_entities.assert_called_once_with("/data/entities.jsonl")
        finally:
            del sys.modules["glinker"]

    def test_from_neo4j_kb(self):
        """Test loading KB from Neo4j."""
        import sys
        mock_glinker = MagicMock()
        sys.modules["glinker"] = mock_glinker

        try:
            mock_executor = MagicMock()
            mock_glinker.ProcessorFactory.create_simple.return_value = mock_executor

            with patch("grapsit.store.graph.neo4j_store.GraphDatabase") as mock_gdb:
                mock_driver = MagicMock()
                mock_session = MagicMock()

                # Mock get_all_entities query results
                mock_record1 = MagicMock()
                mock_record1.data.return_value = {"e": {"id": "e1", "label": "Einstein", "entity_type": "person"}}
                mock_record2 = MagicMock()
                mock_record2.data.return_value = {"e": {"id": "e2", "label": "Ulm", "entity_type": "location"}}

                mock_result = MagicMock()
                mock_result.__iter__ = lambda self: iter([mock_record1, mock_record2])
                mock_session.run.return_value = mock_result
                mock_session.__enter__ = lambda self: self
                mock_session.__exit__ = MagicMock(return_value=False)
                mock_driver.session.return_value = mock_session
                mock_gdb.driver.return_value = mock_driver

                proc = EntityLinkerProcessor({
                    "model": "test-model",
                    "neo4j_uri": "bolt://localhost:7687",
                })
                proc._ensure_executor(external_entities=False)

                mock_executor.load_entities.assert_called_once()
                loaded = mock_executor.load_entities.call_args[0][0]
                assert len(loaded) == 2
                assert loaded[0]["entity_id"] == "e1"
                assert loaded[0]["label"] == "Einstein"
        finally:
            del sys.modules["glinker"]

    def test_prebuilt_executor_skips_creation(self):
        """Pre-built executor should not call ProcessorFactory."""
        executor = _make_mock_executor([])
        proc = EntityLinkerProcessor({"executor": executor})
        proc._ensure_executor(external_entities=True)
        assert proc._executor is executor


class TestRegistration:
    def test_registered(self):
        from grapsit.core.registry import processor_registry
        assert "entity_linker" in processor_registry._factories
