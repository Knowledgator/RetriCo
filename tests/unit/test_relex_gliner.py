"""Tests for relex GLiNER processor (mocked model)."""

import pytest
from unittest.mock import MagicMock, patch, call

from grapsit.construct.relex_gliner import RelexGLiNERProcessor
from grapsit.extraction.utils import mentions_to_gliner_spans as _mentions_to_gliner_spans
from grapsit.models.document import Chunk
from grapsit.models.entity import EntityMention


class TestMentionsToGlinerSpans:
    def test_from_entity_mentions(self):
        mentions = [[
            EntityMention(text="Einstein", label="person", start=0, end=8, score=0.95),
            EntityMention(text="Ulm", label="location", start=21, end=24, score=0.85),
        ]]
        spans = _mentions_to_gliner_spans(mentions)

        assert len(spans) == 1
        assert len(spans[0]) == 2
        assert spans[0][0] == {"start": 0, "end": 8, "text": "Einstein", "label": "person", "score": 0.95}
        assert spans[0][1] == {"start": 21, "end": 24, "text": "Ulm", "label": "location", "score": 0.85}

    def test_from_dicts(self):
        mentions = [[
            {"text": "Einstein", "label": "person", "start": 0, "end": 8, "score": 0.95},
        ]]
        spans = _mentions_to_gliner_spans(mentions)
        assert spans[0][0]["text"] == "Einstein"

    def test_empty(self):
        assert _mentions_to_gliner_spans([]) == []
        assert _mentions_to_gliner_spans([[]]) == [[]]


class TestRelexGLiNERProcessor:
    def test_standalone_mode(self):
        """Without entities input — model does its own NER + relex."""
        proc = RelexGLiNERProcessor({
            "model": "test",
            "entity_labels": ["person", "location"],
            "relation_labels": ["born in"],
        })
        proc._engine._model = MagicMock()
        proc._engine._model.inference.return_value = (
            [[{"text": "Einstein", "label": "person", "start": 0, "end": 8, "score": 0.9},
              {"text": "Ulm", "label": "location", "start": 21, "end": 24, "score": 0.8}]],
            [[{
                "head": {"entity_idx": 0},
                "tail": {"entity_idx": 1},
                "relation": "born in",
                "score": 0.8,
            }]],
        )

        chunks = [Chunk(id="c1", text="Einstein was born in Ulm.")]
        result = proc(chunks=chunks)

        # Should NOT pass input_spans when no entities provided
        call_kwargs = proc._engine._model.inference.call_args[1]
        assert "input_spans" not in call_kwargs
        assert call_kwargs["batch_size"] == 8

        assert len(result["relations"]) == 1
        rel = result["relations"][0][0]
        assert rel.relation_type == "born in"
        assert rel.head_text == "Einstein"
        assert rel.head_label == "person"
        assert rel.tail_text == "Ulm"
        assert rel.tail_label == "location"

        # Entities should also be returned
        assert len(result["entities"]) == 1
        assert len(result["entities"][0]) == 2

    def test_with_preextracted_entities(self):
        """With entities input — passes input_spans to model."""
        proc = RelexGLiNERProcessor({
            "model": "test",
            "entity_labels": ["person", "location"],
            "relation_labels": ["born in"],
        })
        proc._engine._model = MagicMock()
        proc._engine._model.inference.return_value = (
            [[{"text": "Einstein", "label": "person", "start": 0, "end": 8, "score": 0.9},
              {"text": "Ulm", "label": "location", "start": 21, "end": 24, "score": 0.8}]],
            [[{
                "head": {"entity_idx": 0},
                "tail": {"entity_idx": 1},
                "relation": "born in",
                "score": 0.8,
            }]],
        )

        chunks = [Chunk(id="c1", text="Einstein was born in Ulm.")]
        pre_entities = [[
            EntityMention(text="Einstein", label="person", start=0, end=8, score=0.95, chunk_id="c1"),
            EntityMention(text="Ulm", label="location", start=21, end=24, score=0.85, chunk_id="c1"),
        ]]

        result = proc(chunks=chunks, entities=pre_entities)

        # Should pass input_spans when entities are provided
        call_kwargs = proc._engine._model.inference.call_args[1]
        assert "input_spans" in call_kwargs
        spans = call_kwargs["input_spans"]
        assert len(spans) == 1
        assert len(spans[0]) == 2
        assert spans[0][0]["text"] == "Einstein"
        assert spans[0][1]["text"] == "Ulm"

        assert len(result["relations"]) == 1
        assert result["relations"][0][0].head_text == "Einstein"
        assert result["relations"][0][0].head_label == "person"

    def test_with_dict_entities(self):
        """Entities as plain dicts (e.g. from LLM NER) are also converted."""
        proc = RelexGLiNERProcessor({
            "model": "test",
            "entity_labels": ["person"],
            "relation_labels": ["knows"],
        })
        proc._engine._model = MagicMock()
        proc._engine._model.inference.return_value = (
            [[{"text": "Alice", "label": "person", "start": 0, "end": 5, "score": 0.9}]],
            [[]],
        )

        dict_entities = [[
            {"text": "Alice", "label": "person", "start": 0, "end": 5, "score": 0.9},
        ]]
        result = proc(chunks=[Chunk(id="c1", text="Alice knows Bob.")], entities=dict_entities)

        call_kwargs = proc._engine._model.inference.call_args[1]
        assert "input_spans" in call_kwargs
        assert call_kwargs["input_spans"][0][0]["text"] == "Alice"

    def test_multiple_chunks_batched(self):
        """All chunks processed in a single batched inference call."""
        proc = RelexGLiNERProcessor({
            "model": "test",
            "entity_labels": ["person", "location"],
            "relation_labels": ["born in"],
            "batch_size": 4,
        })
        proc._engine._model = MagicMock()
        proc._engine._model.inference.return_value = (
            [
                [{"text": "Einstein", "label": "person", "start": 0, "end": 8, "score": 0.9}],
                [{"text": "Newton", "label": "person", "start": 0, "end": 6, "score": 0.85}],
            ],
            [[], []],
        )

        chunks = [
            Chunk(id="c1", text="Einstein was born in Ulm."),
            Chunk(id="c2", text="Newton worked at Cambridge."),
        ]
        result = proc(chunks=chunks)

        # Single batched call
        proc._engine._model.inference.assert_called_once()
        call_kwargs = proc._engine._model.inference.call_args[1]
        assert call_kwargs["texts"] == ["Einstein was born in Ulm.", "Newton worked at Cambridge."]
        assert call_kwargs["batch_size"] == 4

        assert len(result["entities"]) == 2
        assert result["entities"][0][0].text == "Einstein"
        assert result["entities"][0][0].chunk_id == "c1"
        assert result["entities"][1][0].text == "Newton"
        assert result["entities"][1][0].chunk_id == "c2"
