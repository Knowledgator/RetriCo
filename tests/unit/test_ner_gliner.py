"""Tests for NER GLiNER processor (mocked model)."""

import pytest
from unittest.mock import MagicMock, patch

from grapsit.construct.ner_gliner import NERGLiNERProcessor
from grapsit.models.document import Chunk


class TestNERGLiNERProcessor:
    def test_basic_ner(self):
        proc = NERGLiNERProcessor({"model": "test", "labels": ["person", "location"], "threshold": 0.3})
        proc._engine._model = MagicMock()
        proc._engine._model.inference.return_value = [
            [
                {"text": "Einstein", "label": "person", "start": 0, "end": 8, "score": 0.95},
                {"text": "Ulm", "label": "location", "start": 21, "end": 24, "score": 0.85},
            ]
        ]

        chunks = [Chunk(id="c1", text="Einstein was born in Ulm.")]
        result = proc(chunks=chunks)

        assert "entities" in result
        assert len(result["entities"]) == 1
        assert len(result["entities"][0]) == 2
        assert result["entities"][0][0].text == "Einstein"
        assert result["entities"][0][0].chunk_id == "c1"

        # Verify inference was called with batched texts
        call_kwargs = proc._engine._model.inference.call_args[1]
        assert call_kwargs["texts"] == ["Einstein was born in Ulm."]
        assert call_kwargs["labels"] == ["person", "location"]
        assert call_kwargs["batch_size"] == 8

    def test_empty_chunks(self):
        proc = NERGLiNERProcessor({"model": "test", "labels": ["person"]})
        proc._engine._model = MagicMock()
        proc._engine._model.inference.return_value = []
        result = proc(chunks=[])
        assert result["entities"] == []

    def test_multiple_chunks_batched(self):
        proc = NERGLiNERProcessor({"model": "test", "labels": ["person"], "batch_size": 4})
        proc._engine._model = MagicMock()
        proc._engine._model.inference.return_value = [
            [{"text": "Alice", "label": "person", "start": 0, "end": 5, "score": 0.9}],
            [{"text": "Bob", "label": "person", "start": 0, "end": 3, "score": 0.85}],
        ]

        chunks = [
            Chunk(id="c1", text="Alice went home."),
            Chunk(id="c2", text="Bob stayed."),
        ]
        result = proc(chunks=chunks)

        # Single batched call with all texts
        proc._engine._model.inference.assert_called_once()
        call_kwargs = proc._engine._model.inference.call_args[1]
        assert call_kwargs["texts"] == ["Alice went home.", "Bob stayed."]
        assert call_kwargs["batch_size"] == 4

        assert len(result["entities"]) == 2
        assert result["entities"][0][0].text == "Alice"
        assert result["entities"][0][0].chunk_id == "c1"
        assert result["entities"][1][0].text == "Bob"
        assert result["entities"][1][0].chunk_id == "c2"
