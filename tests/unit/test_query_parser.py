"""Tests for query parser processor."""

import json
import pytest
from unittest.mock import MagicMock, patch

from grapsit.query.parser import QueryParserProcessor


class TestQueryParserGLiNER:
    @patch("grapsit.query.parser.QueryParserProcessor._load_gliner")
    def test_basic_parsing(self, mock_load):
        proc = QueryParserProcessor({
            "method": "gliner",
            "model": "test-model",
            "labels": ["person", "location"],
            "threshold": 0.3,
        })
        proc._model = MagicMock()
        proc._model.inference.return_value = [[
            {"text": "Einstein", "label": "person", "start": 10, "end": 18, "score": 0.95},
            {"text": "Ulm", "label": "location", "start": 27, "end": 30, "score": 0.85},
        ]]

        result = proc(query="Where was Einstein born in Ulm?")

        assert result["query"] == "Where was Einstein born in Ulm?"
        assert len(result["entities"]) == 2
        assert result["entities"][0].text == "Einstein"
        assert result["entities"][0].label == "person"
        assert result["entities"][1].text == "Ulm"

    @patch("grapsit.query.parser.QueryParserProcessor._load_gliner")
    def test_no_entities_found(self, mock_load):
        proc = QueryParserProcessor({
            "method": "gliner",
            "labels": ["person"],
        })
        proc._model = MagicMock()
        proc._model.inference.return_value = [[]]

        result = proc(query="What is the meaning of life?")

        assert result["query"] == "What is the meaning of life?"
        assert result["entities"] == []

    @patch("grapsit.query.parser.QueryParserProcessor._load_gliner")
    def test_labels_passed_to_model(self, mock_load):
        proc = QueryParserProcessor({
            "method": "gliner",
            "labels": ["person", "organization"],
            "threshold": 0.5,
        })
        proc._model = MagicMock()
        proc._model.inference.return_value = [[]]

        proc(query="test query")

        proc._model.inference.assert_called_once_with(
            texts=["test query"],
            labels=["person", "organization"],
            threshold=0.5,
            flat_ner=True,
        )


class TestQueryParserLLM:
    def test_basic_llm_parsing(self):
        proc = QueryParserProcessor({
            "method": "llm",
            "api_key": "test",
            "model": "test-model",
            "labels": ["person", "location"],
        })
        proc._client = MagicMock()
        proc._client.complete.return_value = json.dumps({"entities": [
            {"text": "Einstein", "label": "person"},
            {"text": "Germany", "label": "location"},
        ]})

        result = proc(query="Was Einstein born in Germany?")

        assert result["query"] == "Was Einstein born in Germany?"
        assert len(result["entities"]) == 2
        assert result["entities"][0].text == "Einstein"
        assert result["entities"][0].label == "person"
        assert result["entities"][0].score == 1.0

    def test_llm_no_entities(self):
        proc = QueryParserProcessor({
            "method": "llm",
            "api_key": "test",
            "labels": ["person"],
        })
        proc._client = MagicMock()
        proc._client.complete.return_value = json.dumps({"entities": []})

        result = proc(query="What is 2+2?")
        assert result["entities"] == []

    def test_llm_label_filtering(self):
        proc = QueryParserProcessor({
            "method": "llm",
            "api_key": "test",
            "labels": ["person"],
        })
        proc._client = MagicMock()
        proc._client.complete.return_value = json.dumps({"entities": [
            {"text": "Einstein", "label": "person"},
            {"text": "Ulm", "label": "location"},
        ]})

        result = proc(query="Einstein in Ulm")
        assert len(result["entities"]) == 1
        assert result["entities"][0].text == "Einstein"

    def test_llm_failure_returns_empty(self):
        proc = QueryParserProcessor({
            "method": "llm",
            "api_key": "test",
            "labels": ["person"],
        })
        proc._client = MagicMock()
        proc._client.complete.side_effect = Exception("API error")

        result = proc(query="test query")
        assert result["entities"] == []

    def test_llm_offsets_computed(self):
        proc = QueryParserProcessor({
            "method": "llm",
            "api_key": "test",
            "labels": ["person"],
        })
        proc._client = MagicMock()
        proc._client.complete.return_value = json.dumps({"entities": [
            {"text": "Einstein", "label": "person"},
        ]})

        result = proc(query="Was Einstein a physicist?")
        mention = result["entities"][0]
        assert mention.start == 4
        assert mention.end == 12
