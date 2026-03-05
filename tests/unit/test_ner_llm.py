"""Tests for NER LLM processor (mocked LLM client)."""

import json
import pytest
from unittest.mock import MagicMock, patch

from retrico.construct.ner_llm import NERLLMProcessor
from retrico.extraction.utils import (
    parse_entities_json as _parse_entities_json,
    find_entity_offsets as _find_entity_offsets,
)
from retrico.models.document import Chunk


class TestParseEntitiesJson:
    def test_plain_array(self):
        raw = '[{"text": "Einstein", "label": "person", "start": 0, "end": 8}]'
        result = _parse_entities_json(raw)
        assert len(result) == 1
        assert result[0]["text"] == "Einstein"

    def test_wrapped_object(self):
        raw = '{"entities": [{"text": "Ulm", "label": "location"}]}'
        result = _parse_entities_json(raw)
        assert len(result) == 1
        assert result[0]["text"] == "Ulm"

    def test_markdown_code_block(self):
        raw = '```json\n[{"text": "Einstein", "label": "person"}]\n```'
        result = _parse_entities_json(raw)
        assert len(result) == 1

    def test_invalid_json(self):
        result = _parse_entities_json("not json at all")
        assert result == []

    def test_empty_array(self):
        assert _parse_entities_json("[]") == []


class TestFindEntityOffsets:
    def test_exact_match_with_hint(self):
        text = "Einstein was born in Ulm."
        start, end = _find_entity_offsets(text, "Einstein", 0)
        assert start == 0
        assert end == 8

    def test_search_fallback(self):
        text = "Einstein was born in Ulm."
        start, end = _find_entity_offsets(text, "Ulm", -1)
        assert start == 21
        assert end == 24

    def test_case_insensitive_fallback(self):
        text = "Einstein was born in Ulm."
        start, end = _find_entity_offsets(text, "ulm", -1)
        assert start == 21
        assert end == 24

    def test_not_found(self):
        text = "Hello world."
        start, end = _find_entity_offsets(text, "xyz", -1)
        assert start == 0
        assert end == 0


class TestNERLLMProcessor:
    def _make_processor(self, **config_overrides):
        config = {
            "api_key": "test-key",
            "model": "test-model",
            "labels": ["person", "location"],
        }
        config.update(config_overrides)
        proc = NERLLMProcessor(config)
        proc._engine._client = MagicMock()
        return proc

    def test_basic_ner(self):
        proc = self._make_processor()
        proc._engine._client.complete.return_value = json.dumps({"entities": [
            {"text": "Einstein", "label": "person", "start": 0, "end": 8},
            {"text": "Ulm", "label": "location", "start": 21, "end": 24},
        ]})

        chunks = [Chunk(id="c1", text="Einstein was born in Ulm.")]
        result = proc(chunks=chunks)

        assert len(result["entities"]) == 1
        assert len(result["entities"][0]) == 2
        assert result["entities"][0][0].text == "Einstein"
        assert result["entities"][0][0].label == "person"
        assert result["entities"][0][0].chunk_id == "c1"
        assert result["entities"][0][0].score == 1.0
        assert result["entities"][0][1].text == "Ulm"

    def test_empty_chunks(self):
        proc = self._make_processor()
        result = proc(chunks=[])
        assert result["entities"] == []

    def test_multiple_chunks(self):
        proc = self._make_processor()
        proc._engine._client.complete.side_effect = [
            json.dumps([{"text": "Alice", "label": "person", "start": 0, "end": 5}]),
            json.dumps([{"text": "Bob", "label": "person", "start": 0, "end": 3}]),
        ]

        chunks = [
            Chunk(id="c1", text="Alice went home."),
            Chunk(id="c2", text="Bob stayed."),
        ]
        result = proc(chunks=chunks)

        assert len(result["entities"]) == 2
        assert result["entities"][0][0].text == "Alice"
        assert result["entities"][0][0].chunk_id == "c1"
        assert result["entities"][1][0].text == "Bob"
        assert result["entities"][1][0].chunk_id == "c2"

    def test_filters_unknown_labels(self):
        proc = self._make_processor()
        proc._engine._client.complete.return_value = json.dumps([
            {"text": "Einstein", "label": "person", "start": 0, "end": 8},
            {"text": "1879", "label": "date", "start": 40, "end": 44},
        ])

        chunks = [Chunk(id="c1", text="Einstein was born in Ulm in 1879.")]
        result = proc(chunks=chunks)

        # "date" not in labels, should be filtered
        assert len(result["entities"][0]) == 1
        assert result["entities"][0][0].text == "Einstein"

    def test_api_error_returns_empty(self):
        proc = self._make_processor()
        proc._engine._client.complete.side_effect = Exception("API error")

        chunks = [Chunk(id="c1", text="Einstein was born in Ulm.")]
        result = proc(chunks=chunks)

        assert len(result["entities"]) == 1
        assert result["entities"][0] == []

    def test_offset_search_fallback(self):
        """When LLM omits offsets, they are found by text search."""
        proc = self._make_processor()
        proc._engine._client.complete.return_value = json.dumps([
            {"text": "Einstein", "label": "person"},
        ])

        chunks = [Chunk(id="c1", text="Einstein was born in Ulm.")]
        result = proc(chunks=chunks)

        mention = result["entities"][0][0]
        assert mention.start == 0
        assert mention.end == 8

    def test_no_labels_open_ended(self):
        """When labels are empty, LLM infers entity types freely."""
        proc = self._make_processor(labels=[])
        proc._engine._client.complete.return_value = json.dumps({"entities": [
            {"text": "Einstein", "label": "scientist"},
            {"text": "Ulm", "label": "city"},
        ]})

        chunks = [Chunk(id="c1", text="Einstein was born in Ulm.")]
        result = proc(chunks=chunks)

        # All entities accepted regardless of label
        assert len(result["entities"][0]) == 2
        assert result["entities"][0][0].label == "scientist"
        assert result["entities"][0][1].label == "city"

    def test_no_labels_prompt_wording(self):
        """Open-ended prompt should not mention specific labels."""
        proc = self._make_processor(labels=[])
        proc._engine._client.complete.return_value = json.dumps({"entities": []})

        proc(chunks=[Chunk(id="c1", text="Hello world.")])

        call_args = proc._engine._client.complete.call_args[0][0]
        user_msg = call_args[1]["content"]
        assert "No specific entity labels" in user_msg
        assert "Entity labels:" not in user_msg

    def test_structured_output_used_by_default(self):
        """Default: uses json_schema response format."""
        proc = self._make_processor()
        proc._engine._client.complete.return_value = json.dumps({"entities": [
            {"text": "Einstein", "label": "person"},
        ]})

        proc(chunks=[Chunk(id="c1", text="Einstein was a physicist.")])

        call_kwargs = proc._engine._client.complete.call_args[1]
        assert call_kwargs["response_format"]["type"] == "json_schema"
        assert "json_schema" in call_kwargs["response_format"]

    def test_structured_output_disabled(self):
        """structured_output=False uses json_object mode."""
        proc = self._make_processor(structured_output=False)
        proc._engine._client.complete.return_value = json.dumps({"entities": [
            {"text": "Einstein", "label": "person"},
        ]})

        proc(chunks=[Chunk(id="c1", text="Einstein was a physicist.")])

        call_kwargs = proc._engine._client.complete.call_args[1]
        assert call_kwargs["response_format"] == {"type": "json_object"}

    def test_structured_output_fallback_on_error(self):
        """If structured output fails, falls back to json_object for subsequent calls."""
        proc = self._make_processor()

        # First call: structured output raises, fallback succeeds
        proc._engine._client.complete.side_effect = [
            Exception("Unsupported response_format"),  # structured attempt
            json.dumps({"entities": [{"text": "Einstein", "label": "person"}]}),  # json_object fallback
            json.dumps({"entities": [{"text": "Newton", "label": "person"}]}),  # second chunk, json_object directly
        ]

        chunks = [
            Chunk(id="c1", text="Einstein was a physicist."),
            Chunk(id="c2", text="Newton was a physicist."),
        ]
        result = proc(chunks=chunks)

        assert len(result["entities"][0]) == 1
        assert result["entities"][0][0].text == "Einstein"
        assert len(result["entities"][1]) == 1
        assert result["entities"][1][0].text == "Newton"

        # After fallback, _structured_failed should be True
        assert proc._engine._structured_failed is True
        # Third call should go directly to json_object (no structured attempt)
        last_call_kwargs = proc._engine._client.complete.call_args_list[-1][1]
        assert last_call_kwargs["response_format"] == {"type": "json_object"}
