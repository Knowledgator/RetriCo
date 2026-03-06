"""Tests for relex LLM processor (mocked LLM client)."""

import json
import pytest
from unittest.mock import MagicMock, patch

from retrico.construct.relex_llm import RelexLLMProcessor
from retrico.extraction.utils import (
    parse_standalone_json as _parse_standalone_json,
    parse_relations_json as _parse_relations_json,
    format_entities_list as _format_entities_list,
    normalize_mentions as _normalize_mentions,
)
from retrico.models.document import Chunk
from retrico.models.entity import EntityMention


class TestParseStandaloneJson:
    def test_valid_object(self):
        raw = json.dumps({
            "entities": [{"text": "Einstein", "label": "person"}],
            "relations": [{"head": "Einstein", "tail": "Ulm", "relation": "born in"}],
        })
        entities, relations = _parse_standalone_json(raw)
        assert len(entities) == 1
        assert len(relations) == 1

    def test_markdown_code_block(self):
        raw = '```json\n{"entities": [], "relations": []}\n```'
        entities, relations = _parse_standalone_json(raw)
        assert entities == []
        assert relations == []

    def test_invalid_json(self):
        entities, relations = _parse_standalone_json("not json")
        assert entities == []
        assert relations == []


class TestParseRelationsJson:
    def test_plain_array(self):
        raw = '[{"head": "Einstein", "tail": "Ulm", "relation": "born in"}]'
        result = _parse_relations_json(raw)
        assert len(result) == 1

    def test_wrapped_object(self):
        raw = '{"relations": [{"head": "A", "tail": "B", "relation": "r"}]}'
        result = _parse_relations_json(raw)
        assert len(result) == 1

    def test_invalid_json(self):
        assert _parse_relations_json("bad") == []


class TestFormatEntitiesList:
    def test_formats_mentions(self):
        mentions = [
            EntityMention(text="Einstein", label="person", start=0, end=8),
            EntityMention(text="Ulm", label="location", start=21, end=24),
        ]
        result = _format_entities_list(mentions)
        assert '"Einstein" (person)' in result
        assert '"Ulm" (location)' in result

    def test_empty(self):
        assert _format_entities_list([]) == "(none)"


class TestNormalizeMentions:
    def test_from_entity_mentions(self):
        mentions = [EntityMention(text="Einstein", label="person", start=0, end=8)]
        result = _normalize_mentions(mentions, "c1")
        assert len(result) == 1
        assert result[0].text == "Einstein"

    def test_from_dicts(self):
        dicts = [{"text": "Einstein", "label": "person", "start": 0, "end": 8}]
        result = _normalize_mentions(dicts, "c1")
        assert len(result) == 1
        assert result[0].text == "Einstein"
        assert result[0].chunk_id == "c1"


class TestRelexLLMProcessor:
    def _make_processor(self, **config_overrides):
        config = {
            "api_key": "test-key",
            "model": "test-model",
            "entity_labels": ["person", "location"],
            "relation_labels": ["born in"],
        }
        config.update(config_overrides)
        proc = RelexLLMProcessor(config)
        proc._engine._client = MagicMock()
        return proc

    def test_standalone_mode(self):
        """Without entities input — LLM extracts both, labels derived from entities."""
        proc = self._make_processor()
        proc._engine._client.complete.return_value = json.dumps({
            "entities": [
                {"text": "Einstein", "label": "person"},
                {"text": "Ulm", "label": "location"},
            ],
            "relations": [
                {"head": "Einstein", "tail": "Ulm", "relation": "born in"},
            ],
        })

        chunks = [Chunk(id="c1", text="Einstein was born in Ulm.")]
        result = proc(chunks=chunks)

        assert len(result["entities"][0]) == 2
        assert len(result["relations"][0]) == 1
        rel = result["relations"][0][0]
        assert rel.head_text == "Einstein"
        assert rel.tail_text == "Ulm"
        assert rel.relation_type == "born in"
        # Labels derived from entities section
        assert rel.head_label == "person"
        assert rel.tail_label == "location"
        assert rel.score == 1.0
        assert rel.chunk_id == ["c1"]

    def test_with_preextracted_entities(self):
        """With entities input — LLM only extracts relations, labels from entities."""
        proc = self._make_processor()
        # LLM response has no head_label/tail_label — they come from entities
        proc._engine._client.complete.return_value = json.dumps({"relations": [
            {"head": "Einstein", "tail": "Ulm", "relation": "born in"},
        ]})

        chunks = [Chunk(id="c1", text="Einstein was born in Ulm.")]
        pre_entities = [[
            EntityMention(text="Einstein", label="person", start=0, end=8, score=0.95, chunk_id="c1"),
            EntityMention(text="Ulm", label="location", start=21, end=24, score=0.85, chunk_id="c1"),
        ]]

        result = proc(chunks=chunks, entities=pre_entities)

        assert len(result["relations"][0]) == 1
        rel = result["relations"][0][0]
        assert rel.head_text == "Einstein"
        assert rel.tail_text == "Ulm"
        # Labels resolved from pre-extracted entities
        assert rel.head_label == "person"
        assert rel.tail_label == "location"
        # Pre-extracted entities passed through
        assert len(result["entities"][0]) == 2
        assert result["entities"][0][0].text == "Einstein"

    def test_with_dict_entities(self):
        """Entities as plain dicts are normalized."""
        proc = self._make_processor()
        proc._engine._client.complete.return_value = json.dumps({"relations": [
            {"head": "Alice", "tail": "Bob", "relation": "born in"},
        ]})

        dict_entities = [[
            {"text": "Alice", "label": "person", "start": 0, "end": 5, "score": 0.9},
        ]]
        result = proc(
            chunks=[Chunk(id="c1", text="Alice knows Bob.")],
            entities=dict_entities,
        )

        assert result["entities"][0][0].text == "Alice"
        assert isinstance(result["entities"][0][0], EntityMention)

    def test_multiple_chunks(self):
        proc = self._make_processor()
        proc._engine._client.complete.side_effect = [
            json.dumps({
                "entities": [{"text": "Einstein", "label": "person"}],
                "relations": [],
            }),
            json.dumps({
                "entities": [{"text": "Newton", "label": "person"}],
                "relations": [],
            }),
        ]

        chunks = [
            Chunk(id="c1", text="Einstein was a physicist."),
            Chunk(id="c2", text="Newton worked at Cambridge."),
        ]
        result = proc(chunks=chunks)

        assert len(result["entities"]) == 2
        assert result["entities"][0][0].text == "Einstein"
        assert result["entities"][0][0].chunk_id == "c1"
        assert result["entities"][1][0].text == "Newton"
        assert result["entities"][1][0].chunk_id == "c2"

    def test_api_error_standalone(self):
        proc = self._make_processor()
        proc._engine._client.complete.side_effect = Exception("API error")

        chunks = [Chunk(id="c1", text="Einstein was born in Ulm.")]
        result = proc(chunks=chunks)

        assert result["entities"][0] == []
        assert result["relations"][0] == []

    def test_api_error_with_entities(self):
        proc = self._make_processor()
        proc._engine._client.complete.side_effect = Exception("API error")

        pre_entities = [[
            EntityMention(text="Einstein", label="person", start=0, end=8, chunk_id="c1"),
        ]]
        chunks = [Chunk(id="c1", text="Einstein was born in Ulm.")]
        result = proc(chunks=chunks, entities=pre_entities)

        # Entities preserved, relations empty
        assert len(result["entities"][0]) == 1
        assert result["relations"][0] == []

    def test_no_labels_open_ended_standalone(self):
        """When labels are empty, LLM infers types freely in standalone mode."""
        proc = self._make_processor(entity_labels=[], relation_labels=[])
        proc._engine._client.complete.return_value = json.dumps({
            "entities": [
                {"text": "Einstein", "label": "scientist"},
                {"text": "Ulm", "label": "city"},
            ],
            "relations": [
                {"head": "Einstein", "tail": "Ulm", "relation": "birthplace"},
            ],
        })

        chunks = [Chunk(id="c1", text="Einstein was born in Ulm.")]
        result = proc(chunks=chunks)

        assert len(result["entities"][0]) == 2
        assert result["entities"][0][0].label == "scientist"
        assert len(result["relations"][0]) == 1
        assert result["relations"][0][0].relation_type == "birthplace"
        assert result["relations"][0][0].head_label == "scientist"
        assert result["relations"][0][0].tail_label == "city"

    def test_no_labels_prompt_wording(self):
        """Open-ended prompt should not mention specific labels."""
        proc = self._make_processor(entity_labels=[], relation_labels=[])
        proc._engine._client.complete.return_value = json.dumps({
            "entities": [], "relations": [],
        })

        proc(chunks=[Chunk(id="c1", text="Hello world.")])

        call_args = proc._engine._client.complete.call_args[0][0]
        user_msg = call_args[1]["content"]
        assert "No specific entity labels" in user_msg
        assert "No specific relation labels" in user_msg

    def test_structured_output_standalone(self):
        """Default: standalone uses json_schema with standalone schema."""
        proc = self._make_processor()
        proc._engine._client.complete.return_value = json.dumps({
            "entities": [{"text": "Einstein", "label": "person"}],
            "relations": [{"head": "Einstein", "tail": "Ulm", "relation": "born in"}],
        })

        proc(chunks=[Chunk(id="c1", text="Einstein was a physicist.")])

        call_kwargs = proc._engine._client.complete.call_args[1]
        assert call_kwargs["response_format"]["type"] == "json_schema"
        schema_name = call_kwargs["response_format"]["json_schema"]["name"]
        assert schema_name == "relex_standalone_response"

    def test_structured_output_with_entities(self):
        """Default: with-entities uses json_schema with with-entities schema."""
        proc = self._make_processor()
        proc._engine._client.complete.return_value = json.dumps({"relations": []})

        pre_entities = [[
            EntityMention(text="Einstein", label="person", start=0, end=8, chunk_id="c1"),
        ]]
        proc(chunks=[Chunk(id="c1", text="Einstein was a physicist.")], entities=pre_entities)

        call_kwargs = proc._engine._client.complete.call_args[1]
        assert call_kwargs["response_format"]["type"] == "json_schema"
        schema_name = call_kwargs["response_format"]["json_schema"]["name"]
        assert schema_name == "relex_with_entities_response"

    def test_structured_output_disabled(self):
        """structured_output=False uses json_object mode."""
        proc = self._make_processor(structured_output=False)
        proc._engine._client.complete.return_value = json.dumps({
            "entities": [], "relations": [],
        })

        proc(chunks=[Chunk(id="c1", text="Einstein was a physicist.")])

        call_kwargs = proc._engine._client.complete.call_args[1]
        assert call_kwargs["response_format"] == {"type": "json_object"}
