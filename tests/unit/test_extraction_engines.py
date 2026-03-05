"""Tests for extraction engines (GLiNEREngine, LLMExtractionEngine, utils)."""

import json
import pytest
from unittest.mock import MagicMock, patch

from grapsit.extraction import (
    GLiNEREngine,
    LLMExtractionEngine,
    ExtractionResult,
    strip_markdown_fences,
    parse_entities_json,
    parse_relations_json,
    parse_standalone_json,
    find_entity_offsets,
    build_labels_instruction,
    build_relation_labels_instruction,
    normalize_mentions,
    format_entities_list,
    mentions_to_gliner_spans,
)
from grapsit.models.entity import EntityMention
from grapsit.models.relation import Relation


# -- Utils tests ---------------------------------------------------------------


class TestStripMarkdownFences:
    def test_json_fence(self):
        assert strip_markdown_fences("```json\n{}\n```") == "{}"

    def test_plain_fence(self):
        assert strip_markdown_fences("```\nfoo\n```") == "foo"

    def test_no_fence(self):
        assert strip_markdown_fences("hello") == "hello"


class TestParseEntitiesJson:
    def test_plain_array(self):
        result = parse_entities_json('[{"text": "A", "label": "x"}]')
        assert len(result) == 1

    def test_wrapped(self):
        result = parse_entities_json('{"entities": [{"text": "A", "label": "x"}]}')
        assert len(result) == 1

    def test_markdown(self):
        result = parse_entities_json('```json\n[{"text": "A", "label": "x"}]\n```')
        assert len(result) == 1

    def test_invalid(self):
        assert parse_entities_json("bad") == []


class TestParseRelationsJson:
    def test_array(self):
        result = parse_relations_json('[{"head": "A", "tail": "B", "relation": "r"}]')
        assert len(result) == 1

    def test_wrapped(self):
        result = parse_relations_json('{"relations": [{"head": "A", "tail": "B", "relation": "r"}]}')
        assert len(result) == 1

    def test_invalid(self):
        assert parse_relations_json("bad") == []


class TestParseStandaloneJson:
    def test_valid(self):
        raw = json.dumps({"entities": [{"text": "A", "label": "x"}], "relations": []})
        ents, rels = parse_standalone_json(raw)
        assert len(ents) == 1
        assert rels == []

    def test_invalid(self):
        ents, rels = parse_standalone_json("bad")
        assert ents == []
        assert rels == []


class TestFindEntityOffsets:
    def test_exact(self):
        assert find_entity_offsets("Einstein was born", "Einstein", 0) == (0, 8)

    def test_search(self):
        assert find_entity_offsets("Einstein was born in Ulm", "Ulm", -1) == (21, 24)

    def test_case_insensitive(self):
        assert find_entity_offsets("Einstein was born in Ulm", "ulm", -1) == (21, 24)

    def test_not_found(self):
        assert find_entity_offsets("Hello", "xyz", -1) == (0, 0)


class TestBuildLabelsInstruction:
    def test_with_labels(self):
        instr, constraint = build_labels_instruction(["person", "location"])
        assert "person" in instr
        assert "one of the labels" in constraint

    def test_empty_labels(self):
        instr, constraint = build_labels_instruction([])
        assert "No specific" in instr
        assert constraint == ""


class TestBuildRelationLabelsInstruction:
    def test_with_labels(self):
        assert "born_in" in build_relation_labels_instruction(["born_in"])

    def test_empty(self):
        assert "No specific" in build_relation_labels_instruction([])


class TestNormalizeMentions:
    def test_entity_mentions(self):
        mentions = [EntityMention(text="A", label="x", start=0, end=1)]
        result = normalize_mentions(mentions, "c1")
        assert result[0] is mentions[0]

    def test_dicts(self):
        dicts = [{"text": "A", "label": "x", "start": 0, "end": 1}]
        result = normalize_mentions(dicts, "c1")
        assert result[0].text == "A"
        assert result[0].chunk_id == "c1"


class TestFormatEntitiesList:
    def test_formats(self):
        mentions = [EntityMention(text="Einstein", label="person", start=0, end=8)]
        result = format_entities_list(mentions)
        assert '"Einstein" (person)' in result

    def test_empty(self):
        assert format_entities_list([]) == "(none)"


class TestMentionsToGlinerSpans:
    def test_entity_mentions(self):
        mentions = [[EntityMention(text="A", label="x", start=0, end=1, score=0.9)]]
        spans = mentions_to_gliner_spans(mentions)
        assert spans[0][0] == {"start": 0, "end": 1, "text": "A", "label": "x", "score": 0.9}

    def test_dicts(self):
        mentions = [[{"text": "A", "label": "x", "start": 0, "end": 1, "score": 0.9}]]
        spans = mentions_to_gliner_spans(mentions)
        assert spans[0][0]["text"] == "A"

    def test_empty(self):
        assert mentions_to_gliner_spans([]) == []


# -- GLiNEREngine tests --------------------------------------------------------


class TestGLiNEREngine:
    def _make_engine(self, **kwargs):
        engine = GLiNEREngine(**kwargs)
        engine._model = MagicMock()
        return engine

    def test_ner_only_mode(self):
        engine = self._make_engine(labels=["person", "location"])
        assert not engine.has_relex

        engine._model.inference.return_value = [
            [{"text": "Einstein", "label": "person", "start": 0, "end": 8, "score": 0.95}],
        ]

        result = engine.extract(["Einstein was born in Ulm."])

        assert isinstance(result, ExtractionResult)
        assert len(result.entities) == 1
        assert len(result.entities[0]) == 1
        assert result.entities[0][0].text == "Einstein"
        assert result.relations == [[]]

        call_kwargs = engine._model.inference.call_args[1]
        assert call_kwargs["texts"] == ["Einstein was born in Ulm."]
        assert call_kwargs["labels"] == ["person", "location"]

    def test_relex_mode(self):
        engine = self._make_engine(
            labels=["person", "location"],
            relation_labels=["born in"],
        )
        assert engine.has_relex
        assert engine.flat_ner is False

        engine._model.inference.return_value = (
            [[
                {"text": "Einstein", "label": "person", "start": 0, "end": 8, "score": 0.9},
                {"text": "Ulm", "label": "location", "start": 21, "end": 24, "score": 0.8},
            ]],
            [[{
                "head": {"entity_idx": 0},
                "tail": {"entity_idx": 1},
                "relation": "born in",
                "score": 0.8,
            }]],
        )

        result = engine.extract(["Einstein was born in Ulm."])

        assert len(result.entities[0]) == 2
        assert len(result.relations[0]) == 1
        assert result.relations[0][0].relation_type == "born in"
        assert result.relations[0][0].head_text == "Einstein"
        assert result.relations[0][0].tail_text == "Ulm"

    def test_relex_with_preextracted_entities(self):
        engine = self._make_engine(
            labels=["person", "location"],
            relation_labels=["born in"],
        )

        engine._model.inference.return_value = (
            [[
                {"text": "Einstein", "label": "person", "start": 0, "end": 8, "score": 0.9},
                {"text": "Ulm", "label": "location", "start": 21, "end": 24, "score": 0.8},
            ]],
            [[{
                "head": {"entity_idx": 0},
                "tail": {"entity_idx": 1},
                "relation": "born in",
                "score": 0.8,
            }]],
        )

        pre_entities = [[
            EntityMention(text="Einstein", label="person", start=0, end=8, score=0.95),
            EntityMention(text="Ulm", label="location", start=21, end=24, score=0.85),
        ]]

        result = engine.extract(["Einstein was born in Ulm."], entities=pre_entities)

        call_kwargs = engine._model.inference.call_args[1]
        assert "input_spans" in call_kwargs
        assert len(call_kwargs["input_spans"][0]) == 2

    def test_empty_texts(self):
        engine = self._make_engine(labels=["person"])
        result = engine.extract([])
        assert result.entities == []
        assert result.relations == []

    def test_extract_single(self):
        engine = self._make_engine(labels=["person"])
        engine._model.inference.return_value = [
            [{"text": "Alice", "label": "person", "start": 0, "end": 5, "score": 0.9}],
        ]

        result = engine.extract_single("Alice went home.")
        assert len(result.entities) == 1
        assert result.entities[0][0].text == "Alice"

    def test_generated_label_fallback(self):
        engine = self._make_engine(labels=["person"])
        engine._model.inference.return_value = [
            [{"text": "Alice", "generated_label": ["scientist"], "start": 0, "end": 5, "score": 0.9}],
        ]

        result = engine.extract(["Alice went home."])
        assert result.entities[0][0].label == "scientist"

    def test_batch_size_passed(self):
        engine = self._make_engine(labels=["person"], batch_size=4)
        engine._model.inference.return_value = [[], []]

        engine.extract(["text1", "text2"])

        call_kwargs = engine._model.inference.call_args[1]
        assert call_kwargs["batch_size"] == 4


# -- LLMExtractionEngine tests ------------------------------------------------


class TestLLMExtractionEngine:
    def _make_engine(self, **kwargs):
        defaults = {"api_key": "test", "model": "test-model", "labels": ["person", "location"]}
        defaults.update(kwargs)
        engine = LLMExtractionEngine(**defaults)
        engine._client = MagicMock()
        return engine

    def test_ner_only_extract(self):
        engine = self._make_engine()
        assert not engine.has_relex

        engine._client.complete.return_value = json.dumps({"entities": [
            {"text": "Einstein", "label": "person"},
            {"text": "Ulm", "label": "location"},
        ]})

        result = engine.extract(["Einstein was born in Ulm."])

        assert len(result.entities) == 1
        assert len(result.entities[0]) == 2
        assert result.entities[0][0].text == "Einstein"
        assert result.relations == [[]]

    def test_standalone_relex(self):
        engine = self._make_engine(relation_labels=["born in"])
        assert engine.has_relex

        engine._client.complete.return_value = json.dumps({
            "entities": [
                {"text": "Einstein", "label": "person"},
                {"text": "Ulm", "label": "location"},
            ],
            "relations": [
                {"head": "Einstein", "tail": "Ulm", "relation": "born in"},
            ],
        })

        result = engine.extract(["Einstein was born in Ulm."])

        assert len(result.entities[0]) == 2
        assert len(result.relations[0]) == 1
        assert result.relations[0][0].head_text == "Einstein"
        assert result.relations[0][0].head_label == "person"

    def test_with_preextracted_entities(self):
        engine = self._make_engine(relation_labels=["born in"])
        engine._client.complete.return_value = json.dumps({"relations": [
            {"head": "Einstein", "tail": "Ulm", "relation": "born in"},
        ]})

        pre_entities = [[
            EntityMention(text="Einstein", label="person", start=0, end=8, score=0.95),
            EntityMention(text="Ulm", label="location", start=21, end=24, score=0.85),
        ]]

        result = engine.extract(["Einstein was born in Ulm."], entities=pre_entities)

        assert len(result.relations[0]) == 1
        assert result.relations[0][0].head_label == "person"
        assert result.relations[0][0].tail_label == "location"
        assert result.entities[0][0].text == "Einstein"

    def test_extract_from_query(self):
        engine = self._make_engine()
        engine._client.complete.return_value = json.dumps({"entities": [
            {"text": "Einstein", "label": "person"},
        ]})

        mentions = engine.extract_from_query("Where was Einstein born?")

        assert len(mentions) == 1
        assert mentions[0].text == "Einstein"
        assert mentions[0].start == 10
        assert mentions[0].end == 18

    def test_extract_from_query_filters_labels(self):
        engine = self._make_engine(labels=["person"])
        engine._client.complete.return_value = json.dumps({"entities": [
            {"text": "Einstein", "label": "person"},
            {"text": "Ulm", "label": "location"},
        ]})

        mentions = engine.extract_from_query("Einstein in Ulm")
        assert len(mentions) == 1
        assert mentions[0].text == "Einstein"

    def test_api_error_returns_empty(self):
        engine = self._make_engine()
        engine._client.complete.side_effect = Exception("API error")

        result = engine.extract(["some text"])
        assert result.entities[0] == []

    def test_structured_output_fallback(self):
        engine = self._make_engine()

        engine._client.complete.side_effect = [
            Exception("Unsupported"),
            json.dumps({"entities": [{"text": "Einstein", "label": "person"}]}),
            json.dumps({"entities": [{"text": "Newton", "label": "person"}]}),
        ]

        result = engine.extract(["Einstein text", "Newton text"])

        assert len(result.entities[0]) == 1
        assert result.entities[0][0].text == "Einstein"
        assert len(result.entities[1]) == 1
        assert result.entities[1][0].text == "Newton"
        assert engine._structured_failed is True

    def test_structured_output_disabled(self):
        engine = self._make_engine(structured_output=False)
        engine._client.complete.return_value = json.dumps({"entities": [
            {"text": "Einstein", "label": "person"},
        ]})

        engine.extract(["Einstein"])

        call_kwargs = engine._client.complete.call_args[1]
        assert call_kwargs["response_format"] == {"type": "json_object"}

    def test_extract_from_text_ner_only(self):
        engine = self._make_engine()
        engine._client.complete.return_value = json.dumps({"entities": [
            {"text": "Einstein", "label": "person"},
        ]})

        mentions, relations = engine.extract_from_text("Einstein was a physicist.")
        assert len(mentions) == 1
        assert relations == []

    def test_extract_from_text_standalone(self):
        engine = self._make_engine(relation_labels=["born in"])
        engine._client.complete.return_value = json.dumps({
            "entities": [{"text": "Einstein", "label": "person"}],
            "relations": [{"head": "Einstein", "tail": "Ulm", "relation": "born in"}],
        })

        mentions, relations = engine.extract_from_text("Einstein was born in Ulm.")
        assert len(mentions) == 1
        assert len(relations) == 1

    def test_extract_from_text_with_entities(self):
        engine = self._make_engine(relation_labels=["born in"])
        engine._client.complete.return_value = json.dumps({"relations": [
            {"head": "Einstein", "tail": "Ulm", "relation": "born in"},
        ]})

        pre_entities = [
            EntityMention(text="Einstein", label="person", start=0, end=8),
            EntityMention(text="Ulm", label="location", start=21, end=24),
        ]

        mentions, relations = engine.extract_from_text("Einstein was born in Ulm.", entities=pre_entities)
        assert len(relations) == 1
        assert len(mentions) == 2
