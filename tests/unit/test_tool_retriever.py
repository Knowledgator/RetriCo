"""Tests for tool retriever processor."""

import json
import pytest
from unittest.mock import MagicMock, patch

from retrico.query.tool_retriever import ToolRetrieverProcessor


class TestToolRetrieverProcessor:
    def _make_proc(self, **config_overrides):
        config = {
            "neo4j_uri": "bolt://localhost:7687",
            "api_key": "test-key",
            "model": "test-model",
            "entity_types": ["person", "location"],
            "relation_types": ["BORN_IN", "WORKS_AT"],
            "max_tool_rounds": 3,
        }
        config.update(config_overrides)
        proc = ToolRetrieverProcessor(config)
        proc._store = MagicMock()
        proc._llm = MagicMock()
        return proc

    def test_single_round_tool_call(self):
        proc = self._make_proc()

        # LLM makes one tool call, then responds with no tool calls
        proc._llm.complete_with_tools.side_effect = [
            {
                "content": None,
                "tool_calls": [{
                    "id": "tc1",
                    "name": "search_entity",
                    "arguments": {"label": "Einstein"},
                }],
            },
            {
                "content": "Einstein was born in Ulm.",
                "tool_calls": [],
            },
        ]
        proc._store.run_cypher.return_value = [
            {"e": {"id": "e1", "label": "Einstein", "entity_type": "person"}},
        ]

        result = proc(query="Where was Einstein born?")

        assert "subgraph" in result
        sg = result["subgraph"]
        assert len(sg.entities) == 1
        assert sg.entities[0].label == "Einstein"

    def test_no_tool_calls(self):
        proc = self._make_proc()

        # LLM responds immediately without tool calls
        proc._llm.complete_with_tools.return_value = {
            "content": "I don't know.",
            "tool_calls": [],
        }

        result = proc(query="What is the meaning of life?")
        sg = result["subgraph"]
        assert len(sg.entities) == 0

    def test_max_rounds_respected(self):
        proc = self._make_proc(max_tool_rounds=2)

        # LLM keeps calling tools every round
        proc._llm.complete_with_tools.return_value = {
            "content": None,
            "tool_calls": [{
                "id": "tc1",
                "name": "search_entity",
                "arguments": {"label": "Einstein"},
            }],
        }
        proc._store.run_cypher.return_value = [
            {"e": {"id": "e1", "label": "Einstein", "entity_type": "person"}},
        ]

        result = proc(query="query")

        # Should have been called max_tool_rounds times
        assert proc._llm.complete_with_tools.call_count == 2

    def test_tool_call_error_handling(self):
        proc = self._make_proc()

        proc._llm.complete_with_tools.side_effect = [
            {
                "content": None,
                "tool_calls": [{
                    "id": "tc1",
                    "name": "search_entity",
                    "arguments": {"label": "Bad"},
                }],
            },
            {
                "content": "Done",
                "tool_calls": [],
            },
        ]
        proc._store.run_cypher.side_effect = Exception("DB error")

        # Should not raise — error is sent back to LLM as tool result
        result = proc(query="query")
        assert "subgraph" in result

    def test_multiple_tool_calls_per_round(self):
        proc = self._make_proc()

        proc._llm.complete_with_tools.side_effect = [
            {
                "content": None,
                "tool_calls": [
                    {"id": "tc1", "name": "search_entity", "arguments": {"label": "Einstein"}},
                    {"id": "tc2", "name": "search_entity", "arguments": {"label": "Ulm"}},
                ],
            },
            {"content": "Done", "tool_calls": []},
        ]
        proc._store.run_cypher.side_effect = [
            [{"e": {"id": "e1", "label": "Einstein", "entity_type": "person"}}],
            [{"e": {"id": "e2", "label": "Ulm", "entity_type": "location"}}],
        ]

        result = proc(query="Einstein and Ulm")
        sg = result["subgraph"]
        assert len(sg.entities) == 2

    def test_config_defaults(self):
        proc = ToolRetrieverProcessor({})
        assert proc.max_tool_rounds == 3
        assert proc.entity_types == []
        assert proc.relation_types == []

    def test_collect_relations_from_results(self):
        proc = self._make_proc()

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
            },
        ]

        result = proc(query="Where was Einstein born?")
        sg = result["subgraph"]
        assert len(sg.relations) == 1
        assert sg.relations[0].relation_type == "BORN_IN"
