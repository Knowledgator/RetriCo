"""Tests for tool-calling query parser (method='tool')."""

import json
import pytest
from unittest.mock import MagicMock, patch

from grapsit.query.parser import QueryParserProcessor


class TestQueryParserTool:
    def test_basic_tool_parsing(self):
        proc = QueryParserProcessor({
            "method": "tool",
            "api_key": "test",
            "model": "test-model",
            "labels": ["person", "location"],
            "relation_labels": ["born_in", "works_at"],
        })
        proc._llm_client = MagicMock()
        proc._llm_client.complete_with_tools.return_value = {
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "search_triples",
                    "arguments": {
                        "head": "Einstein",
                        "relation": "born_in",
                        "tail": None,
                    },
                },
            ],
        }

        result = proc(query="Where was Einstein born?")

        assert result["query"] == "Where was Einstein born?"
        assert "triple_queries" in result
        assert len(result["triple_queries"]) == 1
        assert result["triple_queries"][0]["head"] == "Einstein"
        assert result["triple_queries"][0]["relation"] == "born_in"
        assert result["triple_queries"][0]["tail"] is None
        # Backward compat: entities extracted from head/tail
        assert len(result["entities"]) == 1
        assert result["entities"][0].text == "Einstein"

    def test_multiple_tool_calls(self):
        proc = QueryParserProcessor({
            "method": "tool",
            "api_key": "test",
            "labels": ["person", "organization"],
        })
        proc._llm_client = MagicMock()
        proc._llm_client.complete_with_tools.return_value = {
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "search_triples",
                    "arguments": {"head": "Einstein", "relation": None, "tail": "MIT"},
                },
                {
                    "id": "call_2",
                    "name": "search_triples",
                    "arguments": {"head": None, "relation": "works_at", "tail": "MIT"},
                },
            ],
        }

        result = proc(query="What is Einstein's relationship to MIT and who works at MIT?")

        assert len(result["triple_queries"]) == 2
        # Entities should be deduplicated
        entity_texts = {e.text for e in result["entities"]}
        assert "Einstein" in entity_texts
        assert "MIT" in entity_texts

    def test_no_tool_calls(self):
        proc = QueryParserProcessor({
            "method": "tool",
            "api_key": "test",
            "labels": ["person"],
        })
        proc._llm_client = MagicMock()
        proc._llm_client.complete_with_tools.return_value = {
            "content": "I don't understand the query.",
            "tool_calls": [],
        }

        result = proc(query="What is 2+2?")

        assert result["triple_queries"] == []
        assert result["entities"] == []

    def test_api_failure_returns_empty(self):
        proc = QueryParserProcessor({
            "method": "tool",
            "api_key": "test",
            "labels": ["person"],
        })
        proc._llm_client = MagicMock()
        proc._llm_client.complete_with_tools.side_effect = Exception("API error")

        result = proc(query="test query")

        assert result["triple_queries"] == []
        assert result["entities"] == []

    def test_non_search_triples_calls_ignored(self):
        proc = QueryParserProcessor({
            "method": "tool",
            "api_key": "test",
            "labels": ["person"],
        })
        proc._llm_client = MagicMock()
        proc._llm_client.complete_with_tools.return_value = {
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "some_other_tool",
                    "arguments": {"foo": "bar"},
                },
                {
                    "id": "call_2",
                    "name": "search_triples",
                    "arguments": {"head": "Einstein", "relation": None, "tail": None},
                },
            ],
        }

        result = proc(query="Tell me about Einstein")

        assert len(result["triple_queries"]) == 1
        assert result["triple_queries"][0]["head"] == "Einstein"

    def test_entity_offset_computation(self):
        proc = QueryParserProcessor({
            "method": "tool",
            "api_key": "test",
            "labels": ["person"],
        })
        proc._llm_client = MagicMock()
        proc._llm_client.complete_with_tools.return_value = {
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "search_triples",
                    "arguments": {"head": "Einstein", "relation": "born_in", "tail": None},
                },
            ],
        }

        result = proc(query="Was Einstein born in Germany?")

        mention = result["entities"][0]
        assert mention.text == "Einstein"
        assert mention.start == 4
        assert mention.end == 12

    def test_tool_calls_with_triple_query_tools(self):
        proc = QueryParserProcessor({
            "method": "tool",
            "api_key": "test",
            "labels": ["person", "location"],
            "relation_labels": ["born_in"],
        })
        proc._llm_client = MagicMock()
        proc._llm_client.complete_with_tools.return_value = {
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "search_triples",
                    "arguments": {"head": "Einstein", "relation": "born_in", "tail": None},
                },
            ],
        }

        result = proc(query="Where was Einstein born?")

        # Verify the tools passed to complete_with_tools
        call_args = proc._llm_client.complete_with_tools.call_args
        tools = call_args[1].get("tools") or call_args[0][1]
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "search_triples"

    def test_system_prompt_includes_schema(self):
        proc = QueryParserProcessor({
            "method": "tool",
            "api_key": "test",
            "labels": ["person", "location"],
            "relation_labels": ["born_in", "works_at"],
        })
        proc._llm_client = MagicMock()
        proc._llm_client.complete_with_tools.return_value = {
            "content": None,
            "tool_calls": [],
        }

        proc(query="test")

        call_args = proc._llm_client.complete_with_tools.call_args
        messages = call_args[0][0]
        system_msg = messages[0]["content"]
        assert "person" in system_msg
        assert "location" in system_msg
        assert "born_in" in system_msg
        assert "works_at" in system_msg

    def test_tail_only_query(self):
        proc = QueryParserProcessor({
            "method": "tool",
            "api_key": "test",
        })
        proc._llm_client = MagicMock()
        proc._llm_client.complete_with_tools.return_value = {
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "search_triples",
                    "arguments": {"head": None, "relation": "works_at", "tail": "MIT"},
                },
            ],
        }

        result = proc(query="Who works at MIT?")

        assert len(result["triple_queries"]) == 1
        assert result["triple_queries"][0]["head"] is None
        assert result["triple_queries"][0]["tail"] == "MIT"
        assert len(result["entities"]) == 1
        assert result["entities"][0].text == "MIT"
