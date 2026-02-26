"""Tests for OpenAI LLM client."""

import json

import pytest
from unittest.mock import MagicMock, patch

from grapsit.llm.base import (
    BaseLLMClient,
    GRAPH_TOOLS,
    PROPERTY_FILTER_SCHEMA,
    tool_call_to_cypher,
    register_tool_translator,
    build_graph_schema_prompt,
)
from grapsit.llm.openai_client import OpenAIClient


class TestGraphTools:
    def test_graph_tools_defined(self):
        """Built-in graph tools are defined."""
        assert len(GRAPH_TOOLS) >= 7
        names = [t["function"]["name"] for t in GRAPH_TOOLS]
        assert "search_entity" in names
        assert "list_entities" in names
        assert "get_entity_relations" in names
        assert "get_neighbors" in names
        assert "get_subgraph" in names
        assert "get_chunks_for_entity" in names
        assert "find_shortest_path" in names

    def test_graph_tools_format(self):
        """Each tool has the OpenAI function-calling schema."""
        for tool in GRAPH_TOOLS:
            assert tool["type"] == "function"
            func = tool["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func
            assert func["parameters"]["type"] == "object"
            assert "required" in func["parameters"]

    def test_property_filter_schema(self):
        """PROPERTY_FILTER_SCHEMA defines array of filter objects."""
        assert PROPERTY_FILTER_SCHEMA["type"] == "array"
        item = PROPERTY_FILTER_SCHEMA["items"]
        assert "property" in item["properties"]
        assert "operator" in item["properties"]
        assert "value" in item["properties"]
        operators = item["properties"]["operator"]["enum"]
        assert "eq" in operators
        assert "gte" in operators
        assert "contains" in operators
        assert "starts_with" in operators

    def test_search_entity_has_entity_type_filter(self):
        """search_entity supports entity_type filter."""
        tool = next(t for t in GRAPH_TOOLS if t["function"]["name"] == "search_entity")
        props = tool["function"]["parameters"]["properties"]
        assert "entity_type" in props

    def test_list_entities_has_filters(self):
        """list_entities supports entity_type, property filters, and limit."""
        tool = next(t for t in GRAPH_TOOLS if t["function"]["name"] == "list_entities")
        props = tool["function"]["parameters"]["properties"]
        assert "entity_type" in props
        assert "filters" in props
        assert "limit" in props
        assert props["filters"]["type"] == "array"

    def test_get_entity_relations_has_filters(self):
        """get_entity_relations supports relation_type, target_entity_type, min_score, and property filters."""
        tool = next(t for t in GRAPH_TOOLS if t["function"]["name"] == "get_entity_relations")
        props = tool["function"]["parameters"]["properties"]
        assert "relation_type" in props
        assert "target_entity_type" in props
        assert "min_score" in props
        assert "filters" in props

    def test_get_neighbors_has_filters(self):
        """get_neighbors supports entity_type, relation_type, and property filters."""
        tool = next(t for t in GRAPH_TOOLS if t["function"]["name"] == "get_neighbors")
        props = tool["function"]["parameters"]["properties"]
        assert "entity_type" in props
        assert "relation_type" in props
        assert "filters" in props

    def test_find_shortest_path_has_relation_type(self):
        """find_shortest_path supports relation_type filter."""
        tool = next(t for t in GRAPH_TOOLS if t["function"]["name"] == "find_shortest_path")
        props = tool["function"]["parameters"]["properties"]
        assert "relation_type" in props


class TestBaseLLMClient:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            BaseLLMClient()

    def test_complete_with_tools_not_implemented(self):
        """Default complete_with_tools raises NotImplementedError."""

        class MinimalClient(BaseLLMClient):
            def complete(self, messages, **kwargs):
                return "ok"

        client = MinimalClient()
        with pytest.raises(NotImplementedError, match="MinimalClient"):
            client.complete_with_tools([{"role": "user", "content": "hi"}])


class TestOpenAIClient:
    def test_lazy_loading(self):
        """Client is not created until first use."""
        client = OpenAIClient(api_key="test-key", model="gpt-4o-mini")
        assert client._client is None

    @patch("grapsit.llm.openai_client.OpenAIClient._ensure_client")
    def test_complete(self, mock_ensure):
        """complete() calls the underlying OpenAI API."""
        client = OpenAIClient(api_key="test-key", model="gpt-4o-mini")

        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"entities": []}'
        mock_openai.chat.completions.create.return_value = mock_response
        client._client = mock_openai

        result = client.complete(
            [{"role": "user", "content": "test"}],
            response_format={"type": "json_object"},
        )

        assert result == '{"entities": []}'
        mock_openai.chat.completions.create.assert_called_once()
        call_kwargs = mock_openai.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs["response_format"] == {"type": "json_object"}

    @patch("grapsit.llm.openai_client.OpenAIClient._ensure_client")
    def test_temperature_override(self, mock_ensure):
        """Temperature can be overridden per call."""
        client = OpenAIClient(api_key="test-key", temperature=0.1)
        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"
        mock_openai.chat.completions.create.return_value = mock_response
        client._client = mock_openai

        client.complete(
            [{"role": "user", "content": "test"}],
            temperature=0.7,
        )

        call_kwargs = mock_openai.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.7

    def test_config_stored(self):
        """Config values are stored correctly."""
        client = OpenAIClient(
            api_key="key123",
            base_url="http://localhost:8000/v1",
            model="llama3",
            temperature=0.5,
            max_completion_tokens=2048,
            timeout=30.0,
            max_retries=3,
        )
        assert client.api_key == "key123"
        assert client.base_url == "http://localhost:8000/v1"
        assert client.model == "llama3"
        assert client.default_temperature == 0.5
        assert client.default_max_completion_tokens == 2048
        assert client.timeout == 30.0
        assert client.max_retries == 3


class TestCompleteWithTools:
    """Tests for complete_with_tools() on OpenAIClient."""

    def _make_client(self):
        client = OpenAIClient(api_key="test-key", model="gpt-4o-mini")
        mock_openai = MagicMock()
        client._client = mock_openai
        return client, mock_openai

    def _mock_tool_call(self, *, tc_id="call_abc", name="search_entity", arguments='{"label": "Einstein"}'):
        tc = MagicMock()
        tc.id = tc_id
        tc.function.name = name
        tc.function.arguments = arguments
        return tc

    def _mock_response(self, content=None, tool_calls=None):
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = content
        resp.choices[0].message.tool_calls = tool_calls
        return resp

    @patch("grapsit.llm.openai_client.OpenAIClient._ensure_client")
    def test_tool_call_returned(self, mock_ensure):
        """Tool calls are parsed from the response."""
        client, mock_openai = self._make_client()
        tc = self._mock_tool_call()
        mock_openai.chat.completions.create.return_value = self._mock_response(
            tool_calls=[tc]
        )

        result = client.complete_with_tools(
            [{"role": "user", "content": "Find Einstein"}],
        )

        assert result["content"] is None
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["id"] == "call_abc"
        assert result["tool_calls"][0]["name"] == "search_entity"
        assert result["tool_calls"][0]["arguments"] == {"label": "Einstein"}

    @patch("grapsit.llm.openai_client.OpenAIClient._ensure_client")
    def test_text_response_no_tools(self, mock_ensure):
        """When no tool calls, returns content with empty tool_calls list."""
        client, mock_openai = self._make_client()
        mock_openai.chat.completions.create.return_value = self._mock_response(
            content="Einstein was born in Ulm.", tool_calls=None
        )

        result = client.complete_with_tools(
            [{"role": "user", "content": "Where was Einstein born?"}],
        )

        assert result["content"] == "Einstein was born in Ulm."
        assert result["tool_calls"] == []

    @patch("grapsit.llm.openai_client.OpenAIClient._ensure_client")
    def test_multiple_tool_calls(self, mock_ensure):
        """Multiple tool calls in a single response are all returned."""
        client, mock_openai = self._make_client()
        tc1 = self._mock_tool_call(tc_id="call_1", name="search_entity", arguments='{"label": "Einstein"}')
        tc2 = self._mock_tool_call(tc_id="call_2", name="search_entity", arguments='{"label": "Ulm"}')
        mock_openai.chat.completions.create.return_value = self._mock_response(
            tool_calls=[tc1, tc2]
        )

        result = client.complete_with_tools(
            [{"role": "user", "content": "Find Einstein and Ulm"}],
        )

        assert len(result["tool_calls"]) == 2
        assert result["tool_calls"][0]["name"] == "search_entity"
        assert result["tool_calls"][1]["arguments"] == {"label": "Ulm"}

    @patch("grapsit.llm.openai_client.OpenAIClient._ensure_client")
    def test_default_tools_are_graph_tools(self, mock_ensure):
        """When tools=None, GRAPH_TOOLS are passed to the API."""
        client, mock_openai = self._make_client()
        mock_openai.chat.completions.create.return_value = self._mock_response(
            content="ok", tool_calls=None
        )

        client.complete_with_tools(
            [{"role": "user", "content": "hi"}],
        )

        call_kwargs = mock_openai.chat.completions.create.call_args[1]
        assert call_kwargs["tools"] is GRAPH_TOOLS

    @patch("grapsit.llm.openai_client.OpenAIClient._ensure_client")
    def test_custom_tools(self, mock_ensure):
        """Custom tools are passed through to the API."""
        client, mock_openai = self._make_client()
        mock_openai.chat.completions.create.return_value = self._mock_response(
            content="ok", tool_calls=None
        )

        custom_tools = [{
            "type": "function",
            "function": {
                "name": "my_tool",
                "description": "A custom tool",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }]

        client.complete_with_tools(
            [{"role": "user", "content": "hi"}],
            tools=custom_tools,
        )

        call_kwargs = mock_openai.chat.completions.create.call_args[1]
        assert call_kwargs["tools"] is custom_tools

    @patch("grapsit.llm.openai_client.OpenAIClient._ensure_client")
    def test_combined_builtin_and_custom_tools(self, mock_ensure):
        """Users can combine GRAPH_TOOLS with custom tools."""
        client, mock_openai = self._make_client()
        mock_openai.chat.completions.create.return_value = self._mock_response(
            content="ok", tool_calls=None
        )

        custom = [{
            "type": "function",
            "function": {
                "name": "custom_query",
                "description": "Run a custom query",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }]
        combined = GRAPH_TOOLS + custom

        client.complete_with_tools(
            [{"role": "user", "content": "hi"}],
            tools=combined,
        )

        call_kwargs = mock_openai.chat.completions.create.call_args[1]
        assert len(call_kwargs["tools"]) == len(GRAPH_TOOLS) + 1

    @patch("grapsit.llm.openai_client.OpenAIClient._ensure_client")
    def test_temperature_override(self, mock_ensure):
        """Temperature can be overridden in complete_with_tools."""
        client, mock_openai = self._make_client()
        mock_openai.chat.completions.create.return_value = self._mock_response(
            content="ok", tool_calls=None
        )

        client.complete_with_tools(
            [{"role": "user", "content": "hi"}],
            temperature=0.9,
        )

        call_kwargs = mock_openai.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.9

    @patch("grapsit.llm.openai_client.OpenAIClient._ensure_client")
    def test_malformed_arguments_returned_as_string(self, mock_ensure):
        """If arguments JSON is malformed, the raw string is returned."""
        client, mock_openai = self._make_client()
        tc = self._mock_tool_call(arguments="not valid json{{{")
        mock_openai.chat.completions.create.return_value = self._mock_response(
            tool_calls=[tc]
        )

        result = client.complete_with_tools(
            [{"role": "user", "content": "hi"}],
        )

        assert result["tool_calls"][0]["arguments"] == "not valid json{{{"

    @patch("grapsit.llm.openai_client.OpenAIClient._ensure_client")
    def test_tool_call_with_property_filters(self, mock_ensure):
        """Tool calls can include property filters (e.g. list_entities with filters)."""
        client, mock_openai = self._make_client()
        args = {
            "entity_type": "organization",
            "filters": [
                {"property": "location", "operator": "eq", "value": "Cambridge"},
                {"property": "founded_year", "operator": "gte", "value": 2020},
            ],
            "limit": 10,
        }
        tc = self._mock_tool_call(
            tc_id="call_filter",
            name="list_entities",
            arguments=json.dumps(args),
        )
        mock_openai.chat.completions.create.return_value = self._mock_response(
            tool_calls=[tc]
        )

        result = client.complete_with_tools(
            [{"role": "user", "content": "List companies in Cambridge founded after 2020"}],
        )

        parsed = result["tool_calls"][0]
        assert parsed["name"] == "list_entities"
        assert parsed["arguments"]["entity_type"] == "organization"
        assert len(parsed["arguments"]["filters"]) == 2
        assert parsed["arguments"]["filters"][0]["operator"] == "eq"
        assert parsed["arguments"]["filters"][1]["value"] == 2020

    @patch("grapsit.llm.openai_client.OpenAIClient._ensure_client")
    def test_tool_call_relations_with_filters(self, mock_ensure):
        """get_entity_relations can include relation_type and min_score filters."""
        client, mock_openai = self._make_client()
        args = {
            "entity_id": "entity-123",
            "relation_type": "COLLABORATED_WITH",
            "target_entity_type": "organization",
            "min_score": 0.8,
        }
        tc = self._mock_tool_call(
            tc_id="call_rel",
            name="get_entity_relations",
            arguments=json.dumps(args),
        )
        mock_openai.chat.completions.create.return_value = self._mock_response(
            tool_calls=[tc]
        )

        result = client.complete_with_tools(
            [{"role": "user", "content": "What organizations collaborated with Pfizer?"}],
        )

        parsed = result["tool_calls"][0]
        assert parsed["name"] == "get_entity_relations"
        assert parsed["arguments"]["relation_type"] == "COLLABORATED_WITH"
        assert parsed["arguments"]["min_score"] == 0.8


class TestToolCallToCypher:
    """Tests for tool_call_to_cypher() — translating tool arguments to Cypher."""

    def test_search_entity_basic(self):
        cypher, params = tool_call_to_cypher("search_entity", {"label": "Einstein"})
        assert "MATCH (e:Entity)" in cypher
        assert "toLower(e.label) = toLower($label)" in cypher
        assert params["label"] == "Einstein"

    def test_search_entity_with_type(self):
        cypher, params = tool_call_to_cypher(
            "search_entity", {"label": "MIT", "entity_type": "organization"}
        )
        assert "toLower(e.entity_type) = toLower($entity_type)" in cypher
        assert params["entity_type"] == "organization"

    def test_list_entities_no_filters(self):
        cypher, params = tool_call_to_cypher("list_entities", {})
        assert "MATCH (e:Entity)" in cypher
        assert "LIMIT $limit" in cypher
        assert params["limit"] == 50

    def test_list_entities_with_type(self):
        cypher, params = tool_call_to_cypher(
            "list_entities", {"entity_type": "person", "limit": 10}
        )
        assert "toLower(e.entity_type) = toLower($entity_type)" in cypher
        assert params["entity_type"] == "person"
        assert params["limit"] == 10

    def test_list_entities_with_property_filters(self):
        """The Cambridge/founded_year scenario."""
        cypher, params = tool_call_to_cypher("list_entities", {
            "entity_type": "organization",
            "filters": [
                {"property": "location", "operator": "eq", "value": "Cambridge"},
                {"property": "founded_year", "operator": "gte", "value": 2020},
            ],
            "limit": 20,
        })
        assert "e.location = $f_0_location" in cypher
        assert "e.founded_year >= $f_1_founded_year" in cypher
        assert params["f_0_location"] == "Cambridge"
        assert params["f_1_founded_year"] == 2020
        assert params["limit"] == 20

    def test_list_entities_contains_filter(self):
        cypher, params = tool_call_to_cypher("list_entities", {
            "filters": [
                {"property": "name", "operator": "contains", "value": "bio"},
            ],
        })
        assert "toLower(e.name) CONTAINS toLower($f_0_name)" in cypher
        assert params["f_0_name"] == "bio"

    def test_get_entity_relations_basic(self):
        cypher, params = tool_call_to_cypher(
            "get_entity_relations", {"entity_id": "id-123"}
        )
        assert "MATCH (e:Entity)-[r]-(other:Entity)" in cypher
        assert "e.id = $entity_id" in cypher
        assert params["entity_id"] == "id-123"

    def test_get_entity_relations_with_type(self):
        cypher, params = tool_call_to_cypher("get_entity_relations", {
            "entity_id": "id-123",
            "relation_type": "COLLABORATED_WITH",
        })
        assert "COLLABORATED_WITH" in cypher
        assert params["entity_id"] == "id-123"

    def test_get_entity_relations_with_all_filters(self):
        cypher, params = tool_call_to_cypher("get_entity_relations", {
            "entity_id": "id-123",
            "relation_type": "WORKS_AT",
            "target_entity_type": "organization",
            "min_score": 0.7,
            "filters": [
                {"property": "since", "operator": "gte", "value": 2020},
            ],
        })
        assert "WORKS_AT" in cypher
        assert "toLower(other.entity_type) = toLower($target_entity_type)" in cypher
        assert "r.score >= $min_score" in cypher
        assert "r.since >= $rf_0_since" in cypher
        assert params["min_score"] == 0.7
        assert params["rf_0_since"] == 2020

    def test_get_neighbors_basic(self):
        cypher, params = tool_call_to_cypher(
            "get_neighbors", {"entity_id": "id-1"}
        )
        assert "[*1..1]" in cypher
        assert "RETURN DISTINCT neighbor" in cypher
        assert params["entity_id"] == "id-1"

    def test_get_neighbors_with_hops_and_filters(self):
        cypher, params = tool_call_to_cypher("get_neighbors", {
            "entity_id": "id-1",
            "max_hops": 3,
            "entity_type": "person",
            "relation_type": "KNOWS",
            "filters": [
                {"property": "age", "operator": "gt", "value": 30},
            ],
        })
        assert "KNOWS" in cypher
        assert "*1..3" in cypher
        assert "toLower(neighbor.entity_type) = toLower($entity_type)" in cypher
        assert "neighbor.age > $nf_0_age" in cypher
        assert params["nf_0_age"] == 30

    def test_get_subgraph(self):
        cypher, params = tool_call_to_cypher(
            "get_subgraph", {"entity_ids": ["a", "b"], "max_hops": 2}
        )
        assert "[*1..2]" in cypher
        assert "e.id IN $entity_ids" in cypher
        assert params["entity_ids"] == ["a", "b"]

    def test_get_chunks_for_entity(self):
        cypher, params = tool_call_to_cypher(
            "get_chunks_for_entity", {"entity_id": "id-x"}
        )
        assert "MENTIONED_IN" in cypher
        assert params["entity_id"] == "id-x"

    def test_find_shortest_path_basic(self):
        cypher, params = tool_call_to_cypher("find_shortest_path", {
            "source_entity_id": "a",
            "target_entity_id": "b",
        })
        assert "shortestPath" in cypher
        assert "[*..5]" in cypher
        assert params["source_id"] == "a"
        assert params["target_id"] == "b"

    def test_find_shortest_path_with_relation_type(self):
        cypher, params = tool_call_to_cypher("find_shortest_path", {
            "source_entity_id": "a",
            "target_entity_id": "b",
            "max_depth": 3,
            "relation_type": "WORKS_AT",
        })
        assert "WORKS_AT" in cypher
        assert "*..3" in cypher

    def test_unknown_tool_raises(self):
        with pytest.raises(KeyError, match="no_such_tool"):
            tool_call_to_cypher("no_such_tool", {})

    def test_register_custom_translator(self):
        """Users can register custom tool → Cypher translators."""
        def my_translator(args):
            return (
                f"MATCH (n) WHERE n.custom = $val RETURN n",
                {"val": args["val"]},
            )

        register_tool_translator("my_custom_tool", my_translator)
        cypher, params = tool_call_to_cypher("my_custom_tool", {"val": 42})
        assert "n.custom = $val" in cypher
        assert params["val"] == 42


class TestBuildGraphSchemaPrompt:
    def test_basic_schema(self):
        prompt = build_graph_schema_prompt(
            entity_types=["person", "organization"],
            relation_types=["WORKS_AT", "BORN_IN"],
        )
        assert "person" in prompt
        assert "organization" in prompt
        assert "WORKS_AT" in prompt
        assert "BORN_IN" in prompt

    def test_schema_with_properties(self):
        prompt = build_graph_schema_prompt(
            entity_types=["organization"],
            relation_types=["COLLABORATED_WITH"],
            property_keys={"organization": ["founded_year", "location", "revenue"]},
        )
        assert "founded_year" in prompt
        assert "location" in prompt
        assert "revenue" in prompt

    def test_schema_contains_tool_instruction(self):
        prompt = build_graph_schema_prompt(
            entity_types=["person"],
            relation_types=["KNOWS"],
        )
        assert "Cypher" in prompt
