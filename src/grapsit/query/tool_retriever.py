"""Tool retriever — LLM agent with graph query tools."""

from typing import Any, Dict, List
import json
import logging

from ..core.registry import processor_registry
from ..models.entity import Entity
from ..models.relation import Relation
from ..models.graph import Subgraph
from .base_retriever import BaseRetriever

logger = logging.getLogger(__name__)


class ToolRetrieverProcessor(BaseRetriever):
    """Retrieve subgraph via an LLM agent that calls graph query tools.

    The LLM receives the user query plus graph schema context, and can call
    tools (search_entity, get_entity_relations, get_neighbors, etc.) which
    are translated to Cypher and executed against the graph store. Multiple
    rounds of tool calls are supported.

    Config keys:
        api_key: str — OpenAI API key
        base_url: str — optional API base URL
        model: str — LLM model name (default: "gpt-4o-mini")
        temperature: float — (default: 0.1)
        max_completion_tokens: int — (default: 4096)
        timeout: float — (default: 60.0)
        entity_types: List[str] — entity types in the graph (for schema prompt)
        relation_types: List[str] — relation types in the graph (for schema prompt)
        max_tool_rounds: int — max agentic loop iterations (default: 3)
        store_type, neo4j_uri, etc. — passed to create_store
    """

    # Maximum number of result records returned per tool call
    MAX_RESULTS_PER_CALL = 50
    # Maximum characters of JSON serialised tool output sent to the LLM
    MAX_RESULT_CHARS = 8000

    def __init__(self, config_dict: Dict[str, Any], pipeline: Any = None):
        super().__init__(config_dict, pipeline)
        self.max_tool_rounds: int = config_dict.get("max_tool_rounds", 3)
        self.entity_types: List[str] = config_dict.get("entity_types", [])
        self.relation_types: List[str] = config_dict.get("relation_types", [])
        self._llm = None

    def _ensure_llm(self):
        """Lazily create the LLM client."""
        if self._llm is None:
            from ..llm.openai_client import OpenAIClient
            self._llm = OpenAIClient(
                api_key=self.config_dict.get("api_key"),
                base_url=self.config_dict.get("base_url"),
                model=self.config_dict.get("model", "gpt-4o-mini"),
                temperature=self.config_dict.get("temperature", 0.1),
                max_completion_tokens=self.config_dict.get("max_completion_tokens", 4096),
                timeout=self.config_dict.get("timeout", 60.0),
            )

    def __call__(self, *, query: str, **kwargs) -> Dict[str, Any]:
        """Run an agentic tool-calling loop to retrieve a subgraph.

        Returns:
            {"subgraph": Subgraph}
        """
        self._ensure_store()
        self._ensure_llm()

        from ..llm.tools import (
            GRAPH_TOOLS,
            tool_call_to_cypher,
            build_graph_schema_prompt,
        )

        # Build schema context
        schema_prompt = build_graph_schema_prompt(
            self.entity_types, self.relation_types
        )

        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are a knowledge graph query agent. Use the available tools "
                    "to find information relevant to the user's question.\n\n"
                    f"{schema_prompt}\n\n"
                    "IMPORTANT: Entity nodes only have these properties: id, label, "
                    "entity_type. Do NOT use property filters for attributes like "
                    "'location', 'founded_year', etc. — they do not exist. Instead:\n"
                    "- Use search_entity to find entities by label text.\n"
                    "- Use get_entity_relations and get_neighbors to discover "
                    "connections between entities.\n"
                    "- Use find_shortest_path to find paths between two entities.\n\n"
                    "Call tools to search the graph. When you have enough information, "
                    "respond with a final text summary."
                ),
            },
            {"role": "user", "content": query},
        ]

        collected_entities: Dict[str, Entity] = {}
        collected_relations: List[Relation] = []

        for _round in range(self.max_tool_rounds):
            response = self._llm.complete_with_tools(messages, tools=GRAPH_TOOLS)

            tool_calls = response.get("tool_calls", [])
            if not tool_calls:
                # LLM is done — no more tool calls
                break

            # Add assistant message with tool calls
            messages.append({
                "role": "assistant",
                "content": response.get("content"),
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"]),
                        },
                    }
                    for tc in tool_calls
                ],
            })

            # Execute each tool call
            for tc in tool_calls:
                results = []
                try:
                    cypher, params = tool_call_to_cypher(tc["name"], tc["arguments"])
                    results = self._store.run_cypher(cypher, params)
                except Exception as e:
                    results = []
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps({"error": str(e)}),
                    })
                    continue

                # Collect all results internally (for the subgraph)
                self._collect_from_results(results, collected_entities, collected_relations)

                # Truncate results sent to the LLM to stay within context limits
                truncated = results[: self.MAX_RESULTS_PER_CALL]
                result_str = json.dumps(truncated, default=str)
                if len(result_str) > self.MAX_RESULT_CHARS:
                    result_str = result_str[: self.MAX_RESULT_CHARS] + '..."]}'
                    result_str = (
                        f'{result_str}\n[Truncated: {len(results)} total results, '
                        f'showing first {self.MAX_RESULTS_PER_CALL}]'
                    )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_str,
                })

        return {
            "subgraph": Subgraph(
                entities=list(collected_entities.values()),
                relations=collected_relations,
            )
        }

    def _collect_from_results(
        self,
        results: list,
        entities: Dict[str, Entity],
        relations: List[Relation],
    ):
        """Extract entities and relations from Cypher query results."""
        for record in results:
            if not isinstance(record, dict):
                continue
            # Try to extract entities from various result shapes
            for key, val in record.items():
                if isinstance(val, dict) and "id" in val and "label" in val:
                    eid = val["id"]
                    if eid not in entities:
                        entities[eid] = Entity(
                            id=eid,
                            label=val.get("label", ""),
                            entity_type=val.get("entity_type", ""),
                        )
                # Extract relation info
                if key == "relation_type" and "target_id" in record:
                    relations.append(Relation(
                        head_text=record.get("entity_id", ""),
                        tail_text=record.get("target_id", ""),
                        relation_type=record.get("relation_type", ""),
                        score=record.get("score", 0.0) or 0.0,
                    ))


@processor_registry.register("tool_retriever")
def create_tool_retriever(config_dict: dict, pipeline=None):
    return ToolRetrieverProcessor(config_dict, pipeline)
