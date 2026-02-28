"""Tool retriever — LLM agent with graph query tools."""

from typing import Any, Dict, List
import json
import logging

from ..core.registry import query_registry
from ..llm.prompts import TOOL_RETRIEVER_SYSTEM_PROMPT
from ..models.entity import Entity
from ..models.relation import Relation
from ..models.graph import Subgraph
from .base_retriever import BaseRetriever

logger = logging.getLogger(__name__)

# Properties to exclude when serializing graph objects for the LLM
_EXCLUDE_PROPS = {"embedding"}


def _obj_to_dict(obj: Any) -> Any:
    """Convert a FalkorDB Node/Edge or Neo4j record value to a plain dict.

    - FalkorDB ``Node``: ``{**node.properties, "_labels": node.labels}``
    - FalkorDB ``Edge``: ``{**edge.properties, "_relation": edge.relation}``
    - ``dict`` / primitive: returned as-is.

    Large properties (embeddings) are stripped out.
    """
    cls = type(obj).__name__

    if cls == "Node" and hasattr(obj, "properties"):
        d = {k: v for k, v in obj.properties.items() if k not in _EXCLUDE_PROPS}
        d["_labels"] = getattr(obj, "labels", [])
        return d
    if cls == "Edge" and hasattr(obj, "properties"):
        d = {k: v for k, v in obj.properties.items() if k not in _EXCLUDE_PROPS}
        d["_relation"] = getattr(obj, "relation", "")
        return d
    if isinstance(obj, dict):
        return {k: v for k, v in obj.items() if k not in _EXCLUDE_PROPS}
    return obj


def _normalize_results(raw_results: list) -> List[Dict[str, Any]]:
    """Normalize raw Cypher results to a list of dicts.

    Neo4j returns ``[{"e": {...}, "r": {...}}, ...]``.
    FalkorDB returns ``[[Node, Edge, Node], ...]``.
    This function produces a uniform list of flat dicts.
    """
    normalized = []
    for record in raw_results:
        if isinstance(record, dict):
            # Neo4j style — convert any nested Node/Edge objects
            row: Dict[str, Any] = {}
            for key, val in record.items():
                row[key] = _obj_to_dict(val)
            normalized.append(row)
        elif isinstance(record, (list, tuple)):
            # FalkorDB style — list of Node/Edge objects
            row = {}
            for i, val in enumerate(record):
                converted = _obj_to_dict(val)
                if isinstance(converted, dict):
                    # Merge into row, prefixing duplicate keys
                    for k, v in converted.items():
                        dest_key = k if k not in row else f"{k}_{i}"
                        row[dest_key] = v
                else:
                    row[f"col_{i}"] = converted
            normalized.append(row)
    return normalized


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
        system_prompt: str — override default tool retriever system prompt.
            Available placeholder: {schema_prompt}.
    """

    default_inputs = {"query": "$input.query"}
    default_output = "tool_retriever_result"

    # Maximum number of result records returned per tool call
    MAX_RESULTS_PER_CALL = 50
    # Maximum characters of JSON serialised tool output sent to the LLM
    MAX_RESULT_CHARS = 8000

    def __init__(self, config_dict: Dict[str, Any], pipeline: Any = None):
        super().__init__(config_dict, pipeline)
        self.max_tool_rounds: int = config_dict.get("max_tool_rounds", 3)
        self.entity_types: List[str] = config_dict.get("entity_types", [])
        self.relation_types: List[str] = config_dict.get("relation_types", [])
        self._system_prompt_template = config_dict.get("system_prompt", TOOL_RETRIEVER_SYSTEM_PROMPT)
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

        system_content = self._system_prompt_template.format(
            schema_prompt=schema_prompt,
        )

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_content},
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
                    raw = self._store.run_cypher(cypher, params)
                    results = _normalize_results(raw)
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
        results: List[Dict[str, Any]],
        entities: Dict[str, Entity],
        relations: List[Relation],
    ):
        """Extract entities and relations from normalized Cypher results."""
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

            # Handle flat records (FalkorDB normalized): entity props merged at top level
            if "id" in record and "label" in record and "_labels" in record:
                eid = record["id"]
                if eid not in entities:
                    entities[eid] = Entity(
                        id=eid,
                        label=record.get("label", ""),
                        entity_type=record.get("entity_type", ""),
                    )

            # Handle relations from flat records (e.g. RETURN e, r, other)
            if "_relation" in record:
                rel_type = record["_relation"]
                # Find entity IDs — look for id and id_N patterns
                entity_ids = []
                for k, v in record.items():
                    if k == "id" or (k.startswith("id_") and isinstance(v, str)):
                        entity_ids.append(v)
                entity_labels = []
                for k, v in record.items():
                    if k == "label" or (k.startswith("label_") and isinstance(v, str)):
                        entity_labels.append(v)
                if len(entity_ids) >= 2:
                    relations.append(Relation(
                        head_text=entity_labels[0] if entity_labels else entity_ids[0],
                        tail_text=entity_labels[1] if len(entity_labels) > 1 else entity_ids[1],
                        relation_type=rel_type,
                        score=record.get("score", 0.0) or 0.0,
                    ))


@query_registry.register("tool_retriever")
def create_tool_retriever(config_dict: dict, pipeline=None):
    return ToolRetrieverProcessor(config_dict, pipeline)
