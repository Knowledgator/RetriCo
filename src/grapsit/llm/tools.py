"""Built-in graph query tools and Cypher translation.

The LLM receives the user query plus graph metadata (entity types, property
keys, relation types) as context.  It produces structured tool calls whose
arguments are translated into parameterised Cypher queries by
``tool_call_to_cypher()``.

Users can extend the built-in tool set by adding their own tool definitions
and registering a corresponding Cypher translator via ``register_tool_translator``.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Reusable filter schema for arbitrary entity/relation properties
# ---------------------------------------------------------------------------

PROPERTY_FILTER_SCHEMA: Dict[str, Any] = {
    "type": "array",
    "description": (
        "Filters on arbitrary node properties. Each filter has a property name, "
        "an operator, and a value. Example: "
        '[{"property": "founded_year", "operator": ">=", "value": 2020}, '
        '{"property": "location", "operator": "eq", "value": "Cambridge"}]'
    ),
    "items": {
        "type": "object",
        "properties": {
            "property": {
                "type": "string",
                "description": "The property key to filter on (e.g. 'founded_year', 'location', 'revenue').",
            },
            "operator": {
                "type": "string",
                "enum": ["eq", "neq", "gt", "gte", "lt", "lte", "contains", "starts_with"],
                "description": (
                    "Comparison operator: eq (equals), neq (not equals), "
                    "gt/gte/lt/lte (numeric comparisons), "
                    "contains (substring match), starts_with (prefix match)."
                ),
            },
            "value": {
                "description": "The value to compare against. Type should match the property (string, number, boolean).",
            },
        },
        "required": ["property", "operator", "value"],
    },
}

# ---------------------------------------------------------------------------
# Built-in graph query tools (OpenAI function-calling format)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Triple query tool for tool-calling parser
# ---------------------------------------------------------------------------

TRIPLE_QUERY_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_triples",
            "description": (
                "Search for knowledge graph triples. Specify head entity, "
                "relation, and/or tail entity. Use null for unknown parts "
                "you want to find."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "head": {
                        "type": "string",
                        "description": "Source entity label (or null if unknown).",
                    },
                    "relation": {
                        "type": "string",
                        "description": "Relation type (or null if unknown).",
                    },
                    "tail": {
                        "type": "string",
                        "description": "Target entity label (or null if unknown).",
                    },
                },
                "required": [],
            },
        },
    },
]


GRAPH_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_entity",
            "description": (
                "Find an entity node by its label (case-insensitive). "
                "Optionally filter by entity_type. "
                "Returns entity id, label, entity_type, and all properties."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "The entity label to search for (e.g. 'Einstein', 'MIT').",
                    },
                    "entity_type": {
                        "type": "string",
                        "description": "Filter by entity type (e.g. 'person', 'organization', 'location').",
                    },
                },
                "required": ["label"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_entities",
            "description": (
                "List entities in the knowledge graph with optional filters. "
                "Supports filtering by entity_type and arbitrary property conditions. "
                "Example: list all companies in Cambridge founded after 2020."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_type": {
                        "type": "string",
                        "description": "Filter by entity type (e.g. 'person', 'organization').",
                    },
                    "filters": PROPERTY_FILTER_SCHEMA,
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of entities to return (default 50).",
                        "default": 50,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_entity_relations",
            "description": (
                "Get relationships involving an entity. "
                "Optionally filter by relation type, the entity type of the related entity, "
                "minimum confidence score, and arbitrary properties on the relation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "The entity ID to look up relations for.",
                    },
                    "relation_type": {
                        "type": "string",
                        "description": "Filter by relation type (e.g. 'BORN_IN', 'WORKS_AT', 'COLLABORATED_WITH').",
                    },
                    "target_entity_type": {
                        "type": "string",
                        "description": "Filter by the entity type of the related entity (e.g. 'location').",
                    },
                    "min_score": {
                        "type": "number",
                        "description": "Minimum confidence score threshold (0.0-1.0).",
                    },
                    "filters": PROPERTY_FILTER_SCHEMA,
                },
                "required": ["entity_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_neighbors",
            "description": (
                "Get neighboring entities within a given number of hops. "
                "Optionally filter neighbors by entity type, relation type, "
                "and arbitrary properties on the neighbor entities."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "The starting entity ID.",
                    },
                    "max_hops": {
                        "type": "integer",
                        "description": "Maximum number of hops to traverse (default 1).",
                        "default": 1,
                    },
                    "entity_type": {
                        "type": "string",
                        "description": "Only return neighbors of this entity type (e.g. 'person').",
                    },
                    "relation_type": {
                        "type": "string",
                        "description": "Only traverse edges of this relation type (e.g. 'WORKS_AT').",
                    },
                    "filters": PROPERTY_FILTER_SCHEMA,
                },
                "required": ["entity_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_subgraph",
            "description": "Retrieve a subgraph around a set of entities. Returns entities and relationships within max_hops of the given entity IDs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of entity IDs to center the subgraph on.",
                    },
                    "max_hops": {
                        "type": "integer",
                        "description": "Maximum number of hops to traverse (default 1).",
                        "default": 1,
                    },
                },
                "required": ["entity_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_chunks_for_entity",
            "description": "Get source text chunks where an entity is mentioned. Returns chunk text, document info, and mention details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "The entity ID to find source chunks for.",
                    },
                },
                "required": ["entity_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_shortest_path",
            "description": (
                "Find the shortest path between two entities in the knowledge graph. "
                "Returns the sequence of entities and relations along the path. "
                "Optionally restrict traversal to a specific relation type."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_entity_id": {
                        "type": "string",
                        "description": "The starting entity ID.",
                    },
                    "target_entity_id": {
                        "type": "string",
                        "description": "The destination entity ID.",
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum path length to search (default 5).",
                        "default": 5,
                    },
                    "relation_type": {
                        "type": "string",
                        "description": "Only traverse edges of this relation type when searching for paths.",
                    },
                },
                "required": ["source_entity_id", "target_entity_id"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Operator mapping for property filters  →  Cypher expressions
# ---------------------------------------------------------------------------

_OPERATOR_MAP = {
    "eq": "=",
    "neq": "<>",
    "gt": ">",
    "gte": ">=",
    "lt": "<",
    "lte": "<=",
    "contains": "CONTAINS",
    "starts_with": "STARTS WITH",
}


def _build_filter_clauses(
    filters: List[Dict[str, Any]],
    node_var: str,
    param_prefix: str,
    params: Dict[str, Any],
) -> List[str]:
    """Turn a list of property filter dicts into Cypher WHERE fragments.

    Each filter ``{"property": "x", "operator": "gte", "value": 10}``
    becomes ``n.x >= $f_0_x`` with ``params["f_0_x"] = 10``.
    """
    clauses: List[str] = []
    for i, f in enumerate(filters):
        prop = f["property"]
        op = _OPERATOR_MAP.get(f["operator"])
        if op is None:
            continue
        pkey = f"{param_prefix}_{i}_{prop}"
        if op in ("CONTAINS", "STARTS WITH"):
            clauses.append(f"toLower({node_var}.{prop}) {op} toLower(${pkey})")
        else:
            clauses.append(f"{node_var}.{prop} {op} ${pkey}")
        params[pkey] = f["value"]
    return clauses


# ---------------------------------------------------------------------------
# Built-in tool  →  Cypher translators
# ---------------------------------------------------------------------------

def _translate_search_entity(args: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    params: Dict[str, Any] = {"label": args["label"]}
    where = ["toLower(e.label) = toLower($label)"]
    if "entity_type" in args:
        where.append("toLower(e.entity_type) = toLower($entity_type)")
        params["entity_type"] = args["entity_type"]
    return (
        f"MATCH (e:Entity) WHERE {' AND '.join(where)} RETURN e",
        params,
    )


def _translate_list_entities(args: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    params: Dict[str, Any] = {}
    where: List[str] = []
    if "entity_type" in args:
        where.append("toLower(e.entity_type) = toLower($entity_type)")
        params["entity_type"] = args["entity_type"]
    if "filters" in args:
        where.extend(_build_filter_clauses(args["filters"], "e", "f", params))
    limit = args.get("limit", 50)
    params["limit"] = limit
    where_clause = f" WHERE {' AND '.join(where)}" if where else ""
    return (
        f"MATCH (e:Entity){where_clause} RETURN e LIMIT $limit",
        params,
    )


def _translate_get_entity_relations(args: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    params: Dict[str, Any] = {"entity_id": args["entity_id"]}
    where = ["e.id = $entity_id"]
    post_where: List[str] = []

    rel_pattern = "[r]-"
    if "relation_type" in args:
        rel_type = args["relation_type"].upper().replace(" ", "_")
        rel_pattern = f"[r:`{rel_type}`]-"

    if "target_entity_type" in args:
        post_where.append("toLower(other.entity_type) = toLower($target_entity_type)")
        params["target_entity_type"] = args["target_entity_type"]
    if "min_score" in args:
        post_where.append("r.score >= $min_score")
        params["min_score"] = args["min_score"]
    if "filters" in args:
        post_where.extend(_build_filter_clauses(args["filters"], "r", "rf", params))

    post_clause = f" AND {' AND '.join(post_where)}" if post_where else ""
    return (
        f"MATCH (e:Entity)-{rel_pattern}(other:Entity) "
        f"WHERE {' AND '.join(where)}{post_clause} "
        f"RETURN e, r, other",
        params,
    )


def _translate_get_neighbors(args: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    max_hops = args.get("max_hops", 1)
    params: Dict[str, Any] = {"entity_id": args["entity_id"]}

    rel_pattern = f"[*1..{max_hops}]"
    if "relation_type" in args:
        rel_type = args["relation_type"].upper().replace(" ", "_")
        rel_pattern = f"[:`{rel_type}`*1..{max_hops}]"

    post_where: List[str] = []
    if "entity_type" in args:
        post_where.append("toLower(neighbor.entity_type) = toLower($entity_type)")
        params["entity_type"] = args["entity_type"]
    if "filters" in args:
        post_where.extend(_build_filter_clauses(args["filters"], "neighbor", "nf", params))

    post_clause = f" AND {' AND '.join(post_where)}" if post_where else ""
    return (
        f"MATCH (e:Entity)-{rel_pattern}-(neighbor:Entity) "
        f"WHERE e.id = $entity_id{post_clause} "
        f"RETURN DISTINCT neighbor",
        params,
    )


def _translate_get_subgraph(args: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    max_hops = args.get("max_hops", 1)
    params: Dict[str, Any] = {"entity_ids": args["entity_ids"]}
    return (
        f"MATCH path = (e:Entity)-[*1..{max_hops}]-(other:Entity) "
        f"WHERE e.id IN $entity_ids "
        f"RETURN path",
        params,
    )


def _translate_get_chunks_for_entity(args: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    params: Dict[str, Any] = {"entity_id": args["entity_id"]}
    return (
        "MATCH (e:Entity)-[m:MENTIONED_IN]->(c:Chunk) "
        "WHERE e.id = $entity_id "
        "RETURN c, m",
        params,
    )


def _translate_find_shortest_path(args: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    max_depth = args.get("max_depth", 5)
    params: Dict[str, Any] = {
        "source_id": args["source_entity_id"],
        "target_id": args["target_entity_id"],
    }

    rel_pattern = f"[*..{max_depth}]"
    if "relation_type" in args:
        rel_type = args["relation_type"].upper().replace(" ", "_")
        rel_pattern = f"[:`{rel_type}`*..{max_depth}]"

    return (
        f"MATCH (src:Entity), (tgt:Entity) "
        f"WHERE src.id = $source_id AND tgt.id = $target_id "
        f"MATCH p = shortestPath((src)-{rel_pattern}-(tgt)) "
        f"RETURN p",
        params,
    )


# ---------------------------------------------------------------------------
# Translator registry  (tool_name → function(args) → (cypher, params))
# ---------------------------------------------------------------------------

_TOOL_TRANSLATORS: Dict[str, Callable[[Dict[str, Any]], Tuple[str, Dict[str, Any]]]] = {
    "search_entity": _translate_search_entity,
    "list_entities": _translate_list_entities,
    "get_entity_relations": _translate_get_entity_relations,
    "get_neighbors": _translate_get_neighbors,
    "get_subgraph": _translate_get_subgraph,
    "get_chunks_for_entity": _translate_get_chunks_for_entity,
    "find_shortest_path": _translate_find_shortest_path,
}


def register_tool_translator(
    tool_name: str,
    translator: Callable[[Dict[str, Any]], Tuple[str, Dict[str, Any]]],
) -> None:
    """Register a custom tool → Cypher translator.

    Args:
        tool_name: The function name (must match the tool definition).
        translator: ``fn(arguments_dict) -> (cypher_str, params_dict)``.
    """
    _TOOL_TRANSLATORS[tool_name] = translator


def tool_call_to_cypher(
    tool_name: str,
    arguments: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """Translate a structured tool call into a parameterised Cypher query.

    Args:
        tool_name: The tool/function name from the LLM response.
        arguments: Parsed arguments dict from the tool call.

    Returns:
        ``(cypher_query, params)`` tuple ready to execute on the graph store.

    Raises:
        KeyError: If no translator is registered for *tool_name*.
    """
    translator = _TOOL_TRANSLATORS.get(tool_name)
    if translator is None:
        raise KeyError(
            f"No Cypher translator registered for tool '{tool_name}'. "
            f"Available: {sorted(_TOOL_TRANSLATORS)}"
        )
    return translator(arguments)


# ---------------------------------------------------------------------------
# Graph schema prompt builder
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Relational (chunk/document) store tools
# ---------------------------------------------------------------------------

RELATIONAL_FILTER_SCHEMA: Dict[str, Any] = {
    "type": "array",
    "description": (
        "Filters on record fields. Each filter has a field name, "
        "an operator, and a value. Example: "
        '[{"field": "document_id", "operator": "eq", "value": "doc_001"}, '
        '{"field": "index", "operator": "gte", "value": 5}]'
    ),
    "items": {
        "type": "object",
        "properties": {
            "field": {
                "type": "string",
                "description": "The field name to filter on.",
            },
            "operator": {
                "type": "string",
                "enum": ["eq", "neq", "gt", "gte", "lt", "lte", "contains", "starts_with"],
                "description": (
                    "Comparison operator: eq, neq, gt, gte, lt, lte, "
                    "contains (substring), starts_with (prefix)."
                ),
            },
            "value": {
                "description": "The value to compare against.",
            },
        },
        "required": ["field", "operator", "value"],
    },
}

RELATIONAL_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_chunks",
            "description": (
                "Full-text search over stored text chunks. "
                "Returns the most relevant chunks matching the query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search text.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum results to return (default 10).",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_chunk",
            "description": "Retrieve a single text chunk by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "string",
                        "description": "The chunk ID.",
                    },
                },
                "required": ["chunk_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_records",
            "description": (
                "Query records from the chunk/document store with filtering "
                "and sorting. Use this for structured queries like 'all chunks "
                "from document X' or 'chunks with index > 5'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {
                        "type": "string",
                        "description": "Table name (default: 'chunks').",
                        "default": "chunks",
                    },
                    "filters": RELATIONAL_FILTER_SCHEMA,
                    "sort_by": {
                        "type": "string",
                        "description": "Field name to sort by.",
                    },
                    "sort_order": {
                        "type": "string",
                        "enum": ["asc", "desc"],
                        "description": "Sort direction (default: 'asc').",
                        "default": "asc",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results (default 50).",
                        "default": 50,
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Pagination offset (default 0).",
                        "default": 0,
                    },
                },
                "required": [],
            },
        },
    },
]

RELATIONAL_TOOL_NAMES: set = {"search_chunks", "get_chunk", "query_records"}


def execute_relational_tool(
    tool_name: str,
    args: Dict[str, Any],
    store: Any,
    chunk_table: str = "chunks",
) -> List[Dict[str, Any]]:
    """Dispatch a relational tool call to the appropriate store method.

    Args:
        tool_name: One of ``RELATIONAL_TOOL_NAMES``.
        args: Parsed arguments from the LLM tool call.
        store: A ``BaseRelationalStore`` instance.
        chunk_table: Default table name for chunk operations.

    Returns:
        List of record dicts.
    """
    if tool_name == "search_chunks":
        return store.search(
            table=chunk_table,
            query=args["query"],
            top_k=args.get("top_k", 10),
        )
    if tool_name == "get_chunk":
        record = store.get_record(chunk_table, args["chunk_id"])
        return [record] if record else []
    if tool_name == "query_records":
        return store.query_records(
            table=args.get("table", chunk_table),
            filters=args.get("filters"),
            sort_by=args.get("sort_by"),
            sort_order=args.get("sort_order", "asc"),
            limit=args.get("limit", 50),
            offset=args.get("offset", 0),
        )
    raise KeyError(f"Unknown relational tool: {tool_name}")


# ---------------------------------------------------------------------------
# Graph schema prompt builder
# ---------------------------------------------------------------------------

def build_graph_schema_prompt(
    entity_types: List[str],
    relation_types: List[str],
    property_keys: Optional[Dict[str, List[str]]] = None,
) -> str:
    """Build a system prompt section describing the graph schema.

    This is injected into the LLM context so it knows what entity types,
    relation types, and properties exist in the knowledge graph.

    Args:
        entity_types: List of entity types (e.g. ``["person", "organization"]``).
        relation_types: List of relation types (e.g. ``["WORKS_AT", "BORN_IN"]``).
        property_keys: Optional mapping of entity_type → list of property
            names (e.g. ``{"organization": ["founded_year", "location"]}``).

    Returns:
        A formatted string suitable for a system message.
    """
    lines = [
        "# Knowledge Graph Schema",
        "",
        "## Entity types",
    ]
    for et in entity_types:
        props_str = ""
        if property_keys and et in property_keys:
            props_str = f"  (properties: {', '.join(property_keys[et])})"
        lines.append(f"- {et}{props_str}")

    lines.append("")
    lines.append("## Relation types")
    for rt in relation_types:
        lines.append(f"- {rt}")

    lines.append("")
    lines.append(
        "Use the available tools to query the graph. "
        "Each tool call will be translated into a Cypher query and executed."
    )
    return "\n".join(lines)