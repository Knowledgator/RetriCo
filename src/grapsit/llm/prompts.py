"""Centralized default LLM prompts for all grapsit processors.

Each processor that uses LLM prompting imports its defaults from here.
Users can override any prompt via the processor's config dict by passing
``system_prompt`` and/or ``user_prompt_template`` (or the processor-specific
key documented in the processor's docstring).
"""

# ---------------------------------------------------------------------------
# NER LLM
# ---------------------------------------------------------------------------

NER_SYSTEM_PROMPT = (
    "You are a named entity recognition system. "
    "Extract entities from the given text and return them as JSON."
)

NER_USER_PROMPT_TEMPLATE = """\
Extract all named entities from the text below.

{labels_instruction}

Text:
\"\"\"{text}\"\"\"

Return a JSON array of objects with these fields:
- "text": the exact entity text as it appears
- "label": the entity type{label_constraint}

Return ONLY the JSON array, no other text."""

# ---------------------------------------------------------------------------
# Relation Extraction LLM
# ---------------------------------------------------------------------------

RELEX_SYSTEM_PROMPT = (
    "You are a relation extraction system. "
    "Extract relationships between entities in the given text and return them as JSON."
)

RELEX_STANDALONE_PROMPT_TEMPLATE = """\
Extract all entities and relationships from the text below.

{entity_labels_instruction}
{relation_labels_instruction}

Text:
\"\"\"{text}\"\"\"

Return a JSON object with two fields:
- "entities": array of {{"text": str, "label": str}}
- "relations": array of {{"head": str, "tail": str, "relation": str}}

Where "head" and "tail" are the exact entity text as it appears in the text.
Return ONLY the JSON object, no other text."""

RELEX_WITH_ENTITIES_PROMPT_TEMPLATE = """\
Given the entities already extracted from the text, find relationships between them.

{relation_labels_instruction}

Entities found:
{entities_list}

Text:
\"\"\"{text}\"\"\"

Return a JSON array of relationship objects:
{{"head": str, "tail": str, "relation": str}}

Where "head" and "tail" are the exact entity text from the list above.
Return ONLY the JSON array, no other text."""

# ---------------------------------------------------------------------------
# Query Parser (LLM mode)
# ---------------------------------------------------------------------------

QUERY_PARSER_SYSTEM_PROMPT = (
    "You are an entity and concept extraction system. "
    "Extract named entities, key concepts, scientific terms, "
    "and domain-specific terminology from queries."
)

QUERY_PARSER_USER_PROMPT_TEMPLATE = """\
Extract all named entities, key concepts, and important terms from this query.

{labels_instruction}

Query: "{query}"

Return a JSON object with an "entities" key containing an array of objects with:
- "text": the entity or concept text as it appears in the query
- "label": the entity type{label_constraint}

IMPORTANT: Extract domain-specific terms, concepts, and topics — not just proper nouns.
Return ONLY valid JSON."""

# ---------------------------------------------------------------------------
# Query Parser (tool-calling mode)
# ---------------------------------------------------------------------------

QUERY_PARSER_TOOL_SYSTEM_PROMPT = """\
You are a knowledge graph query decomposer. \
Given a natural language question, decompose it into one or more \
triple patterns by calling the search_triples tool.

Each call specifies a (head, relation, tail) pattern where unknown \
parts should be null.

Examples:
- "Where was Einstein born?" → search_triples(head="Einstein", relation="born_in", tail=null)
- "Who works at MIT?" → search_triples(head=null, relation="works_at", tail="MIT")
- "What is the relationship between Einstein and Bohr?" → \
search_triples(head="Einstein", relation=null, tail="Bohr")
"""

# ---------------------------------------------------------------------------
# Reasoner
# ---------------------------------------------------------------------------

REASONER_SYSTEM_PROMPT = (
    "You are a knowledge graph reasoning system. "
    "Given a query and a subgraph of entities, relations, and source text chunks, "
    "you analyze the information to answer the query."
)

REASONER_USER_PROMPT_TEMPLATE = """\
Query: {query}

Knowledge Graph Context:
{context}

Based on the subgraph above, respond with a JSON object:
{{
  "inferred_relations": [{{"head": "entity1", "tail": "entity2", "relation": "relation_type"}}],
  "relevant_chunk_ids": ["chunk_id_1", ...],
  "answer": "Your concise answer to the query"
}}

- inferred_relations: new relations you can infer that are not explicitly stated
- relevant_chunk_ids: IDs of the most relevant source chunks for answering
- answer: a concise answer based on the available evidence

Return ONLY the JSON object."""

# ---------------------------------------------------------------------------
# Tool Retriever (agent)
# ---------------------------------------------------------------------------

TOOL_RETRIEVER_SYSTEM_PROMPT = """\
You are a knowledge graph query agent. Use the available tools \
to find information relevant to the user's question.

{schema_prompt}

IMPORTANT: Entity nodes only have these properties: id, label, \
entity_type. Do NOT use property filters for attributes like \
'location', 'founded_year', etc. — they do not exist. Instead:
- Use search_entity to find entities by label text.
- Use get_entity_relations and get_neighbors to discover \
connections between entities.
- Use find_shortest_path to find paths between two entities.
{chunk_tools_prompt}
Call tools to search the graph. When you have enough information, \
respond with a final text summary."""

CHUNK_TOOLS_PROMPT = """
You also have access to a chunk/document store. Use search_chunks \
for full-text search over source text, get_chunk to retrieve a \
chunk by ID, and query_records for filtered queries with sorting \
(e.g. all chunks from a specific document)."""

# ---------------------------------------------------------------------------
# Community Summarizer
# ---------------------------------------------------------------------------

COMMUNITY_SUMMARIZER_SYSTEM_PROMPT = "You are a knowledge graph analyst."

COMMUNITY_SUMMARIZER_USER_PROMPT_TEMPLATE = """\
You are analyzing a community of entities in a knowledge graph. \
Based on the following entities and their relationships, provide:
1. A short title (max 10 words) describing the community theme
2. A concise summary (2-3 sentences) of what this community represents

Entities and relationships:
{context}

Respond in this exact format:
TITLE: <title>
SUMMARY: <summary>"""