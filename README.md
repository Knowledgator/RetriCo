# grapsit

End-to-end Graph RAG framework that turns unstructured text into a queryable knowledge graph. Built on [Knowledgator](https://github.com/Knowledgator) technologies (GLiNER, GLinker) with Neo4j for graph storage.

**Build pipeline**: `text chunking → entity recognition → entity linking → relation extraction → Neo4j graph`

**Query pipeline**: `query parsing → entity linking → subgraph retrieval → chunk retrieval → LLM reasoning`

Supports multiple extraction backends — mix and match freely:
- **GLiNER** — fast local inference, no API keys needed
- **LLM** — any OpenAI-compatible API (OpenAI, vLLM, Ollama, LM Studio, etc.)
- **GLinker** — entity linking against a reference knowledge base

## Installation

```bash
pip install -e .
```

For LLM-based extraction, also install the OpenAI SDK:

```bash
pip install openai
```

For entity linking, also install GLinker:

```bash
pip install glinker
```

Requires Python 3.10+ and a running [Neo4j](https://neo4j.com/) instance.

## Quickstart

One function call to go from raw text to a populated knowledge graph:

```python
import grapsit

result = grapsit.build_graph(
    texts=[
        "Albert Einstein was born in Ulm, Germany in 1879.",
        "Marie Curie worked at the University of Paris.",
    ],
    entity_labels=["person", "organization", "location", "date"],
    relation_labels=["born in", "works at", "located in"],
    neo4j_uri="bolt://localhost:7687",
    neo4j_password="password",
)

stats = result.get("writer_result")
print(f"Entities: {stats['entity_count']}, Relations: {stats['relation_count']}")
```

Query the graph with natural language:

```python
answer = grapsit.query_graph(
    query="Where was Albert Einstein born?",
    entity_labels=["person", "location"],
    neo4j_uri="bolt://localhost:7687",
    neo4j_password="password",
    api_key="sk-...",       # enables LLM reasoner
    model="gpt-4o-mini",
)
print(answer.answer)        # "Albert Einstein was born in Ulm, Germany."
print(answer.subgraph)      # retrieved entities, relations, source chunks
```

### What happens under the hood

**Build pipeline:**

1. **Chunking** — splits texts into sentences (configurable: sentence/paragraph/fixed)
2. **NER** — [GLiNER](https://github.com/urchade/GLiNER) or LLM extracts entities matching your labels
3. **Entity linking** (optional) — [GLinker](https://github.com/Knowledgator/GLinker) links entity mentions to a reference knowledge base
4. **Relation extraction** — [GLiNER-relex](https://huggingface.co/knowledgator/gliner-relex-large-v0.5) or LLM finds relations between entities
5. **Graph writing** — deduplicates entities (using linked IDs when available), writes nodes and edges to Neo4j

**Query pipeline:**

1. **Query parsing** — GLiNER or LLM extracts entities from the natural language query
2. **Entity linking** (optional) — links parsed entities to the knowledge base for precise lookup
3. **Retrieval** — looks up entities in Neo4j and expands to a k-hop subgraph
4. **Chunk retrieval** — fetches source text chunks where subgraph entities were mentioned
5. **Reasoning** (optional) — LLM generates an answer from the subgraph and source chunks

## Building the graph

### Option 1: `build_graph()` convenience function (GLiNER)

Simplest way — pass texts and labels, get a graph:

```python
import grapsit

result = grapsit.build_graph(
    texts=["Einstein developed relativity at the Swiss Patent Office in Bern."],
    entity_labels=["person", "organization", "location", "concept"],
    relation_labels=["works at", "developed", "located in"],
    # NER config
    ner_model="urchade/gliner_multi-v2.1",
    ner_threshold=0.3,
    # Relation extraction config
    relex_model="knowledgator/gliner-relex-large-v0.5",
    relex_threshold=0.5,
    # Chunking
    chunk_method="sentence",  # "sentence", "paragraph", or "fixed"
    # Neo4j
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password",
    # Hardware
    device="cpu",  # or "cuda"
    verbose=True,
)
```

Skip relation extraction by omitting `relation_labels` — only entities will be extracted and stored.

### Option 2: Builder API (GLiNER)

Fine-grained control over each pipeline stage:

```python
from grapsit import BuildConfigBuilder

builder = BuildConfigBuilder(name="my_pipeline")

builder.chunker(method="sentence")

builder.ner_gliner(
    model="knowledgator/gliner-decoder-small-v1.0",
    labels=["person", "organization", "location"],
    threshold=0.3,
    device="cpu",
)

builder.relex_gliner(
    model="knowledgator/gliner-relex-large-v0.5",
    entity_labels=["person", "organization", "location"],
    relation_labels=["works at", "located in", "founded"],
    relation_threshold=0.5,
)

builder.graph_writer(
    neo4j_uri="bolt://localhost:7687",
    neo4j_password="password",
)

# Build the executor and run
executor = builder.build(verbose=True)
result = executor.execute({"texts": ["Isaac Newton worked at Cambridge."]})

# Save config for reproducibility
builder.save("configs/my_pipeline.yaml")
```

### Option 3: LLM-based extraction

Use any OpenAI-compatible API for entity and relation extraction. This works with OpenAI, local vLLM servers, Ollama, LM Studio, or any other compatible endpoint.

#### Running a local LLM with vLLM

```bash
# Install vLLM
pip install vllm

# Start a local OpenAI-compatible server
vllm serve nowledgator/instruct-it-base --port 8000

# Or with GPU memory constraints
vllm serve nowledgator/instruct-it-base --port 8000 --gpu-memory-utilization 0.8

# Or using a quantized model for lower memory usage
vllm serve knowledgator/instruct-it-base --port 8000 --quantization awq
```

Other local LLM servers work too:

```bash
# Ollama
ollama serve
ollama pull qwen2.5:7b
# base_url = "http://localhost:11434/v1"

# LM Studio — start from the GUI, enable server mode
# base_url = "http://localhost:1234/v1"
```

#### All-LLM pipeline

Both NER and relation extraction are performed by the LLM:

```python
from grapsit import BuildConfigBuilder

builder = BuildConfigBuilder(name="llm_pipeline")
builder.chunker(method="sentence")

builder.ner_llm(
    base_url="http://localhost:8000/v1",  # your vLLM server
    api_key="dummy",                       # local servers don't need a real key
    model="Qwen/Qwen2.5-7B-Instruct",
    labels=["person", "organization", "location", "date"],
    temperature=0.1,
)

builder.relex_llm(
    base_url="http://localhost:8000/v1",
    api_key="dummy",
    model="Qwen/Qwen2.5-7B-Instruct",
    entity_labels=["person", "organization", "location", "date"],
    relation_labels=["works at", "born in", "located in"],
    temperature=0.1,
)

builder.graph_writer(
    neo4j_uri="bolt://localhost:7687",
    neo4j_password="password",
)

executor = builder.build(verbose=True)
result = executor.execute({"texts": ["Einstein was born in Ulm, Germany."]})
```

#### Using OpenAI

```python
builder.ner_llm(
    api_key="sk-...",      # or set OPENAI_API_KEY env var
    model="gpt-4o-mini",   # cost-effective for extraction
    labels=["person", "organization", "location"],
)
```

#### Mixed pipeline (GLiNER NER + LLM relation extraction)

Combine GLiNER's fast local NER with LLM's higher-quality relation extraction:

```python
from grapsit import BuildConfigBuilder

builder = BuildConfigBuilder(name="mixed_pipeline")
builder.chunker(method="sentence")

# Fast local NER with GLiNER
builder.ner_gliner(
    model="urchade/gliner_multi-v2.1",
    labels=["person", "organization", "location"],
    threshold=0.3,
)

# LLM for relation extraction (receives GLiNER entities)
builder.relex_llm(
    base_url="http://localhost:8000/v1",
    api_key="dummy",
    model="Qwen/Qwen2.5-7B-Instruct",
    entity_labels=["person", "organization", "location"],
    relation_labels=["works at", "born in", "located in", "founded"],
)

builder.graph_writer(
    neo4j_uri="bolt://localhost:7687",
    neo4j_password="password",
)

executor = builder.build(verbose=True)
result = executor.execute({"texts": ["Einstein worked at Princeton."]})
```

All NER and relex processors are interchangeable — any combination works:

| NER | Relex | Use case |
|-----|-------|----------|
| `ner_gliner` | `relex_gliner` | Fully local, fast, no API needed |
| `ner_llm` | `relex_llm` | Best quality, needs LLM server |
| `ner_gliner` | `relex_llm` | Fast NER + high-quality relations |
| `ner_llm` | `relex_gliner` | LLM entities + fast local relex |
| — | `relex_gliner` | Standalone: model does NER + relex |
| — | `relex_llm` | Standalone: LLM does NER + relex |

### Option 4: Entity linking with GLinker

Link extracted entity mentions to a reference knowledge base using [GLinker](https://github.com/Knowledgator/GLinker). Linked entities get stable IDs, which improves deduplication and enables precise entity lookup during queries.

#### With a pre-built GLinker executor

Full control over the GLinker pipeline:

```python
from glinker import ProcessorFactory as GLinkerFactory
from grapsit import BuildConfigBuilder

# Define your knowledge base
kb_entities = [
    {"entity_id": "Q937", "label": "Albert Einstein", "description": "theoretical physicist"},
    {"entity_id": "Q3012", "label": "Ulm", "description": "city in Germany"},
    {"entity_id": "Q183", "label": "Germany", "description": "country in Europe"},
]

# Create a GLinker executor
glinker_executor = GLinkerFactory.create_simple(
    model_name="knowledgator/gliner-linker-base-v1.0",
    entities=kb_entities,
    external_entities=True,  # will receive pre-extracted NER entities
)

builder = BuildConfigBuilder(name="linked_pipeline")
builder.chunker(method="sentence")
builder.ner_gliner(labels=["person", "location"])
builder.linker(executor=glinker_executor)      # links NER entities to KB
builder.relex_llm(api_key="sk-...", entity_labels=["person", "location"], relation_labels=["born in"])
builder.graph_writer(neo4j_uri="bolt://localhost:7687", neo4j_password="password")

executor = builder.build(verbose=True)
result = executor.execute({"texts": ["Einstein was born in Ulm, Germany."]})

# Linked entities use KB IDs (e.g. "Q937") instead of auto-generated UUIDs
linker_result = result.get("linker_result")
for chunk_ents in linker_result["entities"]:
    for ent in chunk_ents:
        print(f"{ent.text} -> {ent.linked_entity_id}")  # "Einstein -> Q937"
```

#### With grapsit-managed initialization

Let grapsit create the GLinker executor from parameters:

```python
builder.linker(
    model="knowledgator/gliner-linker-base-v1.0",
    entities="data/entities.jsonl",  # path to JSONL file, or list of dicts
    threshold=0.5,
)
```

#### End-to-end mode (no upstream NER)

GLinker can do NER + linking in one step. Skip `ner_gliner()`/`ner_llm()`:

```python
builder = BuildConfigBuilder(name="linker_only")
builder.chunker()
builder.linker(executor=glinker_executor)  # does NER + linking
builder.graph_writer(neo4j_uri="bolt://localhost:7687", neo4j_password="password")
```

#### With `build_graph()` convenience function

```python
result = grapsit.build_graph(
    texts=["Einstein was born in Ulm."],
    entity_labels=["person", "location"],
    relation_labels=["born in"],
    neo4j_uri="bolt://localhost:7687",
    neo4j_password="password",
    # Entity linking
    linker_executor=glinker_executor,    # or use linker_model + linker_entities
)
```

### Option 5: YAML pipeline

Define your pipeline declaratively and load it:

```yaml
# pipeline.yaml
name: my_pipeline
description: "GLiNER-based graph building"

nodes:
  - id: chunker
    processor: chunker
    inputs:
      texts:
        source: "$input"
        fields: texts
    output:
      key: chunker_result
    config:
      method: sentence

  - id: ner
    processor: ner_gliner
    requires: [chunker]
    inputs:
      chunks:
        source: chunker_result
        fields: chunks
    output:
      key: ner_result
    config:
      model: "urchade/gliner_multi-v2.1"
      labels: [person, organization, location]
      threshold: 0.3

  - id: relex
    processor: relex_gliner
    requires: [chunker]
    inputs:
      chunks:
        source: chunker_result
        fields: chunks
    output:
      key: relex_result
    config:
      model: "knowledgator/gliner-relex-large-v0.5"
      entity_labels: [person, organization, location]
      relation_labels: [works at, located in, founded by]
      threshold: 0.5
      relation_threshold: 0.5

  - id: graph_writer
    processor: graph_writer
    requires: [chunker, ner, relex]
    inputs:
      chunks:
        source: chunker_result
        fields: chunks
      documents:
        source: chunker_result
        fields: documents
      entities:
        source: ner_result
        fields: entities
      relations:
        source: relex_result
        fields: relations
    output:
      key: writer_result
    config:
      neo4j_uri: "bolt://localhost:7687"
      neo4j_password: password
```

```python
from grapsit import ProcessorFactory

executor = ProcessorFactory.create_pipeline("pipeline.yaml", verbose=True)
result = executor.execute({"texts": ["Your text here."]})
```

## Querying the graph

### Query pipeline

Use `query_graph()` for end-to-end query processing:

```python
import grapsit

# With LLM reasoner (generates natural language answer)
result = grapsit.query_graph(
    query="Where was Albert Einstein born?",
    entity_labels=["person", "location"],
    neo4j_uri="bolt://localhost:7687",
    neo4j_password="password",
    api_key="sk-...",           # enables LLM reasoner
    model="gpt-4o-mini",
    max_hops=2,                 # subgraph expansion depth
)
print(result.answer)            # "Albert Einstein was born in Ulm, Germany."
print(result.subgraph.entities) # retrieved entities
print(result.subgraph.relations)# relations in the subgraph
print(result.subgraph.chunks)   # relevant source chunks

# Without reasoner (returns subgraph only, no LLM needed)
result = grapsit.query_graph(
    query="Tell me about Marie Curie",
    entity_labels=["person", "organization"],
    neo4j_uri="bolt://localhost:7687",
    neo4j_password="password",
)
# result.answer is None, but result.subgraph has the retrieved data
```

### Query builder API

For full control over the query pipeline:

```python
from grapsit import QueryConfigBuilder

builder = QueryConfigBuilder(name="my_query")

# Parse the query to extract entities
builder.query_parser(method="gliner", labels=["person", "location"])
# Or use LLM: builder.query_parser(method="llm", api_key="sk-...", labels=["person"])

# Optional: link parsed entities to KB for precise retrieval
builder.linker(executor=glinker_executor)
# Or load KB from Neo4j: builder.linker(neo4j_uri="bolt://localhost:7687")

# Retrieve subgraph from Neo4j
builder.retriever(neo4j_uri="bolt://localhost:7687", max_hops=2)

# Fetch source chunks
builder.chunk_retriever()

# Optional: LLM reasoning over the subgraph
builder.reasoner(api_key="sk-...", model="gpt-4o-mini")

executor = builder.build(verbose=True)
ctx = executor.execute({"query": "Where was Einstein born?"})

# Access intermediate results
parser_result = ctx.get("parser_result")    # {"query": str, "entities": List[EntityMention]}
linker_result = ctx.get("linker_result")    # {"query": str, "entities": List[EntityMention]} (with linked_entity_id)
retriever_result = ctx.get("retriever_result")  # {"subgraph": Subgraph}
chunk_result = ctx.get("chunk_result")      # {"subgraph": Subgraph} (with chunks populated)
reasoner_result = ctx.get("reasoner_result")    # {"result": QueryResult}
```

### Direct Neo4j queries

After building, use `Neo4jGraphStore` to query directly:

```python
from grapsit import Neo4jGraphStore

store = Neo4jGraphStore(uri="bolt://localhost:7687", password="password")

# Look up an entity
entity = store.get_entity_by_label("Albert Einstein")

# Look up by ID (useful for linked entities)
entity = store.get_entity_by_id("Q937")

# Get its relations
relations = store.get_entity_relations(entity["id"])
for rel in relations:
    print(f"  --[{rel['relation_type']}]--> {rel['target_label']}")

# Get source text chunks where the entity was mentioned
chunks = store.get_chunks_for_entity(entity["id"])

# Get k-hop neighborhood
neighbors = store.get_entity_neighbors(entity["id"], max_hops=2)

# Get a subgraph around multiple entities
subgraph = store.get_subgraph(
    entity_ids=[entity["id"], neighbors[0]["id"]],
    max_hops=1,
)

# Get all entities (e.g. for loading as a knowledge base)
all_entities = store.get_all_entities()

store.close()
```

## Accessing intermediate results

`build_graph()` and `executor.execute()` return a `PipeContext` containing every pipeline stage output:

```python
result = grapsit.build_graph(texts=..., entity_labels=...)

# Chunks produced by the chunker
chunks = result.get("chunker_result")["chunks"]

# Per-chunk entity mentions from NER
entities = result.get("ner_result")["entities"]  # List[List[EntityMention]]

# Linked entities (if linker was enabled)
linked = result.get("linker_result")["entities"]  # List[List[EntityMention]] with linked_entity_id

# Per-chunk relations (if relex was enabled)
relations = result.get("relex_result")["relations"]  # List[List[Relation]]

# Final write stats + deduplicated entity map
writer = result.get("writer_result")
print(writer["entity_count"], writer["relation_count"])
entity_map = writer["entity_map"]  # dedup_key -> Entity
```

## Neo4j schema

The graph writer creates this schema in Neo4j:

```
(:Entity {id, label, entity_type, properties})
(:Chunk {id, document_id, text, index, start_char, end_char})
(:Document {id, source, metadata})

(entity)-[:MENTIONED_IN {start, end, score, text}]->(chunk)
(entity)-[:RELATION_TYPE {score, chunk_id, id}]->(entity)
(chunk)-[:PART_OF]->(document)
```

When entity linking is enabled, entity `id` values come from the knowledge base (e.g. Wikidata QIDs) instead of auto-generated UUIDs.

Verify your graph in Neo4j Browser:

```cypher
-- All entities
MATCH (n:Entity) RETURN n LIMIT 25

-- All relations (excluding mention links)
MATCH (a:Entity)-[r]->(b:Entity) WHERE NOT type(r) = 'MENTIONED_IN' RETURN a, r, b LIMIT 25

-- Entity neighborhood
MATCH (e:Entity {label: "Albert Einstein"})-[r]-(n) RETURN e, r, n
```

## Pipeline architecture

grapsit uses a DAG (directed acyclic graph) pipeline engine adapted from [GLinker](https://github.com/Knowledgator/GLinker). Each pipeline is a set of **nodes** connected by data dependencies:

**Build pipeline:**

```
$input (texts)
    │
    ▼
 chunker ──────────────────┐
    │                      │
    ▼                      │
 ner_gliner/ner_llm        │
    │                      │
    ▼                      │
 entity_linker (optional)  │
    │                      ▼
    │            relex_gliner/relex_llm
    │                      │
    └──────┐   ┌───────────┘
           ▼   ▼
       graph_writer
           │
           ▼
     Neo4j database
```

**Query pipeline:**

```
$input (query)
    │
    ▼
 query_parser (GLiNER or LLM)
    │
    ▼
 entity_linker (optional)
    │
    ▼
 retriever (Neo4j lookup + k-hop expansion)
    │
    ▼
 chunk_retriever (source text retrieval)
    │
    ▼
 reasoner (optional, LLM multi-hop reasoning)
    │
    ▼
 QueryResult (answer + subgraph + chunks)
```

Each node specifies:
- **processor** — registered processing function (e.g. `"chunker"`, `"ner_gliner"`)
- **inputs** — where to read data from (other node outputs or `$input`)
- **output** — key to store results under
- **requires** — explicit execution ordering
- **config** — processor-specific parameters

Nodes at the same level run sequentially within a level, but levels are topologically sorted so dependencies are always satisfied.

## Available processors

| Processor | Description | Key config |
|-----------|-------------|------------|
| `chunker` | Split text into chunks | `method` (sentence/paragraph/fixed), `chunk_size`, `overlap` |
| `ner_gliner` | Entity extraction with GLiNER | `model`, `labels`, `threshold`, `device` |
| `ner_llm` | Entity extraction with LLM | `model`, `labels`, `api_key`, `base_url`, `temperature` |
| `entity_linker` | Entity linking with GLinker | `executor`, `model`, `threshold`, `entities` |
| `relex_gliner` | Relation extraction with GLiNER-relex | `model`, `entity_labels`, `relation_labels`, `threshold`, `relation_threshold` |
| `relex_llm` | Relation extraction with LLM | `model`, `entity_labels`, `relation_labels`, `api_key`, `base_url`, `temperature` |
| `graph_writer` | Deduplicate and write to Neo4j | `neo4j_uri`, `neo4j_user`, `neo4j_password`, `neo4j_database` |
| `query_parser` | Extract entities from a query | `method` (gliner/llm), `labels`, `model`, `api_key` |
| `retriever` | Look up entities + k-hop subgraph | `neo4j_uri`, `max_hops` |
| `chunk_retriever` | Fetch source chunks for entities | `neo4j_uri`, `max_chunks` |
| `reasoner` | LLM multi-hop reasoning | `method` (llm), `model`, `api_key`, `base_url` |

## Development

```bash
# Create venv and install
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# Run tests (mocked GLiNER + GLinker + Neo4j, no external services needed)
.venv/bin/pytest tests/ -v    # 139 tests
```

## Roadmap

- ~~**LLM extraction** — OpenAI-compatible NER and relation extraction as alternative to GLiNER~~ Done
- ~~**Entity linking** — GLinker integration for linking mentions to a knowledge base~~ Done
- ~~**Query pipeline** — DAG-based retrieval: query parsing, subgraph retrieval, LLM reasoning~~ Done
- **KG modeling** — node/edge embeddings, community detection (Leiden), path reasoning
- **In-memory store** — testing without Neo4j
- **CLI** — `grapsit build --config pipeline.yaml`
