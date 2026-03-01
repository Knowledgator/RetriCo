# grapsit

End-to-end Graph RAG framework that turns unstructured text into a queryable knowledge graph. Built on [Knowledgator](https://github.com/Knowledgator) technologies (GLiNER, GLinker) with Neo4j, FalkorDB, or Memgraph for graph storage.

**Build pipeline**: `text chunking → entity recognition → entity linking → relation extraction → graph store (Neo4j / FalkorDB / Memgraph) → chunk/entity embedding (optional)` — texts can come from `$input` or be pulled from a relational store via `store_reader`

**Ingest pipeline**: `structured JSON → graph store` (bypass NER/relex, write pre-structured data directly)

**Query pipeline**: `query parsing → entity linking → subgraph retrieval → chunk retrieval → LLM reasoning` (9 retrieval strategies)

**Community pipeline**: `community detection → LLM summarization → vector embedding`

**KG modeling pipeline**: `triple reading → PyKEEN model training → embedding storage` + query-time link prediction

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

For FalkorDB as an alternative graph store:

```bash
pip install falkordb
```

Memgraph uses the same Neo4j Python driver (Bolt protocol), so no additional dependencies are needed beyond `neo4j`.

For KG embedding training with PyKEEN:

```bash
pip install pykeen
```

Requires Python 3.10+ and a running graph database — [Neo4j](https://neo4j.com/), [FalkorDB](https://www.falkordb.com/), or [Memgraph](https://memgraph.com/).

## Database stores

grapsit uses three categories of database stores — each serves a distinct purpose in the pipeline. You can mix and match backends freely within each category.

### Store categories

| Category | Purpose | Used by | Required? |
|----------|---------|---------|-----------|
| **Graph store** | Stores entities, relations, and the knowledge graph structure | `graph_writer`, all retrievers, community detection | Yes — at least one graph store is needed |
| **Vector store** | Stores embeddings for similarity search | `chunk_embedder`, `entity_embedder`, community embedder, embedding-based retrievers | Only if using embedding-based features |
| **Relational store** | Stores text chunks and documents in tabular format with full-text search | `graph_writer` (chunk/doc persistence), `store_reader`, `keyword_retriever`, `tool_retriever` (chunk tools) | Only if using chunk storage, keyword search, or store_reader |

### Graph stores

Graph stores hold the knowledge graph — entities as nodes, relations as edges, plus chunk and document nodes.

| Backend | Config class | Python dependency | Startup |
|---------|-------------|-------------------|---------|
| [Neo4j](https://neo4j.com/) | `Neo4jConfig` | `neo4j` (included) | `docker run -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j` |
| [FalkorDB](https://www.falkordb.com/) | `FalkorDBConfig` | `falkordb` (optional) | `docker run -p 6379:6379 -it --rm falkordb/falkordb` |
| [Memgraph](https://memgraph.com/) | `MemgraphConfig` | `neo4j` (shared driver) | `docker run -p 7687:7687 memgraph/memgraph-platform` |

```python
from grapsit import Neo4jConfig, FalkorDBConfig, MemgraphConfig

# Neo4j
neo4j_cfg = Neo4jConfig(
    uri="bolt://localhost:7687",
    user="neo4j",          # default: "neo4j"
    password="password",
    database="neo4j",      # default: "neo4j"
)

# FalkorDB
falkor_cfg = FalkorDBConfig(
    host="localhost",      # default: "localhost"
    port=6379,             # default: 6379
    graph="knowledge",     # default: "knowledge_graph"
)

# Memgraph (uses Bolt protocol, same driver as Neo4j)
memgraph_cfg = MemgraphConfig(
    uri="bolt://localhost:7687",
    user="",               # default: "" (no auth)
    password="",           # default: "" (no auth)
)
```

Or use flat keyword arguments (no config object needed):

```python
builder.graph_writer(
    store_type="neo4j",  # or "falkordb", "memgraph"
    neo4j_uri="bolt://localhost:7687",
    neo4j_password="password",
)
```

### Vector stores

Vector stores hold embeddings for chunks, entities, and community summaries. Used by embedding-based retrieval strategies.

| Backend | Config class | Python dependency | Description |
|---------|-------------|-------------------|-------------|
| In-memory | `InMemoryVectorConfig` | — (NumPy) | Process-scoped singleton, cosine similarity. Good for testing. |
| [FAISS](https://github.com/facebookresearch/faiss) | `FaissVectorConfig` | `faiss-cpu` or `faiss-gpu` | Fast similarity search, CPU or GPU |
| [Qdrant](https://qdrant.tech/) | `QdrantVectorConfig` | `qdrant-client` | Dedicated vector DB, local or remote |
| Graph DB native | `GraphDBVectorConfig` | — | Uses the graph store's built-in vector index (Neo4j, FalkorDB) |

```python
from grapsit import InMemoryVectorConfig, FaissVectorConfig, QdrantVectorConfig, GraphDBVectorConfig

in_mem = InMemoryVectorConfig()
faiss_cfg = FaissVectorConfig(use_gpu=False)
qdrant_cfg = QdrantVectorConfig(url="http://localhost:6333", api_key="...")
graph_vec = GraphDBVectorConfig(graph_store_name="default")  # reuses named graph store's vector index
```

### Relational stores

Relational stores hold text chunks and documents in a tabular format with full-text search support. They serve multiple purposes:
- **Chunk/document persistence** — `graph_writer` can write chunks to a relational store alongside the graph
- **Source reading** — `store_reader` pulls texts from a relational store to feed the build pipeline
- **Keyword retrieval** — `keyword_retriever` uses full-text search for query-time chunk lookup
- **Tool retriever chunk tools** — `search_chunks`, `get_chunk`, `query_records` tools for the LLM agent

| Backend | Config class | Python dependency | Full-text search |
|---------|-------------|-------------------|-----------------|
| [SQLite](https://www.sqlite.org/) | `SqliteRelationalConfig` | — (stdlib) | FTS5 |
| [PostgreSQL](https://www.postgresql.org/) | `PostgresRelationalConfig` | `psycopg[binary]` | tsvector + GIN indexes |
| [Elasticsearch](https://www.elastic.co/) | `ElasticsearchRelationalConfig` | `elasticsearch` | Multi-match queries |

```python
from grapsit import SqliteRelationalConfig, PostgresRelationalConfig, ElasticsearchRelationalConfig

sqlite_cfg = SqliteRelationalConfig(path="chunks.db")  # or ":memory:" for in-memory
postgres_cfg = PostgresRelationalConfig(
    host="localhost",
    port=5432,
    user="myuser",
    password="mypass",
    database="mydb",
)
es_cfg = ElasticsearchRelationalConfig(
    url="http://localhost:9200",
    api_key="...",              # optional
    index_prefix="grapsit_",   # default prefix for ES indices
)
```

All relational stores auto-create tables and indexes on first write — no manual schema setup needed.

### Registering stores in a pipeline

Use the builder's `graph_store()`, `vector_store()`, and `chunk_store()` methods to register named stores that are shared across all pipeline nodes:

```python
from grapsit import BuildConfigBuilder, Neo4jConfig, FaissVectorConfig, SqliteRelationalConfig

builder = BuildConfigBuilder(name="full_pipeline")

# Register named stores (shared across all processors)
builder.graph_store(Neo4jConfig(uri="bolt://localhost:7687", password="pass"), name="main")
builder.vector_store(FaissVectorConfig(use_gpu=True), name="embeddings")
builder.chunk_store(SqliteRelationalConfig(path="chunks.db"), name="chunks")

# Pipeline nodes — no need to repeat connection details
builder.chunker(method="sentence")
builder.ner_gliner(labels=["person", "organization", "location"])
builder.graph_writer()          # uses "main" graph store + "chunks" relational store
builder.chunk_embedder()        # uses "main" graph store + "embeddings" vector store

with builder.build() as executor:
    result = executor.execute({"texts": ["Einstein was born in Ulm."]})
# All connections closed automatically
```

Without explicit store registration, processors create their own connections from flat config parameters (backward compatible). See [Store pool](#store-pool-shared-connections) for more details on connection sharing.

### Creating stores directly

For programmatic use outside a pipeline:

```python
from grapsit import create_graph_store, create_vector_store, create_relational_store

# From flat dicts
graph = create_graph_store({"store_type": "neo4j", "neo4j_uri": "bolt://localhost:7687"})
vector = create_vector_store({"vector_store_type": "faiss", "use_gpu": True})
relational = create_relational_store({"relational_store_type": "sqlite", "sqlite_path": "chunks.db"})

# From config objects
graph = create_graph_store(Neo4jConfig(uri="bolt://localhost:7687").to_flat_dict())
```

Or import store classes directly:

```python
from grapsit import Neo4jGraphStore, FalkorDBGraphStore, MemgraphGraphStore
from grapsit import SqliteRelationalStore, PostgresRelationalStore, ElasticsearchRelationalStore

store = Neo4jGraphStore(uri="bolt://localhost:7687", password="password")
entity = store.get_entity_by_label("Einstein")
store.close()
```

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

1. **Store reading** (optional) — pulls texts from a relational store (SQLite, PostgreSQL, or Elasticsearch) instead of requiring them in `$input`
2. **Chunking** — splits texts into sentences (configurable: sentence/paragraph/fixed)
3. **NER** — [GLiNER](https://github.com/urchade/GLiNER) or LLM extracts entities matching your labels
4. **Entity linking** (optional) — [GLinker](https://github.com/Knowledgator/GLinker) links entity mentions to a reference knowledge base
5. **Relation extraction** — [GLiNER-relex](https://huggingface.co/knowledgator/gliner-relex-large-v0.5) or LLM finds relations between entities
6. **Graph writing** — deduplicates entities (using linked IDs when available), writes nodes and edges to Neo4j, FalkorDB, or Memgraph
7. **Chunk embedding** (optional) — embeds chunk texts into a vector store for semantic search
8. **Entity embedding** (optional) — embeds entity labels into a vector store for similarity queries

**KG modeling pipeline** (offline, after graph construction):

1. **Triple reading** — reads all triples from the graph store (or TSV file)
2. **Training** — trains a PyKEEN KG embedding model (RotatE, TransE, ComplEx, etc.)
3. **Storing** — saves entity/relation embeddings to vector store and disk, optionally writes to graph DB

**Query pipeline** (9 retrieval strategies):

1. **Query parsing** — GLiNER, LLM, or tool-calling parser extracts entities/triple patterns from the query
2. **Entity linking** (optional) — links parsed entities to the knowledge base for precise lookup
3. **Retrieval** — one of 9 strategies (entity lookup, community, chunk embedding, entity embedding, tool-based, path, KG-scored, keyword)
4. **KG scoring** (optional) — scores retrieved triples and predicts missing links using trained KG embeddings; in KG-scored mode, acts as the retriever
5. **Chunk retrieval** — fetches source text chunks where subgraph entities were mentioned (configurable entity filtering)
6. **Reasoning** (optional) — LLM generates an answer from the subgraph and source chunks

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
vllm serve knowledgator/instruct-it-base --port 8000

# Or with GPU memory constraints
vllm serve knowledgator/instruct-it-base --port 8000 --gpu-memory-utilization 0.8

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

### Using FalkorDB instead of Neo4j

[FalkorDB](https://www.falkordb.com/) is a Redis-based graph database that supports OpenCypher. It can be used as a drop-in replacement for Neo4j in all pipelines.

#### Starting FalkorDB

```bash
docker run -p 6379:6379 -it --rm falkordb/falkordb
```

#### `build_graph()` with FalkorDB

```python
import grapsit

result = grapsit.build_graph(
    texts=["Einstein was born in Ulm, Germany in 1879."],
    entity_labels=["person", "location", "date"],
    relation_labels=["born in"],
    store_type="falkordb",
    falkordb_host="localhost",
    falkordb_port=6379,
    falkordb_graph="my_knowledge_graph",
)
```

#### Builder API with FalkorDB

```python
from grapsit import BuildConfigBuilder

builder = BuildConfigBuilder(name="falkordb_pipeline")
builder.chunker(method="sentence")
builder.ner_gliner(labels=["person", "organization", "location"])
builder.relex_gliner(
    entity_labels=["person", "organization", "location"],
    relation_labels=["works at", "born in", "located in"],
)
builder.graph_writer(
    store_type="falkordb",
    falkordb_host="localhost",
    falkordb_port=6379,
    falkordb_graph="my_knowledge_graph",
)

executor = builder.build(verbose=True)
result = executor.execute({"texts": ["Einstein worked at Princeton."]})
```

#### Querying with FalkorDB

```python
result = grapsit.query_graph(
    query="Where was Einstein born?",
    entity_labels=["person", "location"],
    store_type="falkordb",
    falkordb_host="localhost",
    falkordb_port=6379,
    falkordb_graph="my_knowledge_graph",
    api_key="sk-...",
    model="gpt-4o-mini",
)
print(result.answer)
```

#### Query builder API with FalkorDB

```python
from grapsit import QueryConfigBuilder

builder = QueryConfigBuilder(name="falkordb_query")
builder.query_parser(method="gliner", labels=["person", "location"])
builder.retriever(
    store_type="falkordb",
    falkordb_host="localhost",
    falkordb_port=6379,
    falkordb_graph="my_knowledge_graph",
    max_hops=2,
)
builder.chunk_retriever()
builder.reasoner(api_key="sk-...", model="gpt-4o-mini")

executor = builder.build()
ctx = executor.execute({"query": "Where was Einstein born?"})
```

#### Direct FalkorDB queries

```python
from grapsit import FalkorDBGraphStore

store = FalkorDBGraphStore(host="localhost", port=6379, graph="my_knowledge_graph")

entity = store.get_entity_by_label("Albert Einstein")
relations = store.get_entity_relations(entity["id"])
neighbors = store.get_entity_neighbors(entity["id"], max_hops=2)
subgraph = store.get_subgraph(entity_ids=[entity["id"]], max_hops=1)

store.close()
```

#### YAML config with FalkorDB

```yaml
nodes:
  - id: graph_writer
    processor: graph_writer
    config:
      store_type: falkordb
      falkordb_host: localhost
      falkordb_port: 6379
      falkordb_graph: my_knowledge_graph
```

The `store_type` parameter works everywhere — `build_graph()`, `query_graph()`, builder methods, and YAML configs. All processors (`graph_writer`, `retriever`, `chunk_retriever`, `entity_linker`) use the same parameter to select the backend.

### Using Memgraph instead of Neo4j

[Memgraph](https://memgraph.com/) is a high-performance in-memory graph database that uses the Bolt protocol and OpenCypher query language. It is compatible with the Neo4j Python driver, so no additional dependencies are needed.

#### Starting Memgraph

```bash
# Using Docker (Memgraph Platform includes Lab + MAGE algorithms)
docker run -p 7687:7687 -p 3000:3000 -p 7444:7444 memgraph/memgraph-platform

# Or just the database
docker run -p 7687:7687 memgraph/memgraph
```

#### `build_graph()` with Memgraph

```python
import grapsit

result = grapsit.build_graph(
    texts=["Einstein was born in Ulm, Germany in 1879."],
    entity_labels=["person", "location", "date"],
    relation_labels=["born in"],
    store_type="memgraph",
    memgraph_uri="bolt://localhost:7687",
)
```

#### Builder API with Memgraph

```python
from grapsit import BuildConfigBuilder

builder = BuildConfigBuilder(name="memgraph_pipeline")
builder.chunker(method="sentence")
builder.ner_gliner(labels=["person", "organization", "location"])
builder.relex_gliner(
    entity_labels=["person", "organization", "location"],
    relation_labels=["works at", "born in", "located in"],
)
builder.graph_writer(
    store_type="memgraph",
    memgraph_uri="bolt://localhost:7687",
)

executor = builder.build(verbose=True)
result = executor.execute({"texts": ["Einstein worked at Princeton."]})
```

#### Querying with Memgraph

```python
result = grapsit.query_graph(
    query="Where was Einstein born?",
    entity_labels=["person", "location"],
    store_type="memgraph",
    memgraph_uri="bolt://localhost:7687",
    api_key="sk-...",
    model="gpt-4o-mini",
)
print(result.answer)
```

#### Query builder API with Memgraph

```python
from grapsit import QueryConfigBuilder

builder = QueryConfigBuilder(name="memgraph_query")
builder.query_parser(method="gliner", labels=["person", "location"])
builder.retriever(
    store_type="memgraph",
    memgraph_uri="bolt://localhost:7687",
    max_hops=2,
)
builder.chunk_retriever()
builder.reasoner(api_key="sk-...", model="gpt-4o-mini")

executor = builder.build()
ctx = executor.execute({"query": "Where was Einstein born?"})
```

#### Direct Memgraph queries

```python
from grapsit import MemgraphGraphStore

store = MemgraphGraphStore(uri="bolt://localhost:7687")

entity = store.get_entity_by_label("Albert Einstein")
relations = store.get_entity_relations(entity["id"])
neighbors = store.get_entity_neighbors(entity["id"], max_hops=2)
subgraph = store.get_subgraph(entity_ids=[entity["id"]], max_hops=1)

store.close()
```

#### YAML config with Memgraph

```yaml
nodes:
  - id: graph_writer
    processor: graph_writer
    config:
      store_type: memgraph
      memgraph_uri: bolt://localhost:7687
```

Memgraph defaults to no authentication (empty user/password). If you've configured authentication in Memgraph, pass `memgraph_user` and `memgraph_password`.

## Ingesting structured data

If you already have structured entities and relations (e.g. from an external source, a CSV, or a previous export), you can write them directly to the graph database — no chunking, NER, or relation extraction needed.

### `ingest_data()` convenience function

```python
import grapsit

ctx = grapsit.ingest_data(
    entities=[
        {"text": "Albert Einstein", "label": "person"},
        {"text": "Ulm", "label": "location"},
        {"text": "Princeton University", "label": "organization"},
    ],
    relations=[
        {"head": "Albert Einstein", "tail": "Ulm", "type": "born_in", "score": 1.0},
        {"head": "Albert Einstein", "tail": "Princeton University", "type": "works_at"},
    ],
    neo4j_uri="bolt://localhost:7687",
    neo4j_password="password",
)

stats = ctx.get("writer_result")
print(f"Entities: {stats['entity_count']}, Relations: {stats['relation_count']}")
```

### Input format

**Entities** — each dict requires `text` and `label`:

```python
{"text": "Einstein", "label": "person"}                          # minimal
{"text": "Einstein", "label": "person", "id": "Q937"}           # explicit ID (used for dedup)
{"text": "Einstein", "label": "person", "score": 0.95}          # with confidence
```

**Relations** — each dict requires `head`, `tail`, and `type`:

```python
{"head": "Einstein", "tail": "Ulm", "type": "born_in"}                          # minimal
{"head": "Einstein", "tail": "Ulm", "type": "born_in", "score": 0.9}            # with score
{"head": "Einstein", "tail": "Ulm", "type": "born_in", "head_label": "person",
 "tail_label": "location", "properties": {"year": 1879}}                         # full
```

The `head` and `tail` values must match an entity `text` (case-insensitive).

### Ingest builder API

```python
from grapsit import IngestConfigBuilder

builder = IngestConfigBuilder(name="my_ingest")
builder.graph_writer(
    store_type="memgraph",
    memgraph_uri="bolt://localhost:7687",
)

executor = builder.build()
ctx = executor.execute({
    "entities": [
        {"text": "Einstein", "label": "person"},
        {"text": "Ulm", "label": "location"},
    ],
    "relations": [
        {"head": "Einstein", "tail": "Ulm", "type": "born_in"},
    ],
})

# Save config for reproducibility
builder.save("configs/ingest.yaml")
```

### Ingesting from a JSON file

The ingest format is designed to be loaded directly from JSON:

```python
import json
import grapsit

with open("data/knowledge_graph.json") as f:
    data = json.load(f)

ctx = grapsit.ingest_data(
    entities=data["entities"],
    relations=data["relations"],
    neo4j_uri="bolt://localhost:7687",
)
```

## Exporting to JSON

The `graph_writer` can save extracted data to a JSON file alongside writing to the database. The JSON file uses the same format as `ingest_data()`, so it can be loaded back later.

### With `build_graph()`

```python
result = grapsit.build_graph(
    texts=["Einstein was born in Ulm, Germany in 1879."],
    entity_labels=["person", "location"],
    relation_labels=["born in"],
    neo4j_uri="bolt://localhost:7687",
    neo4j_password="password",
    json_output="output/extracted_data.json",  # save extracted data
)
```

### With the builder API

```python
builder = BuildConfigBuilder(name="my_pipeline")
builder.chunker(method="sentence")
builder.ner_gliner(labels=["person", "location"])
builder.relex_gliner(
    entity_labels=["person", "location"],
    relation_labels=["born in", "located in"],
)
builder.graph_writer(
    neo4j_uri="bolt://localhost:7687",
    json_output="output/extracted_data.json",
)

executor = builder.build()
result = executor.execute({"texts": ["Einstein was born in Ulm."]})
```

### Output format

The JSON file contains deduplicated entities and resolved relations:

```json
{
  "entities": [
    {"text": "Einstein", "label": "person", "id": "..."},
    {"text": "Ulm", "label": "location", "id": "..."}
  ],
  "relations": [
    {"head": "Einstein", "tail": "Ulm", "type": "born in", "score": 0.8}
  ]
}
```

### Round-trip: build, export, re-ingest

Export from one database and import into another:

```python
import json
import grapsit

# Build graph and export to JSON
grapsit.build_graph(
    texts=["Einstein was born in Ulm."],
    entity_labels=["person", "location"],
    relation_labels=["born in"],
    neo4j_uri="bolt://localhost:7687",
    json_output="data/graph_export.json",
)

# Later, ingest into a different database
with open("data/graph_export.json") as f:
    data = json.load(f)

grapsit.ingest_data(
    entities=data["entities"],
    relations=data["relations"],
    store_type="memgraph",
    memgraph_uri="bolt://localhost:7688",
)
```

## Embedding chunks and entities

After building a knowledge graph, you can embed chunk texts and/or entity labels into a vector store. This enables the `chunk_embedding` and `entity_embedding` retrieval strategies during queries.

Both embedders run **after** `graph_writer` in the build pipeline. They use the existing embedding model factory (`sentence_transformer` or `openai`) and vector store factory (`in_memory`, `faiss`, or `qdrant`). Embeddings are stored in the vector store and optionally persisted on graph nodes.

### With `build_graph()`

```python
import grapsit

result = grapsit.build_graph(
    texts=["Einstein was born in Ulm, Germany in 1879."],
    entity_labels=["person", "location", "date"],
    relation_labels=["born in"],
    neo4j_uri="bolt://localhost:7687",
    neo4j_password="password",
    # Enable embeddings
    embed_chunks=True,
    embed_entities=True,
    embedding_method="sentence_transformer",  # or "openai"
    embedding_model_name="all-MiniLM-L6-v2",
    vector_store_type="in_memory",            # or "faiss", "qdrant"
)

# Embedding results
chunk_emb = result.get("chunk_embedder_result")
print(f"Embedded {chunk_emb['embedded_count']} chunks (dim={chunk_emb['dimension']})")

entity_emb = result.get("entity_embedder_result")
print(f"Embedded {entity_emb['embedded_count']} entities (dim={entity_emb['dimension']})")
```

### With the builder API

```python
from grapsit import BuildConfigBuilder

builder = BuildConfigBuilder(name="embedded_pipeline")
builder.chunker(method="sentence")
builder.ner_gliner(labels=["person", "organization", "location"])
builder.relex_gliner(
    entity_labels=["person", "organization", "location"],
    relation_labels=["works at", "born in"],
)
builder.graph_writer(
    neo4j_uri="bolt://localhost:7687",
    neo4j_password="password",
)

# Add embedders (run after graph_writer)
builder.chunk_embedder(
    embedding_method="sentence_transformer",
    model_name="all-MiniLM-L6-v2",
    vector_store_type="in_memory",
    vector_index_name="chunk_embeddings",   # default
)
builder.entity_embedder(
    embedding_method="sentence_transformer",
    model_name="all-MiniLM-L6-v2",
    vector_store_type="faiss",              # or "qdrant", "in_memory"
    vector_index_name="entity_embeddings",  # default
)

executor = builder.build(verbose=True)
result = executor.execute({"texts": ["Einstein worked at Princeton."]})

# Save config (embedder nodes included)
builder.save("configs/my_embedded_pipeline.yaml")
```

### Using OpenAI embeddings

```python
builder.chunk_embedder(
    embedding_method="openai",
    api_key="sk-...",
    model_name="text-embedding-3-small",
)
```

### How it works

- **Chunk embedder** — encodes each chunk's text, stores `(chunk_id, embedding)` pairs in the vector store, and writes `embedding` property on Chunk nodes in the graph database.
- **Entity embedder** — encodes each entity's label, stores `(entity_id, embedding)` pairs in the vector store, and writes `embedding` property on Entity nodes in the graph database.
- Both embedders inherit store connection parameters (URI, credentials) from the `graph_writer` config automatically.
- The vector index names (`chunk_embeddings`, `entity_embeddings`) match the defaults expected by the `chunk_embedding_retriever` and `entity_embedding_retriever` query strategies.

### YAML config

See `configs/build_gliner_embed.yaml` for a complete example with both embedders.

## Building from a relational store

Instead of passing texts directly via `execute({"texts": [...]})`, you can pull texts from an existing relational database (SQLite, PostgreSQL, or Elasticsearch) using the `store_reader` processor. The store_reader reads records, extracts the text field, creates `Document` objects, and feeds them into the standard build pipeline.

**Pipeline**: `store_reader → chunker → NER → relex → graph_writer`

### `build_graph_from_store()` convenience function

```python
import grapsit

result = grapsit.build_graph_from_store(
    # Relational store config
    table="articles",
    text_field="body",              # column containing the text
    id_field="article_id",          # column used as document source ID
    metadata_fields=["author", "date"],  # extra columns → Document.metadata
    relational_store_type="sqlite",
    sqlite_path="/data/articles.db",
    # Standard build config
    entity_labels=["person", "organization", "location"],
    relation_labels=["works at", "born in", "located in"],
    neo4j_uri="bolt://localhost:7687",
    neo4j_password="password",
    # Pagination (optional)
    limit=1000,                     # max records (0 = all)
    offset=0,
    filter_empty=True,              # skip records with blank/missing text
)
```

### Builder API

```python
from grapsit import BuildConfigBuilder

builder = BuildConfigBuilder(name="from_store")

# Configure relational store (SQLite, Postgres, or Elasticsearch)
builder.chunk_store(type="sqlite", sqlite_path="/data/articles.db")

# Add store_reader — pulls texts from the relational store
builder.store_reader(
    table="articles",
    text_field="body",
    id_field="article_id",
    metadata_fields=["author", "date"],
    limit=500,
)

builder.chunker(method="sentence")
builder.ner_gliner(labels=["person", "organization", "location"])
builder.relex_gliner(
    entity_labels=["person", "organization", "location"],
    relation_labels=["works at", "born in"],
)
builder.graph_writer(neo4j_uri="bolt://localhost:7687", neo4j_password="password")

executor = builder.build(verbose=True)
result = executor.execute({})  # empty input — store_reader provides texts
```

The chunker receives `texts` and `documents` from the store_reader output. When `documents` are provided, the chunker uses them directly (preserving source metadata) instead of creating new Document objects.

### With PostgreSQL

```python
builder = BuildConfigBuilder(name="from_postgres")
builder.chunk_store(
    type="postgres",
    postgres_host="localhost",
    postgres_port=5432,
    postgres_user="myuser",
    postgres_password="mypass",
    postgres_database="mydb",
)
builder.store_reader(table="documents", text_field="content")
builder.chunker(method="paragraph")
builder.ner_gliner(labels=["person", "organization"])
builder.graph_writer(neo4j_uri="bolt://localhost:7687", neo4j_password="password")

executor = builder.build()
result = executor.execute({})
```

### With Elasticsearch

```python
builder = BuildConfigBuilder(name="from_elasticsearch")
builder.chunk_store(
    type="elasticsearch",
    elasticsearch_url="http://localhost:9200",
    elasticsearch_api_key="my-api-key",
)
builder.store_reader(table="articles", text_field="text")
builder.chunker(method="sentence")
builder.ner_gliner(labels=["person", "location"])
builder.graph_writer(neo4j_uri="bolt://localhost:7687", neo4j_password="password")

executor = builder.build()
result = executor.execute({})
```

### Backward compatibility

Without `store_reader()`, the pipeline behaves exactly as before — the chunker reads from `$input.texts`. The store_reader is purely additive.

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

### Retrieval strategies

grapsit supports 9 retrieval strategies. Each strategy plugs into the same downstream pipeline (`chunk_retriever → reasoner`), so you can swap strategies without changing anything else.

```
query → parser → entities → entity lookup → subgraph → chunks     (entity — default)
query → parser → linker → entities → entity lookup → subgraph     (entity + linking)
query → embedding → community search → members → subgraph         (community)
query → embedding → chunk search → entities → subgraph            (chunk_embedding)
query → parser → entities → embedding search → subgraph           (entity_embedding)
query → LLM function calling → graph tools → subgraph             (tool)
query → parser → entities → pairs → shortest paths → subgraph     (path)
query → tool parser → triple queries → KG scorer → subgraph       (kg_scored)
query → full-text search → chunks                                 (keyword)
query → full-text search → chunks → entities → subgraph           (keyword + expand)
```

#### Strategy 1: Entity lookup (default)

Parses the query for entities, looks them up in the graph, and expands to a k-hop subgraph.

```python
# Convenience function
result = grapsit.query_graph(
    query="Where was Einstein born?",
    entity_labels=["person", "location"],
    neo4j_uri="bolt://localhost:7687",
    neo4j_password="password",
    retrieval_strategy="entity",  # default
    max_hops=2,
)

# Builder API
builder = QueryConfigBuilder(name="entity_query")
builder.query_parser(labels=["person", "location"])
builder.retriever(neo4j_uri="bolt://localhost:7687", max_hops=2)
builder.chunk_retriever()
builder.reasoner(api_key="sk-...", model="gpt-4o-mini")  # optional
executor = builder.build()
ctx = executor.execute({"query": "Where was Einstein born?"})
```

#### Strategy 2: Entity lookup with linking

Same as entity, but links parsed entities to a knowledge base first for precise lookup by stable ID.

```python
builder = QueryConfigBuilder(name="linked_query")
builder.query_parser(labels=["person", "location"])
builder.linker(executor=glinker_executor)  # or neo4j_uri= to load KB from graph
builder.retriever(neo4j_uri="bolt://localhost:7687", max_hops=2)
builder.chunk_retriever()
executor = builder.build()
```

#### Strategy 3: Community-based retrieval

Embeds the query, searches community embeddings, and retrieves subgraphs around community members. No parser needed — works directly on the query text.

**Prerequisites:** Run `grapsit.detect_communities()` with `api_key` to generate community summaries and embeddings first.

```python
# Convenience function
result = grapsit.query_graph(
    query="Tell me about the physics research group",
    retrieval_strategy="community",
    neo4j_uri="bolt://localhost:7687",
    neo4j_password="password",
    retriever_kwargs={
        "top_k": 3,                              # number of communities to match
        "max_hops": 1,                            # expansion around community members
        "embedding_method": "sentence_transformer",
        "model_name": "all-MiniLM-L6-v2",
        "vector_store_type": "in_memory",
    },
)

# Builder API
builder = QueryConfigBuilder(name="community_query")
builder.community_retriever(
    neo4j_uri="bolt://localhost:7687",
    top_k=3,
    max_hops=1,
    embedding_method="sentence_transformer",
    model_name="all-MiniLM-L6-v2",
    vector_store_type="in_memory",
)
builder.chunk_retriever()
builder.reasoner(api_key="sk-...", model="gpt-4o-mini")  # optional
executor = builder.build()
ctx = executor.execute({"query": "Tell me about the physics research group"})
```

#### Strategy 4: Chunk embedding retrieval

Embeds the query, searches pre-computed chunk embeddings, gets entities from matched chunks, and builds a subgraph. No parser needed.

**Prerequisites:** Chunk embeddings must be pre-populated in the vector store. Use `build_graph(embed_chunks=True)` or `builder.chunk_embedder()` during build time.

```python
# Convenience function
result = grapsit.query_graph(
    query="What happened in 1905?",
    retrieval_strategy="chunk_embedding",
    neo4j_uri="bolt://localhost:7687",
    neo4j_password="password",
    retriever_kwargs={
        "top_k": 5,
        "max_hops": 1,
        "vector_index_name": "chunk_embeddings",
    },
)

# Builder API
builder = QueryConfigBuilder(name="chunk_emb_query")
builder.chunk_embedding_retriever(
    neo4j_uri="bolt://localhost:7687",
    top_k=5,
    max_hops=1,
)
builder.chunk_retriever()
executor = builder.build()
```

#### Strategy 5: Entity embedding retrieval

Parses the query for entities, embeds their text, searches pre-computed entity embeddings for similar entities, and builds a subgraph.

**Prerequisites:** Entity embeddings must be pre-populated in the vector store. Use `build_graph(embed_entities=True)` or `builder.entity_embedder()` during build time.

```python
# Convenience function
result = grapsit.query_graph(
    query="Who is similar to Einstein?",
    entity_labels=["person"],
    retrieval_strategy="entity_embedding",
    neo4j_uri="bolt://localhost:7687",
    neo4j_password="password",
    retriever_kwargs={
        "top_k": 5,
        "max_hops": 2,
        "vector_index_name": "entity_embeddings",
    },
)

# Builder API
builder = QueryConfigBuilder(name="entity_emb_query")
builder.query_parser(labels=["person"])
builder.entity_embedding_retriever(
    neo4j_uri="bolt://localhost:7687",
    top_k=5,
    max_hops=2,
)
builder.chunk_retriever()
executor = builder.build()
```

#### Strategy 6: Tool-based retrieval (LLM function calling)

An LLM agent receives the query and graph schema, then uses function calling to query the graph via structured tools (`search_entity`, `get_neighbors`, `find_shortest_path`, etc.). The LLM decides which tools to call and how many rounds are needed. No parser needed.

```python
# Convenience function
result = grapsit.query_graph(
    query="What companies did Einstein work at, and where are they located?",
    api_key="sk-...",
    model="gpt-4o-mini",
    retrieval_strategy="tool",
    neo4j_uri="bolt://localhost:7687",
    neo4j_password="password",
    retriever_kwargs={
        "entity_types": ["person", "organization", "location"],
        "relation_types": ["WORKS_AT", "BORN_IN", "LOCATED_IN"],
        "max_tool_rounds": 3,  # max agentic loop iterations
    },
)

# Builder API
builder = QueryConfigBuilder(name="tool_query")
builder.tool_retriever(
    api_key="sk-...",
    model="gpt-4o-mini",
    neo4j_uri="bolt://localhost:7687",
    entity_types=["person", "organization", "location"],
    relation_types=["WORKS_AT", "BORN_IN", "LOCATED_IN"],
    max_tool_rounds=3,
)
builder.chunk_retriever()
builder.reasoner(api_key="sk-...", model="gpt-4o-mini")  # optional
executor = builder.build()
ctx = executor.execute({"query": "What companies did Einstein work at?"})
```

The tool retriever uses the same 7 built-in graph tools described in the [LLM function calling](#llm-function-calling-tool-use) section. When a relational store is configured (via `chunk_store()` or pool), the 3 [relational tools](#relational-chunkdocument-tools) (`search_chunks`, `get_chunk`, `query_records`) are also available. The LLM sees the graph schema and decides which tools to call — it does **not** generate raw Cypher.

#### Strategy 7: Path-based retrieval

Parses the query for entities, looks them up, generates entity pairs, and finds shortest paths between each pair.

```python
# Convenience function
result = grapsit.query_graph(
    query="How are Einstein and Bohr connected?",
    entity_labels=["person"],
    retrieval_strategy="path",
    neo4j_uri="bolt://localhost:7687",
    neo4j_password="password",
    retriever_kwargs={
        "max_path_length": 5,
        "max_pairs": 10,
    },
)

# Builder API
builder = QueryConfigBuilder(name="path_query")
builder.query_parser(labels=["person"])
builder.path_retriever(
    neo4j_uri="bolt://localhost:7687",
    max_path_length=5,
    max_pairs=10,
)
builder.chunk_retriever()
executor = builder.build()
ctx = executor.execute({"query": "How are Einstein and Bohr connected?"})
```

#### Strategy 8: KG-scored retrieval (tool parser + KG embeddings)

Uses an LLM tool-calling parser to decompose the query into structured triple patterns (`head?, relation?, tail?`), then resolves those against the graph store and scores them with trained KG embeddings. The KG scorer acts as a universal retriever — no separate retriever node is needed.

**Prerequisites:** A trained KG embedding model (optional but recommended — without it, graph store scores are used). Use `grapsit.train_kg_model()` to train one.

```python
# Convenience function
result = grapsit.query_graph(
    query="Where was Einstein born?",
    api_key="sk-...",
    model="gpt-4o-mini",
    retrieval_strategy="kg_scored",
    neo4j_uri="bolt://localhost:7687",
    neo4j_password="password",
    entity_labels=["person", "location"],  # optional, for parser context
    retriever_kwargs={
        "relation_labels": ["born_in", "works_at"],  # for parser context
        "model_path": "kg_model",           # trained KGE model directory
        "top_k": 10,
        "predict_tails": True,
        "score_threshold": 0.5,
    },
)

# Builder API
builder = QueryConfigBuilder(name="kg_scored_query")
builder.query_parser(
    method="tool",
    api_key="sk-...",
    model="gpt-4o-mini",
    labels=["person", "location"],
    relation_labels=["born_in", "works_at"],
)
builder.kg_scorer(
    model_path="kg_model",
    top_k=10,
    predict_tails=True,
    score_threshold=0.5,
    device="cpu",
    neo4j_uri="bolt://localhost:7687",
    neo4j_password="password",
)
builder.chunk_retriever(chunk_entity_source="both")  # "all", "head", "tail", or "both"
builder.reasoner(api_key="sk-...", model="gpt-4o-mini")  # optional
executor = builder.build()
ctx = executor.execute({"query": "Where was Einstein born?"})

# Access results
scorer_result = ctx.get("kg_scorer_result")
print(scorer_result["scored_triples"])   # matched triples with scores
print(scorer_result["predictions"])       # predicted missing links
print(scorer_result["subgraph"])          # built subgraph
```

The tool-calling parser sends `search_triples(head, relation, tail)` calls to decompose the query. For example, "Where was Einstein born?" becomes `search_triples(head="Einstein", relation="born_in", tail=null)`. The KG scorer then:

1. Looks up the head/tail entities in the graph store
2. Finds matching relations (filtered by relation type if specified)
3. Scores candidate triples with the KGE model (if available)
4. Builds a Subgraph from scored results
5. Optionally predicts missing links for matched entities

The `chunk_entity_source` parameter on `chunk_retriever()` controls which entities' chunks are fetched:
- `"all"` (default) — all entities in the subgraph
- `"head"` — only head entities from scored triples
- `"tail"` — only tail entities from scored triples
- `"both"` — explicit alias for `"all"`

#### Strategy 9: Keyword retrieval (full-text search)

Searches chunks in a relational store (SQLite FTS5, PostgreSQL tsvector, or Elasticsearch) using full-text search. No parser or embeddings needed — works directly on the query text.

Two modes:
- **Chunks-only** (default) — returns matched chunks directly. Only needs a relational store — no graph database required.
- **Entity expansion** (`expand_entities=True`) — additionally looks up entities mentioned in matched chunks via the graph store and builds a full subgraph.

**Prerequisites:** Chunks must be stored in a relational store (SQLite, PostgreSQL, or Elasticsearch).

```python
# Chunks-only (default) — no graph store needed
builder = QueryConfigBuilder(name="keyword_query")
builder.keyword_retriever(
    top_k=10,
    chunk_table="chunks",
    relational_store_type="sqlite",
    sqlite_path="chunks.db",
)
builder.reasoner(api_key="sk-...", model="gpt-4o-mini")  # optional
executor = builder.build()
ctx = executor.execute({"query": "Where was Einstein born?"})
```

```python
# With entity expansion — also builds a subgraph from matched chunks
builder = QueryConfigBuilder(name="keyword_expanded")
builder.keyword_retriever(
    top_k=10,
    expand_entities=True,
    max_hops=1,
    relational_store_type="sqlite",
    sqlite_path="chunks.db",
    neo4j_uri="bolt://localhost:7687",
)
builder.chunk_retriever()
builder.reasoner(api_key="sk-...", model="gpt-4o-mini")  # optional
executor = builder.build()
```

#### Strategy comparison

| Strategy | Needs parser? | Needs embeddings? | Needs LLM? | Best for |
|----------|:---:|:---:|:---:|----------|
| entity (default) | yes | no | no | Direct entity lookup |
| entity + linking | yes | no | no | Precise lookup with KB IDs |
| community | no | yes (community) | no | Topic/cluster-based queries |
| chunk_embedding | no | yes (chunk) | no | Semantic similarity search |
| entity_embedding | yes | yes (entity) | no | Finding similar entities |
| tool | no | no | yes | Complex multi-hop questions |
| path | yes | no | no | Relationship discovery |
| kg_scored | yes (tool) | optional (KGE) | yes | Structured triple matching + link prediction |
| keyword | no | no | no | Full-text keyword search over chunks |

### Direct store queries (Neo4j)

After building, use `Neo4jGraphStore` to query directly (see [FalkorDB section](#direct-falkordb-queries) for FalkorDB, [Memgraph section](#direct-memgraph-queries) for Memgraph):

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

## LLM function calling (tool use)

grapsit includes a tool-calling layer that lets an LLM query the knowledge graph via structured function calls. The LLM receives the graph schema (entity types, relation types, property keys) as context, produces structured tool calls, and each call is translated into a parameterised Cypher query.

### Built-in tools

Seven graph query tools are provided out of the box:

| Tool | Description |
|------|------------|
| `search_entity` | Find an entity by label, optionally filtered by `entity_type` |
| `list_entities` | List entities with filters on `entity_type` and arbitrary properties |
| `get_entity_relations` | Get relations for an entity, filter by `relation_type`, `target_entity_type`, `min_score`, and property filters |
| `get_neighbors` | K-hop neighbor traversal, filter by `entity_type`, `relation_type`, and property filters |
| `get_subgraph` | Retrieve a subgraph around a set of entities |
| `get_chunks_for_entity` | Get source text chunks where an entity is mentioned |
| `find_shortest_path` | Shortest path between two entities, optionally restricted to a relation type |

Tools that accept filters support a `filters` array for arbitrary property conditions:

```python
# "List all organizations in Cambridge founded after 2020"
# The LLM would call list_entities with:
{
    "entity_type": "organization",
    "filters": [
        {"property": "location", "operator": "eq", "value": "Cambridge"},
        {"property": "founded_year", "operator": "gte", "value": 2020},
    ],
    "limit": 20,
}
```

Supported filter operators: `eq`, `neq`, `gt`, `gte`, `lt`, `lte`, `contains`, `starts_with`.

### Relational (chunk/document) tools

When a relational store (SQLite, PostgreSQL, or Elasticsearch) is configured, three additional tools become available for querying stored text chunks and documents:

| Tool | Description |
|------|------------|
| `search_chunks` | Full-text search over stored text chunks. Returns the most relevant chunks matching a query string. |
| `get_chunk` | Retrieve a single text chunk by its ID. |
| `query_records` | Structured query with filtering, sorting, and pagination. Supports arbitrary field filters (e.g. `document_id = "doc_001"`, `index >= 5`). |

These tools use the same filter syntax as graph tools:

```python
# "Get all chunks from document doc_001 sorted by index"
# The LLM would call query_records with:
{
    "table": "chunks",
    "filters": [
        {"field": "document_id", "operator": "eq", "value": "doc_001"},
    ],
    "sort_by": "index",
    "sort_order": "asc",
    "limit": 50,
}
```

#### Using relational tools with the tool retriever

The tool retriever automatically includes relational tools when a relational store is available (via the store pool or direct config). The LLM agent can then combine graph queries and chunk searches in a single session:

```python
from grapsit import QueryConfigBuilder

builder = QueryConfigBuilder(name="tool_with_chunks")
builder.chunk_store(type="sqlite", sqlite_path="chunks.db")
builder.tool_retriever(
    api_key="sk-...",
    model="gpt-4o-mini",
    neo4j_uri="bolt://localhost:7687",
    entity_types=["person", "organization", "location"],
    relation_types=["WORKS_AT", "BORN_IN"],
    max_tool_rounds=5,
)
builder.reasoner(api_key="sk-...", model="gpt-4o-mini")
executor = builder.build()
ctx = executor.execute({"query": "What did Einstein publish about relativity?"})
```

The agent can call `search_entity` to find Einstein in the graph, `get_entity_relations` to find related concepts, and `search_chunks` to find source passages mentioning relativity — all in one agentic loop.

#### Using relational tools directly

You can also dispatch relational tool calls manually:

```python
from grapsit.llm.tools import RELATIONAL_TOOLS, execute_relational_tool
from grapsit.store.relational.sqlite_store import SqliteRelationalStore

store = SqliteRelationalStore(path="chunks.db")

# Full-text search
results = execute_relational_tool("search_chunks", {"query": "relativity", "top_k": 5}, store)

# Get a specific chunk
results = execute_relational_tool("get_chunk", {"chunk_id": "chunk-001"}, store)

# Structured query with filters
results = execute_relational_tool("query_records", {
    "table": "chunks",
    "filters": [{"field": "document_id", "operator": "eq", "value": "doc_001"}],
    "sort_by": "index",
    "limit": 20,
}, store)
```

`RELATIONAL_TOOLS` contains the OpenAI function-calling tool definitions, so you can pass them to any LLM alongside `GRAPH_TOOLS`:

```python
from grapsit.llm.tools import GRAPH_TOOLS, RELATIONAL_TOOLS

all_tools = GRAPH_TOOLS + RELATIONAL_TOOLS
result = client.complete_with_tools(messages=[...], tools=all_tools)
```

### Using `complete_with_tools()`

```python
from grapsit.llm.openai_client import OpenAIClient
from grapsit.llm.tools import build_graph_schema_prompt, tool_call_to_cypher, GRAPH_TOOLS

client = OpenAIClient(api_key="sk-...", model="gpt-4o-mini")

# Build a schema prompt so the LLM knows what's in the graph
schema_prompt = build_graph_schema_prompt(
    entity_types=["person", "organization", "location"],
    relation_types=["WORKS_AT", "BORN_IN", "COLLABORATED_WITH"],
    property_keys={"organization": ["founded_year", "location", "revenue"]},
)

# Send user query with graph tools
result = client.complete_with_tools(
    messages=[
        {"role": "system", "content": schema_prompt},
        {"role": "user", "content": "Find companies in Cambridge founded after 2020"},
    ],
    # tools=GRAPH_TOOLS is the default; pass custom tools or GRAPH_TOOLS + custom
)

# Translate each tool call to Cypher and execute
for tc in result["tool_calls"]:
    cypher, params = tool_call_to_cypher(tc["name"], tc["arguments"])
    print(f"Cypher: {cypher}")
    print(f"Params: {params}")
    # Execute against your graph store...
```

### Cypher translation

`tool_call_to_cypher()` converts each tool call into a parameterised Cypher query:

```python
from grapsit.llm.tools import tool_call_to_cypher

# search_entity
cypher, params = tool_call_to_cypher("search_entity", {"label": "Einstein"})
# -> "MATCH (e:Entity) WHERE toLower(e.label) = toLower($label) RETURN e"
# -> {"label": "Einstein"}

# list_entities with property filters
cypher, params = tool_call_to_cypher("list_entities", {
    "entity_type": "organization",
    "filters": [
        {"property": "location", "operator": "eq", "value": "Cambridge"},
        {"property": "founded_year", "operator": "gte", "value": 2020},
    ],
})
# -> "MATCH (e:Entity) WHERE toLower(e.entity_type) = toLower($entity_type)
#     AND e.location = $f_0_location AND e.founded_year >= $f_1_founded_year
#     RETURN e LIMIT $limit"

# find_shortest_path with relation filter
cypher, params = tool_call_to_cypher("find_shortest_path", {
    "source_entity_id": "id-a",
    "target_entity_id": "id-b",
    "relation_type": "COLLABORATED_WITH",
    "max_depth": 3,
})
# -> "MATCH (src:Entity), (tgt:Entity) WHERE src.id = $source_id AND tgt.id = $target_id
#     MATCH p = shortestPath((src)-[:`COLLABORATED_WITH`*..3]-(tgt)) RETURN p"
```

### Custom tools

Add your own tools by defining a tool schema and registering a Cypher translator:

```python
from grapsit.llm.tools import GRAPH_TOOLS, register_tool_translator

# Define a custom tool
my_tool = {
    "type": "function",
    "function": {
        "name": "count_by_type",
        "description": "Count entities grouped by entity_type.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

# Register a Cypher translator for it
register_tool_translator("count_by_type", lambda args: (
    "MATCH (e:Entity) RETURN e.entity_type AS type, count(*) AS count ORDER BY count DESC",
    {},
))

# Use combined tools
result = client.complete_with_tools(
    messages=[...],
    tools=GRAPH_TOOLS + [my_tool],
)
```

## Community detection

After building a knowledge graph, you can detect communities of densely connected entities, optionally generate LLM summaries for each community, and embed those summaries into a vector store.

**Community pipeline**: `detector → summarizer (optional) → embedder (optional)`

### `detect_communities()` convenience function

```python
import grapsit

# Detection only (no LLM needed)
result = grapsit.detect_communities(
    neo4j_uri="bolt://localhost:7687",
    neo4j_password="password",
    method="louvain",    # "louvain" or "leiden"
    levels=1,            # hierarchical levels (1 = flat)
)

detector_result = result.get("detector_result")
print(f"Found {detector_result['community_count']} communities across {detector_result['levels']} level(s)")
# detector_result["communities"] is a dict: {level: {entity_id: community_id}}

# Detection + LLM summarization + embedding (provide api_key to enable)
result = grapsit.detect_communities(
    neo4j_uri="bolt://localhost:7687",
    neo4j_password="password",
    method="louvain",
    levels=2,
    resolution=1.0,
    # Summarizer (enabled when api_key is provided)
    api_key="sk-...",
    model="gpt-4o-mini",
    top_k=10,                # max entities per community for LLM context
    # Embedder (enabled alongside summarizer)
    embedding_method="sentence_transformer",  # or "openai"
    model_name="all-MiniLM-L6-v2",
    vector_store_type="in_memory",            # or "faiss", "qdrant"
)

summaries = result.get("summarizer_result")["summaries"]
for comm_id, s in summaries.items():
    print(f"  {s['title']}: {s['summary']}")
```

### Community builder API

For full control over the pipeline:

```python
from grapsit import CommunityConfigBuilder

builder = CommunityConfigBuilder(name="my_communities")

# Step 1: Detector (required)
builder.detector(
    method="louvain",       # "louvain" or "leiden"
    levels=2,               # multi-level hierarchical detection
    resolution=1.0,
    store_type="neo4j",     # or "falkordb", "memgraph"
    neo4j_uri="bolt://localhost:7687",
    neo4j_password="password",
)

# Step 2: Summarizer (optional — generates title + summary per community)
builder.summarizer(
    api_key="sk-...",
    model="gpt-4o-mini",
    top_k=10,               # top entities by degree used as LLM context
)

# Step 3: Embedder (optional — embeds summaries into a vector store)
builder.embedder(
    embedding_method="sentence_transformer",
    model_name="all-MiniLM-L6-v2",
    vector_store_type="in_memory",
)

executor = builder.build(verbose=True)
result = executor.execute({})

# Save config for reproducibility
builder.save("configs/community.yaml")
```

Store parameters (URI, credentials, etc.) set on the detector are automatically inherited by the summarizer and embedder — no need to repeat them.

### How it works

**Level 0** — uses the graph database's native community detection algorithm:

| Database | Algorithm | Library |
|----------|-----------|---------|
| Neo4j | Louvain or Leiden | [GDS](https://neo4j.com/docs/graph-data-science/current/) (Graph Data Science) |
| Memgraph | Louvain | [MAGE](https://memgraph.com/docs/advanced-algorithms/) |
| FalkorDB | Label Propagation (CDLP) | [Built-in](https://docs.falkordb.com/algorithms/cdlp.html) |

**Levels 1+** — builds a weighted meta-graph of inter-community edges and runs NetworkX `louvain_communities()` on it. This provides portable hierarchical detection regardless of the backend. Requires `pip install networkx`.

**Summarizer** — for each community, fetches the top-k member entities by degree centrality, builds a context string with their labels and relations, and asks the LLM to produce a title and summary.

**Embedder** — encodes each community's `"title. summary"` text using sentence-transformers (or OpenAI embeddings), stores vectors in the configured vector store, and writes them back to `Community` nodes in the graph.

### Community graph schema

The community pipeline adds these nodes and relationships:

```
(:Community {id, level, title, summary, embedding})

(entity)-[:MEMBER_OF {level}]->(community)
(child_community)-[:CHILD_OF]->(parent_community)
```

### With different graph backends

```python
# FalkorDB
result = grapsit.detect_communities(
    store_type="falkordb",
    falkordb_host="localhost",
    falkordb_port=6379,
    falkordb_graph="my_knowledge_graph",
)

# Memgraph
result = grapsit.detect_communities(
    store_type="memgraph",
    memgraph_uri="bolt://localhost:7687",
)
```

## KG embedding modeling (PyKEEN)

After building a knowledge graph, you can train knowledge graph embeddings using [PyKEEN](https://github.com/pykeen/pykeen). This enables learning entity and relation representations, storing them for retrieval, and scoring/predicting triples at query time.

**Training pipeline**: `kg_triple_reader → kg_trainer → kg_embedding_storer`

**Query-time scoring**: `kg_scorer` (optional node in the query pipeline)

### `train_kg_model()` convenience function

```python
import grapsit

result = grapsit.train_kg_model(
    # Triple source
    source="graph_store",           # "graph_store" or "tsv"
    # tsv_path="triples.tsv",      # if source="tsv"
    # Model config
    model="RotatE",                 # PyKEEN model: RotatE, TransE, ComplEx, etc.
    embedding_dim=128,
    epochs=100,
    batch_size=256,
    lr=0.001,
    device="cpu",                   # or "cuda"
    # Split ratios
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    # Storage
    model_path="kg_model",          # directory for weights + mappings
    vector_store_type="in_memory",  # or "faiss", "qdrant"
    store_to_graph=False,           # write entity embeddings to graph DB nodes
    # Graph store
    neo4j_uri="bolt://localhost:7687",
    neo4j_password="password",
)

# Access results
reader = result.get("reader_result")
print(f"Loaded {reader['triple_count']} triples")

trainer = result.get("trainer_result")
print(f"Metrics: {trainer['metrics']}")

storer = result.get("storer_result")
print(f"Entity embeddings: {storer['entity_embeddings_shape']}")
print(f"Relation embeddings: {storer['relation_embeddings_shape']}")
print(f"Model saved to: {storer['model_path']}")
```

### KG modeling builder API

```python
from grapsit import KGModelingConfigBuilder

builder = KGModelingConfigBuilder(name="my_kg_model")

# Step 1: Read triples from the graph
builder.triple_reader(
    source="graph_store",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    neo4j_uri="bolt://localhost:7687",
    neo4j_password="password",
)

# Step 2: Train the model
builder.trainer(
    model="RotatE",
    embedding_dim=128,
    epochs=100,
    batch_size=256,
    lr=0.001,
    device="cpu",
)

# Step 3: Store embeddings (optional)
builder.storer(
    model_path="kg_model",
    vector_store_type="in_memory",
    store_to_graph=True,            # write embeddings to Entity nodes
)

executor = builder.build(verbose=True)
result = executor.execute({})

# Save config for reproducibility
builder.save("configs/kg_modeling.yaml")
```

Store parameters set on the triple reader are automatically inherited by the storer.

### Query-time link prediction

Add a `kg_scorer` node to any query pipeline to score existing triples and predict missing links:

```python
from grapsit import QueryConfigBuilder

builder = QueryConfigBuilder(name="scored_query")
builder.query_parser(labels=["person", "location"])
builder.retriever(neo4j_uri="bolt://localhost:7687", max_hops=2)
builder.chunk_retriever()

# Add KG scoring — loads trained model from disk
builder.kg_scorer(
    model_path="kg_model",          # directory with saved model
    top_k=10,                       # top predictions per entity
    predict_tails=True,             # predict (entity, relation, ?)
    predict_heads=False,            # predict (?, relation, entity)
    score_threshold=None,           # optional minimum score filter
    device="cpu",
)

builder.reasoner(api_key="sk-...", model="gpt-4o-mini")  # optional
executor = builder.build()
ctx = executor.execute({"query": "Where was Einstein born?"})

# Access scoring results
scorer_result = ctx.get("kg_scorer_result")
print(scorer_result["scored_triples"])   # existing triples with KGE scores
print(scorer_result["predictions"])       # predicted missing links
# scorer_result["subgraph"] is enriched with predicted relations
```

The KG scorer can also act as a **universal retriever** using the `kg_scored` strategy — see [Strategy 8: KG-scored retrieval](#strategy-8-kg-scored-retrieval-tool-parser--kg-embeddings) in the query strategies section.

### How it works

- **`kg_triple_reader`** — reads all `(head_label, relation_type, tail_label)` triples from the graph store (or a TSV file), creates a PyKEEN `TriplesFactory`, and splits into train/validation/test sets.
- **`kg_trainer`** — trains a PyKEEN model (RotatE, TransE, ComplEx, DistMult, etc.) via `pykeen.pipeline.pipeline()`. Returns the trained model and evaluation metrics.
- **`kg_embedding_storer`** — extracts entity and relation embeddings from the trained model, stores them in the vector store (separate indexes), saves model weights and ID mappings to disk, and optionally writes entity embeddings to graph DB nodes.
- **`kg_scorer`** — at query time, scores existing triples in the retrieved subgraph using `model.score_hrt()`, and predicts missing links for query entities (top-k tail/head predictions). Predictions are added to the subgraph as additional relations. In `kg_scored` mode, the scorer also acts as a retriever: it resolves `triple_queries` from the tool-calling parser against the graph store, building a scored subgraph without needing a separate retriever.

### YAML config

See `configs/kg_modeling.yaml` for a complete training pipeline config. Load it with:

```python
executor = grapsit.ProcessorFactory.create_pipeline("configs/kg_modeling.yaml")
result = executor.execute({})
```

## Store pool (shared connections)

By default, each processor in a pipeline creates its own store connection. The **store pool** manages named, lazily-created, shared store instances — so a pipeline with `graph_writer` + `chunk_embedder` + `entity_embedder` shares a single database connection instead of creating three.

### Automatic connection sharing

When using the builder API, all processors automatically share connections — no extra configuration needed:

```python
from grapsit import BuildConfigBuilder

builder = BuildConfigBuilder(name="my_pipeline")
builder.chunker(method="sentence")
builder.ner_gliner(labels=["person", "organization", "location"])
builder.graph_writer(neo4j_uri="bolt://localhost:7687", neo4j_password="password")
builder.chunk_embedder(embedding_method="sentence_transformer", model_name="all-MiniLM-L6-v2")
builder.entity_embedder(embedding_method="sentence_transformer", model_name="all-MiniLM-L6-v2")

# graph_writer, chunk_embedder, and entity_embedder all share the same Neo4j connection
executor = builder.build(verbose=True)
result = executor.execute({"texts": ["Einstein was born in Ulm."]})
```

### Context manager

`DAGExecutor` supports the context manager protocol to automatically close all pooled connections:

```python
with builder.build() as executor:
    result = executor.execute({"texts": ["Einstein was born in Ulm."]})
# All store connections are closed here
```

### Named stores

Register multiple stores by name and reference them from different pipeline nodes:

```python
from grapsit import BuildConfigBuilder, Neo4jConfig, FaissVectorConfig

builder = BuildConfigBuilder(name="multi_store")
builder.chunker(method="sentence")
builder.ner_gliner(labels=["person", "location"])

# Register named stores
builder.graph_store(Neo4jConfig(uri="bolt://localhost:7687", password="password"), name="main")
builder.vector_store(FaissVectorConfig(use_gpu=True), name="embeddings")

builder.graph_writer()
builder.chunk_embedder()

with builder.build() as executor:
    result = executor.execute({"texts": ["Einstein was born in Ulm."]})
```

### Typed store configs

Graph and vector stores can be configured with typed Pydantic config objects:

**Graph store configs:**

```python
from grapsit import Neo4jConfig, FalkorDBConfig, MemgraphConfig

neo4j = Neo4jConfig(uri="bolt://localhost:7687", password="password")
falkor = FalkorDBConfig(host="localhost", port=6379, graph="my_graph")
memgraph = MemgraphConfig(uri="bolt://localhost:7687")
```

**Vector store configs:**

```python
from grapsit import InMemoryVectorConfig, FaissVectorConfig, QdrantVectorConfig, GraphDBVectorConfig

in_mem = InMemoryVectorConfig()
faiss = FaissVectorConfig(use_gpu=True)
qdrant = QdrantVectorConfig(url="http://localhost:6333")
graph_db = GraphDBVectorConfig(graph_store_name="main")  # reuses a named graph store
```

### StorePool directly

For advanced use cases, create and manage a `StorePool` directly:

```python
from grapsit import StorePool

pool = StorePool()
pool.register_graph("main", {"store_type": "neo4j", "neo4j_uri": "bolt://localhost:7687"})
pool.register_vector("embeddings", {"vector_store_type": "faiss", "use_gpu": True})

# Lazy creation — connection is opened on first access
store = pool.get_graph("main")      # creates Neo4j connection
store2 = pool.get_graph("main")     # returns the same instance
vector = pool.get_vector("embeddings")

pool.close()  # closes all instantiated connections
```

### YAML config with stores

The `stores` section in YAML configs defines named stores shared across all nodes:

```yaml
name: my_pipeline
stores:
  graph:
    default:
      store_type: neo4j
      neo4j_uri: bolt://localhost:7687
      neo4j_password: password
  vector:
    default:
      vector_store_type: faiss
      use_gpu: true

nodes:
  - id: graph_writer
    processor: graph_writer
    config:
      graph_store_name: default    # references the named graph store
  # ...
```

### Backward compatibility

All existing configs and calling patterns continue to work unchanged:
- Configs without a `stores` section — processors fall back to creating their own connections
- Flat store parameters in processor configs — work as before
- `create_store()` and `create_vector_store()` factory functions — still work standalone

## Accessing intermediate results

`build_graph()` and `executor.execute()` return a `PipeContext` containing every pipeline stage output:

```python
result = grapsit.build_graph(texts=..., entity_labels=...)

# Texts pulled from a relational store (if store_reader was used)
# store_reader_result = result.get("store_reader_result")  # {"texts", "documents", "source_records"}

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

# Embedding stats (if embedders were enabled)
chunk_emb = result.get("chunk_embedder_result")   # {"embedded_count", "dimension", "index_name"}
entity_emb = result.get("entity_embedder_result")  # {"embedded_count", "dimension", "index_name"}
```

## Graph schema

The graph writer creates this schema (same for Neo4j, FalkorDB, and Memgraph):

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
$input (texts)   OR   relational store (SQLite/Postgres/ES)
    │                          │
    │                   store_reader (optional)
    │                          │
    └──────────┬───────────────┘
               ▼
            chunker ──────────────────┐
               │                      │
               ▼                      │
          ner_gliner/ner_llm          │
               │                      │
               ▼                      │
          entity_linker (optional)    │
               │                      ▼
               │            relex_gliner/relex_llm
               │                      │
               └──────┐   ┌───────────┘
                      ▼   ▼
                  graph_writer ──────────────────┐
                      │                          │
                      ▼                          ▼
             Neo4j / FalkorDB / Memgraph
                      │                          │
                      ▼                          ▼
             chunk_embedder (optional)   entity_embedder (optional)
                      │                          │
                      ▼                          ▼
                vector store              vector store
```

**KG modeling pipeline:**

```
$input {}
    │
    ▼
 kg_triple_reader (read from graph store or TSV)
    │
    ▼
 kg_trainer (PyKEEN model training)
    │
    ▼
 kg_embedding_storer (vector store + disk + optional graph DB)
```

**Query pipeline (standard):**

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
 retriever (graph DB lookup + k-hop expansion)
    │
    ├──────────────────────────┐
    ▼                          ▼
 chunk_retriever       kg_scorer (optional,
    │                  scores + predicts links)
    ▼
 reasoner (optional, LLM multi-hop reasoning)
    │
    ▼
 QueryResult (answer + subgraph + chunks)
```

**Query pipeline (kg_scored — tool parser + KG scorer as retriever):**

```
$input (query)
    │
    ▼
 query_parser (method="tool", LLM tool calling)
    │  → triple_queries: [{head, relation, tail}, ...]
    ▼
 kg_scorer (graph store lookup + KGE scoring)
    │  → scored_triples + subgraph
    ▼
 chunk_retriever (with optional entity filtering)
    │
    ▼
 reasoner (optional)
    │
    ▼
 QueryResult
```

Each node specifies:
- **processor** — registered processing function (e.g. `"chunker"`, `"ner_gliner"`)
- **inputs** — where to read data from (other node outputs or `$input`)
- **output** — key to store results under
- **requires** — explicit execution ordering
- **config** — processor-specific parameters

Nodes at the same level run sequentially within a level, but levels are topologically sorted so dependencies are always satisfied.

When a `StorePool` is attached to the executor (via builders or YAML `stores` section), all processors in the pipeline share the same store connections. The executor supports context manager usage (`with builder.build() as executor:`) for automatic cleanup.

## Available processors

| Processor | Description | Key config |
|-----------|-------------|------------|
| `store_reader` | Pull texts from a relational store | `table`, `text_field`, `id_field`, `metadata_fields`, `limit`, `offset`, `filter_empty`, relational store params |
| `chunker` | Split text into chunks | `method` (sentence/paragraph/fixed), `chunk_size`, `overlap` |
| `ner_gliner` | Entity extraction with GLiNER | `model`, `labels`, `threshold`, `device` |
| `ner_llm` | Entity extraction with LLM | `model`, `labels`, `api_key`, `base_url`, `temperature` |
| `entity_linker` | Entity linking with GLinker | `executor`, `model`, `threshold`, `entities` |
| `relex_gliner` | Relation extraction with GLiNER-relex | `model`, `entity_labels`, `relation_labels`, `threshold`, `relation_threshold` |
| `relex_llm` | Relation extraction with LLM | `model`, `entity_labels`, `relation_labels`, `api_key`, `base_url`, `temperature` |
| `data_ingest` | Convert flat JSON to graph_writer format | (used internally by `IngestConfigBuilder`) |
| `graph_writer` | Deduplicate and write to graph store | `store_type`, `neo4j_uri`/`falkordb_host`/`memgraph_uri`, `json_output`, `graph_store_name`, … |
| `chunk_embedder` | Embed chunk texts into vector store | `embedding_method`, `model_name`, `vector_store_type`, `vector_index_name`, store params |
| `entity_embedder` | Embed entity labels into vector store | `embedding_method`, `model_name`, `vector_store_type`, `vector_index_name`, store params |
| `query_parser` | Extract entities from a query | `method` (gliner/llm/tool), `labels`, `relation_labels`, `model`, `api_key` |
| `retriever` | Look up entities + k-hop subgraph | `store_type`, `neo4j_uri`/`falkordb_host`/`memgraph_uri`, `max_hops` |
| `chunk_retriever` | Fetch source chunks for entities | `store_type`, `neo4j_uri`/`falkordb_host`/`memgraph_uri`, `max_chunks`, `chunk_entity_source` (all/head/tail/both) |
| `reasoner` | LLM multi-hop reasoning | `method` (llm), `model`, `api_key`, `base_url` |
| `community_detector` | Community detection (Louvain/Leiden/CDLP) | `method`, `levels`, `resolution`, `store_type`, store params |
| `community_summarizer` | LLM summaries for communities | `api_key`, `model`, `top_k`, `temperature`, store params |
| `community_embedder` | Embed community summaries | `embedding_method`, `model_name`, `vector_store_type`, store params |
| `kg_triple_reader` | Read triples from graph store or TSV | `source` (graph_store/tsv), `tsv_path`, `train_ratio`, `val_ratio`, `test_ratio`, store params |
| `kg_trainer` | Train PyKEEN KG embedding model | `model` (RotatE/TransE/…), `embedding_dim`, `epochs`, `batch_size`, `lr`, `device` |
| `kg_embedding_storer` | Store trained KG embeddings | `model_path`, `entity_index_name`, `relation_index_name`, `vector_store_type`, `store_to_graph` |
| `kg_scorer` | Score triples, predict links, resolve triple queries | `model_path`, `top_k`, `score_threshold`, `predict_tails`, `predict_heads`, `device`, store params (for triple_queries mode) |

## Adding a custom graph store

You can add your own graph database backend by implementing `BaseGraphStore` and registering it. After registration, your store works everywhere — builders, YAML configs, convenience functions, and the store pool.

### Step 1: Implement `BaseGraphStore`

```python
# my_store.py
from grapsit.store.graph.base import BaseGraphStore


class TigerGraphStore(BaseGraphStore):
    """Custom graph store backed by TigerGraph."""

    def __init__(self, host="localhost", port=9000, graph="MyGraph", token=None):
        self._host = host
        self._port = port
        self._graph = graph
        self._token = token
        self._conn = None

    def _ensure_connection(self):
        if self._conn is None:
            import pyTigerGraph as tg
            self._conn = tg.TigerGraphConnection(
                host=self._host, graphname=self._graph,
                apiToken=self._token,
            )

    def setup_indexes(self):
        self._ensure_connection()
        # Create schema/indexes as needed
        ...

    def close(self):
        self._conn = None

    def write_entity(self, entity):
        self._ensure_connection()
        self._conn.upsertVertex("Entity", entity.id, {
            "label": entity.label,
            "entity_type": entity.entity_type,
        })

    def write_relation(self, relation, head_entity_id, tail_entity_id):
        self._ensure_connection()
        rel_type = relation.relation_type.upper().replace(" ", "_")
        self._conn.upsertEdge("Entity", head_entity_id, rel_type,
                              "Entity", tail_entity_id,
                              {"score": relation.score})

    def get_entity_by_label(self, label):
        self._ensure_connection()
        # Query TigerGraph for entity by label
        ...

    def get_entity_by_id(self, entity_id):
        self._ensure_connection()
        ...

    def get_entity_neighbors(self, entity_id, max_hops=1):
        self._ensure_connection()
        ...

    def get_entity_relations(self, entity_id):
        self._ensure_connection()
        ...

    def get_chunks_for_entity(self, entity_id):
        self._ensure_connection()
        ...

    def get_subgraph(self, entity_ids, max_hops=1):
        self._ensure_connection()
        ...

    # ... implement remaining abstract methods (write_document, write_chunk, etc.)
```

`BaseGraphStore` has ~20 abstract methods. For a minimal working backend, the critical ones are: `setup_indexes`, `close`, `write_entity`, `write_relation`, `get_entity_by_label`, `get_entity_by_id`, `get_entity_neighbors`, `get_entity_relations`, `get_chunks_for_entity`, and `get_subgraph`. See `src/grapsit/store/graph/neo4j_store.py` for a complete reference implementation.

### Step 2: Register the store

```python
import grapsit

def create_tigergraph(config):
    from my_store import TigerGraphStore
    return TigerGraphStore(
        host=config.get("tigergraph_host", "localhost"),
        port=config.get("tigergraph_port", 9000),
        graph=config.get("tigergraph_graph", "MyGraph"),
        token=config.get("tigergraph_token"),
    )

grapsit.register_graph_store("tigergraph", create_tigergraph)
```

Or use the decorator form on the registry directly:

```python
from grapsit.store.graph import graph_store_registry

@graph_store_registry.register("tigergraph")
def create_tigergraph(config):
    from my_store import TigerGraphStore
    return TigerGraphStore(
        host=config.get("tigergraph_host", "localhost"),
        port=config.get("tigergraph_port", 9000),
        graph=config.get("tigergraph_graph", "MyGraph"),
        token=config.get("tigergraph_token"),
    )
```

### Step 3: Use it

Once registered, `store_type="tigergraph"` works across all APIs:

```python
# Convenience function
result = grapsit.build_graph(
    texts=["Einstein was born in Ulm."],
    entity_labels=["person", "location"],
    relation_labels=["born in"],
    store_type="tigergraph",
    tigergraph_host="localhost",
    tigergraph_graph="KnowledgeGraph",
    tigergraph_token="my_token",
)

# Builder API
builder = grapsit.BuildConfigBuilder(name="tigergraph_pipeline")
builder.chunker(method="sentence")
builder.ner_gliner(labels=["person", "location"])
builder.graph_writer(
    store_type="tigergraph",
    tigergraph_host="localhost",
    tigergraph_graph="KnowledgeGraph",
)
executor = builder.build()
result = executor.execute({"texts": ["Einstein was born in Ulm."]})

# Query pipeline
result = grapsit.query_graph(
    query="Where was Einstein born?",
    entity_labels=["person", "location"],
    store_type="tigergraph",
    tigergraph_host="localhost",
    tigergraph_graph="KnowledgeGraph",
)

# Store pool (shared connection)
builder.graph_store({"store_type": "tigergraph", "tigergraph_host": "localhost"}, name="main")
with builder.build() as executor:
    result = executor.execute({"texts": [...]})
```

YAML configs work too:

```yaml
nodes:
  - id: graph_writer
    processor: graph_writer
    config:
      store_type: tigergraph
      tigergraph_host: localhost
      tigergraph_graph: KnowledgeGraph
```

### Adding a custom vector store

The same pattern applies to vector stores:

```python
import grapsit
from grapsit.store.vector.base import BaseVectorStore


class PineconeVectorStore(BaseVectorStore):
    def __init__(self, api_key, index_name, environment="us-east-1"):
        self._api_key = api_key
        self._index_name = index_name
        self._environment = environment
        self._client = None

    def create_index(self, name, dimension):
        ...

    def store_embeddings(self, index_name, items):
        ...

    def search_similar(self, index_name, query_vector, top_k=10):
        ...

    # ... implement remaining abstract methods


def create_pinecone(config):
    return PineconeVectorStore(
        api_key=config.get("pinecone_api_key"),
        index_name=config.get("pinecone_index"),
        environment=config.get("pinecone_environment", "us-east-1"),
    )

grapsit.register_vector_store("pinecone", create_pinecone)

# Now use it
builder.chunk_embedder(
    vector_store_type="pinecone",
    pinecone_api_key="pk-...",
    pinecone_index="my_embeddings",
)
```

### Adding a custom processor

The same registry pattern applies to pipeline processors. There are three category registries — `construct_registry` (build pipeline), `query_registry` (query pipeline), and `modeling_registry` (KG modeling) — plus convenience functions for quick registration.

**Example: custom NER processor**

```python
import grapsit
from grapsit.core.base import BaseProcessor


class SpacyNERProcessor(BaseProcessor):
    """NER processor backed by spaCy."""

    def __init__(self, config, pipeline=None):
        super().__init__(config, pipeline)
        self._nlp = None
        self._labels = config.get("labels", [])

    def __call__(self, chunks, **kwargs):
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load(self.config.get("model", "en_core_web_sm"))

        from grapsit.models import EntityMention

        all_entities = []
        for chunk in chunks:
            doc = self._nlp(chunk.text)
            mentions = [
                EntityMention(
                    text=ent.text, label=ent.label_.lower(),
                    start=ent.start_char, end=ent.end_char, score=1.0,
                )
                for ent in doc.ents
                if not self._labels or ent.label_.lower() in self._labels
            ]
            all_entities.append(mentions)
        return {"entities": all_entities, "chunks": chunks}


# Register it
grapsit.register_construct_processor("ner_spacy", lambda config, pipeline=None: SpacyNERProcessor(config, pipeline))
```

Use the decorator form for more concise registration:

```python
from grapsit.core.registry import construct_registry

@construct_registry.register("ner_spacy")
def create_spacy_ner(config, pipeline=None):
    return SpacyNERProcessor(config, pipeline)
```

Once registered, the processor works in builder and YAML pipelines:

```python
# Builder API — use any registered processor by name
builder = grapsit.BuildConfigBuilder(name="spacy_pipeline")
builder.chunker(method="sentence")
builder.add_node(
    id="ner", processor="ner_spacy",
    config={"model": "en_core_web_sm", "labels": ["person", "org"]},
    inputs={"chunks": "chunker_result.chunks"},
    output="ner_result",
)
builder.relex_llm(api_key="sk-...", entity_labels=["person", "org"], relation_labels=["works at"])
builder.graph_writer(neo4j_uri="bolt://localhost:7687")
executor = builder.build()
result = executor.execute({"texts": ["Einstein worked at Princeton."]})
```

```yaml
# YAML config — just reference the processor name
nodes:
  - id: ner
    processor: ner_spacy
    config:
      model: en_core_web_sm
      labels: [person, org]
    inputs:
      chunks: chunker_result.chunks
    output: ner_result
```

**Example: custom retriever**

```python
from grapsit.core.registry import query_registry

@query_registry.register("my_hybrid_retriever")
def create_hybrid_retriever(config, pipeline=None):
    return MyHybridRetriever(config, pipeline)
```

**Available registries:**

| Registry | Category | Convenience function |
|----------|----------|---------------------|
| `grapsit.construct_registry` | Build pipeline (chunker, NER, relex, graph_writer, ...) | `grapsit.register_construct_processor()` |
| `grapsit.query_registry` | Query pipeline (parser, retrievers, reasoner, ...) | `grapsit.register_query_processor()` |
| `grapsit.modeling_registry` | KG modeling (community detection, KG training, ...) | `grapsit.register_modeling_processor()` |
| `grapsit.processor_registry` | Composite — searches all three registries | N/A (use category-specific functions) |

## Development

```bash
# Create venv and install
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# Run tests (mocked GLiNER + GLinker + Neo4j, no external services needed)
.venv/bin/pytest tests/ -v    # 704 tests
```

## Roadmap

- ~~**LLM extraction** — OpenAI-compatible NER and relation extraction as alternative to GLiNER~~ Done
- ~~**Entity linking** — GLinker integration for linking mentions to a knowledge base~~ Done
- ~~**Query pipeline** — DAG-based retrieval: query parsing, subgraph retrieval, LLM reasoning~~ Done
- ~~**FalkorDB support** — alternative graph store, `store_type` parameter across all APIs~~ Done
- ~~**Memgraph support** — Memgraph as an additional graph store backend~~ Done
- ~~**Data ingest** — `ingest_data()` for writing pre-structured entities/relations directly~~ Done
- ~~**JSON export** — `json_output` parameter to save extracted data in ingest-ready format~~ Done
- ~~**LLM function calling** — tool-use layer with built-in graph query tools, Cypher translation, custom tool registration~~ Done
- ~~**Community detection** — hierarchical community detection, LLM summarization, vector embeddings~~ Done
- ~~**Chunk & entity embeddings** — embed chunks/entities during build for semantic retrieval strategies~~ Done
- ~~**KG modeling** — PyKEEN KG embedding training, storage, and query-time link prediction~~ Done
- ~~**Store pool** — named store pool with shared connections, typed vector store configs, context manager support~~ Done
- **In-memory store** — testing without a database
- **CLI** — `grapsit build --config pipeline.yaml`
