# grapsit

End-to-end Graph RAG framework that turns unstructured text into a queryable knowledge graph. Built on [Knowledgator](https://github.com/Knowledgator) technologies (GLiNER, GLinker) with Neo4j for graph storage.

**Pipeline**: `text chunking → entity recognition → relation extraction → Neo4j graph`

## Installation

```bash
pip install -e .
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

### What happens under the hood

1. **Chunking** — splits texts into sentences (configurable: sentence/paragraph/fixed)
2. **NER** — [GLiNER](https://github.com/urchade/GLiNER) extracts entities matching your labels
3. **Relation extraction** — [GLiNER-relex](https://huggingface.co/knowledgator/gliner-relex-large-v0.5) finds relations between entities
4. **Graph writing** — deduplicates entities, writes nodes and edges to Neo4j

## Usage

### Option 1: `build_graph()` convenience function

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

### Option 2: Builder API

Fine-grained control over each pipeline stage:

```python
from grapsit import BuildConfigBuilder

builder = BuildConfigBuilder(name="my_pipeline")

builder.chunker(method="sentence")

builder.ner_gliner(
    model="urchade/gliner_multi-v2.1",
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

### Option 3: YAML pipeline

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

After building, use `Neo4jGraphStore` to query directly:

```python
from grapsit import Neo4jGraphStore

store = Neo4jGraphStore(uri="bolt://localhost:7687", password="password")

# Look up an entity
entity = store.get_entity_by_label("Albert Einstein")

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

# Per-chunk relations (if relex was enabled)
relations = result.get("relex_result")["relations"]  # List[List[Relation]]

# Final write stats + deduplicated entity map
writer = result.get("writer_result")
print(writer["entity_count"], writer["relation_count"])
entity_map = writer["entity_map"]  # canonical_name -> Entity
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

```
$input (texts)
    │
    ▼
 chunker ──────────────────┐
    │                      │
    ▼                      ▼
 ner_gliner           relex_gliner
    │                      │
    └──────┐   ┌───────────┘
           ▼   ▼
       graph_writer
           │
           ▼
     Neo4j database
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
| `relex_gliner` | Relation extraction with GLiNER-relex | `model`, `entity_labels`, `relation_labels`, `threshold`, `relation_threshold` |
| `graph_writer` | Deduplicate and write to Neo4j | `neo4j_uri`, `neo4j_user`, `neo4j_password`, `neo4j_database` |

## Development

```bash
# Create venv and install
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# Run tests (mocked GLiNER + Neo4j, no external services needed)
.venv/bin/pytest tests/ -v
```

## Roadmap

- **LLM extraction** — OpenAI-compatible NER and relation extraction as alternative to GLiNER
- **Entity linking** — GLinker integration for linking mentions to a knowledge base
- **KG modeling** — node/edge embeddings, community detection (Leiden), path reasoning
- **Query pipeline** — DAG-based retrieval: query parsing, subgraph retrieval, LLM reasoning
- **In-memory store** — testing without Neo4j
- **CLI** — `grapsit build --config pipeline.yaml`
