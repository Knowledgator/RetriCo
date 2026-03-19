# Knowledge Explorer

A simple web app that builds and searches a knowledge graph using retrico. Pre-loaded with data about Knowledgator technologies (GLiNER, GLinker, GLiClass, retrico).

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python app.py                     # FalkorDB Lite (default, no external DB)
python app.py --neo4j             # Neo4j at bolt://localhost:7687
python app.py --openai-api-key sk-...  # Enable LLM reasoning
```

Open http://localhost:8000.

## Features

- **Add Text** — paste text to extract entities and relations into the knowledge graph
- **Search** — query the graph with multiple strategies (entity, semantic, path, fused, graph)
- **Explore** — interactive graph visualization with node detail panel

Seed data is loaded on startup via structured ingest (no NER/relex model needed).