"""
Knowledge Explorer — simple knowledge graph builder & searcher.

Features:
  - Text input -> KG ingestion
  - Search: entity lookup + semantic search + path reasoning (fused)
  - Interactive graph exploration

Usage:
    pip install -e "../../[dev]"
    pip install -r requirements.txt
    python app.py                       # FalkorDB Lite (default, no external DB)
    python app.py --neo4j               # Neo4j at bolt://localhost:7687
    python app.py --port 8080           # custom port

Open http://localhost:8000 in your browser.
"""

import argparse
import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import retrico

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("knowledge_explorer")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

ENTITY_LABELS = [
    "person", "organization", "location", "technology",
    "concept", "country", "city",
]
RELATION_LABELS = [
    "created", "founded by", "located in", "part of",
    "uses", "supports", "specializes in", "related to",
    "built on", "written in", "variant of",
]

# Filled by lifespan
state: dict = {}

# In-memory graph for visualization (accumulated across ingests)
graph_data: dict = {"nodes": {}, "edges": []}


def _build_ingest_pipeline(args) -> retrico.DAGExecutor:
    """Create the graph-building pipeline."""
    builder = retrico.RetriCoBuilder(name="explorer_build")

    if args.neo4j:
        builder.graph_store(retrico.Neo4jConfig(
            uri=args.neo4j_uri, user=args.neo4j_user, password=args.neo4j_password,
        ))

    builder.chunker(method="sentence")
    builder.relex_gliner(
        entity_labels=ENTITY_LABELS,
        relation_labels=RELATION_LABELS,
        relation_threshold=0.5,
    )
    builder.graph_writer(setup_indexes=True)
    builder.entity_embedder(
        embedding_method="sentence_transformer",
        model_name="all-MiniLM-L6-v2",
        vector_store_type="graph_db",
    )
    builder.chunk_embedder(
        embedding_method="sentence_transformer",
        model_name="all-MiniLM-L6-v2",
        vector_store_type="graph_db",
    )
    return builder.build(verbose=True)


def _get_store_kw(args) -> dict:
    if args.neo4j:
        return dict(
            store_type="neo4j",
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password,
        )
    return {}


def _build_search_pipeline(name, args, strategies, use_llm=False) -> retrico.DAGExecutor:
    """Build a search pipeline with the given retrieval strategies."""
    builder = retrico.RetriCoSearch(name=name)
    store_kw = _get_store_kw(args)

    builder.query_parser(method="gliner", labels=ENTITY_LABELS, threshold=0.4)

    for s in strategies:
        if s == "entity":
            builder.retriever(max_hops=2, **store_kw)
        elif s == "semantic":
            builder.entity_embedding_retriever(
                top_k=10,
                embedding_method="sentence_transformer",
                model_name="all-MiniLM-L6-v2",
                vector_store_type="graph_db",
                **store_kw,
            )
        elif s == "path":
            builder.path_retriever(max_path_length=4, **store_kw)

    if len(strategies) > 1:
        builder.fusion(strategy="rrf", top_k=20)

    builder.chunk_retriever(**store_kw)

    if use_llm and args.openai_api_key:
        builder.reasoner(
            api_key=args.openai_api_key,
            model=args.openai_model,
        )

    return builder.build(verbose=True)


# Cache of built pipelines keyed by (strategies_tuple, use_llm)
_pipeline_cache: dict = {}


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Knowledge Explorer app")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--neo4j", action="store_true", help="Use Neo4j instead of FalkorDB Lite")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default="password")
    parser.add_argument("--openai-api-key", default=None, help="OpenAI API key for LLM reasoner")
    parser.add_argument("--openai-model", default="gpt-4o-mini", help="LLM model for reasoner")
    return parser.parse_args()


args = parse_args()


def _get_or_build_pipeline(strategies: list[str], use_llm: bool) -> retrico.DAGExecutor:
    key = (tuple(sorted(strategies)), use_llm)
    if key not in _pipeline_cache:
        name = f"search_{'_'.join(sorted(strategies))}{'_llm' if use_llm else ''}"
        logger.info(f"Building search pipeline: {name}")
        _pipeline_cache[key] = _build_search_pipeline(name, args, strategies, use_llm)
    return _pipeline_cache[key]


def _preload_seed_data():
    """Load seed_data.json into the knowledge graph via structured ingest (no NER/relex)."""
    seed_path = Path(__file__).parent / "seed_data.json"
    if not seed_path.exists():
        logger.info("No seed_data.json found, skipping preload.")
        return
    with open(seed_path) as f:
        seed_items = json.load(f)
    if not seed_items:
        return

    # Add reverse relations so path search can traverse in both directions
    for item in seed_items:
        reverse = []
        for rel in item.get("relations", []):
            reverse.append({
                "head": rel["tail"],
                "tail": rel["head"],
                "type": rel["type"] + "_reverse",
            })
        item.setdefault("relations", []).extend(reverse)

    logger.info(f"Preloading {len(seed_items)} seed items into the knowledge graph...")

    # Build a structured ingest pipeline (bypasses NER/relex)
    builder = retrico.RetriCoIngest(name="seed_ingest")
    if args.neo4j:
        builder.graph_store(retrico.Neo4jConfig(
            uri=args.neo4j_uri, user=args.neo4j_user, password=args.neo4j_password,
        ))
    builder.graph_writer(setup_indexes=True)
    builder.entity_embedder(
        embedding_method="sentence_transformer",
        model_name="all-MiniLM-L6-v2",
        vector_store_type="graph_db",
    )
    builder.chunk_embedder(
        embedding_method="sentence_transformer",
        model_name="all-MiniLM-L6-v2",
        vector_store_type="graph_db",
    )
    executor = builder.build(verbose=True)
    ctx = executor.run({"data": seed_items})

    # Accumulate graph_data for visualization
    writer = ctx.get("writer_result") if ctx.has("writer_result") else {}
    entity_map = writer.get("entity_map", {})
    for key, entity in entity_map.items():
        node_id = entity.canonical_name
        if node_id not in graph_data["nodes"]:
            graph_data["nodes"][node_id] = {
                "id": node_id,
                "label": entity.label,
                "type": entity.entity_type,
                "mentions": len(entity.mentions),
            }
        else:
            graph_data["nodes"][node_id]["mentions"] += len(entity.mentions)

    # Accumulate edges from seed_data directly (skip reverse relations)
    for item in seed_items:
        for rel in item.get("relations", []):
            if rel["type"].endswith("_reverse"):
                continue
            head = rel["head"].strip().lower()
            tail = rel["tail"].strip().lower()
            rel_type = rel["type"].upper().replace(" ", "_")
            edge_key = f"{head}|{rel_type}|{tail}"
            existing = [e for e in graph_data["edges"] if e["_key"] == edge_key]
            if not existing:
                graph_data["edges"].append({
                    "_key": edge_key,
                    "from": head,
                    "to": tail,
                    "type": rel_type,
                    "score": 1.0,
                })

    doc_id = str(uuid.uuid4())[:8]
    state["documents"].append({
        "id": doc_id,
        "source": "seed:knowledgator",
        "text_preview": "Pre-loaded seed data about Knowledgator technologies",
        "entities": writer.get("entity_count", 0),
        "relations": writer.get("relation_count", 0),
        "chunks": writer.get("chunk_count", 0),
    })

    if hasattr(executor, "close"):
        executor.close()
    logger.info("Seed data loaded.")


@asynccontextmanager
async def lifespan(application: FastAPI):
    logger.info("Building pipelines...")
    state["ingest"] = _build_ingest_pipeline(args)
    _pipeline_cache[(("entity", "path", "semantic"), False)] = _build_search_pipeline(
        "explorer_search_fused", args, ["entity", "semantic", "path"], use_llm=False,
    )
    state["documents"] = []
    _preload_seed_data()
    logger.info("Pipelines ready.")
    yield
    if hasattr(state.get("ingest"), "close"):
        state["ingest"].close()
    for executor in _pipeline_cache.values():
        if hasattr(executor, "close"):
            executor.close()


app = FastAPI(title="Knowledge Explorer", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_ingest(texts: list[str], source: str) -> dict:
    """Run the ingest pipeline and return stats."""
    ctx = state["ingest"].run({"texts": texts})
    writer = ctx.get("writer_result") if ctx.has("writer_result") else {}

    entity_map = writer.get("entity_map", {})
    for key, entity in entity_map.items():
        node_id = entity.canonical_name
        if node_id not in graph_data["nodes"]:
            graph_data["nodes"][node_id] = {
                "id": node_id,
                "label": entity.label,
                "type": entity.entity_type,
                "mentions": len(entity.mentions),
            }
        else:
            graph_data["nodes"][node_id]["mentions"] += len(entity.mentions)

    if ctx.has("relex_result"):
        relex = ctx.get("relex_result")
        for chunk_rels in (relex.get("relations") or []):
            for rel in chunk_rels:
                head = rel.head_text.strip().lower()
                tail = rel.tail_text.strip().lower()
                edge_key = f"{head}|{rel.relation_type}|{tail}"
                existing = [e for e in graph_data["edges"] if e["_key"] == edge_key]
                if not existing:
                    graph_data["edges"].append({
                        "_key": edge_key,
                        "from": head,
                        "to": tail,
                        "type": rel.relation_type,
                        "score": rel.score,
                    })

    doc_id = str(uuid.uuid4())[:8]
    record = {
        "id": doc_id,
        "source": source,
        "text_preview": texts[0][:200] if texts else "",
        "entities": writer.get("entity_count", 0),
        "relations": writer.get("relation_count", 0),
        "chunks": writer.get("chunk_count", 0),
    }
    state["documents"].append(record)
    return record


def _extract_subgraph(sg) -> dict:
    return {
        "entities": [
            {"label": e.label, "type": e.entity_type, "id": str(e.id)}
            for e in (sg.entities or [])
        ],
        "relations": [
            {"head": r.tail_text, "tail": r.head_text,
             "type": r.relation_type.removesuffix("_REVERSE"), "score": r.score}
            if r.relation_type.endswith("_REVERSE") else
            {"head": r.head_text, "tail": r.tail_text, "type": r.relation_type, "score": r.score}
            for r in (sg.relations or [])
        ],
        "chunks": [
            {"text": c.text, "document_id": str(c.document_id)}
            for c in (sg.chunks or [])
        ],
    }


def _run_pipeline_search(query: str, strategies: list[str], use_llm: bool) -> dict:
    executor = _get_or_build_pipeline(strategies, use_llm)
    ctx = executor.run({"query": query})

    if ctx.has("reasoner_result"):
        qr = ctx.get("reasoner_result").get("result")
        if qr:
            sg = _extract_subgraph(qr.subgraph) if hasattr(qr, "subgraph") and qr.subgraph else {
                "entities": [], "relations": [], "chunks": [],
            }
            return {"answer": qr.answer if hasattr(qr, "answer") else str(qr), **sg}

    for key in ("chunk_result", "fusion_result", "retriever_result",
                "path_retriever_result", "entity_embedding_retriever_result"):
        if ctx.has(key):
            data = ctx.get(key)
            sg = data.get("subgraph") if isinstance(data, dict) else None
            if sg:
                return {"answer": None, **_extract_subgraph(sg)}
    return {"answer": None, "entities": [], "relations": [], "chunks": []}


def _fuzzy_match(query: str, text: str) -> float:
    q = query.lower()
    t = text.lower()
    if q == t:
        return 1.0
    if q in t or t in q:
        return 0.9
    q_tokens = set(q.split())
    t_tokens = set(t.split())
    if not q_tokens:
        return 0.0
    overlap = len(q_tokens & t_tokens)
    if overlap > 0:
        return 0.5 + 0.4 * (overlap / max(len(q_tokens), len(t_tokens)))
    def bigrams(s):
        return {s[i:i+2] for i in range(len(s) - 1)} if len(s) >= 2 else {s}
    qb, tb = bigrams(q), bigrams(t)
    if not qb or not tb:
        return 0.0
    return len(qb & tb) / max(len(qb), len(tb))


def _graph_search(query: str) -> dict:
    query = query.strip()
    if not query:
        return {"nodes": [], "edges": []}

    query_terms = [query] + query.split()
    scored_nodes = []
    for node_id, node in graph_data["nodes"].items():
        best_score = max(_fuzzy_match(term, node["label"]) for term in query_terms)
        if best_score >= 0.3:
            scored_nodes.append((best_score, node_id, node))

    scored_nodes.sort(key=lambda x: x[0], reverse=True)
    matched_ids = {nid for _, nid, _ in scored_nodes[:20]}

    expanded_ids = set(matched_ids)
    relevant_edges = []
    for e in graph_data["edges"]:
        if e["from"] in matched_ids or e["to"] in matched_ids:
            expanded_ids.add(e["from"])
            expanded_ids.add(e["to"])
            relevant_edges.append({
                "from": e["from"], "to": e["to"],
                "type": e["type"], "score": e["score"],
            })

    nodes = []
    for nid in expanded_ids:
        if nid in graph_data["nodes"]:
            node = dict(graph_data["nodes"][nid])
            node["matched"] = nid in matched_ids
            nodes.append(node)

    return {"nodes": nodes, "edges": relevant_edges}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path(__file__).parent / "static" / "index.html").read_text()


@app.post("/api/ingest/text")
async def ingest_text(text: str = Form(...)):
    if not text.strip():
        return {"error": "Empty text"}
    result = await asyncio.to_thread(_run_ingest, [text.strip()], "text_input")
    return {"status": "ok", **result}


@app.post("/api/search")
async def search(
    query: str = Form(...),
    mode: str = Form("fused"),
    use_llm: str = Form("false"),
):
    if not query.strip():
        return {"error": "Empty query"}

    llm_flag = use_llm.lower() == "true"

    if mode == "graph":
        result = await asyncio.to_thread(_graph_search, query.strip())
        return {"status": "ok", "query": query.strip(), "mode": "graph", **result}

    strategy_map = {
        "entity": ["entity"],
        "semantic": ["semantic"],
        "path": ["path"],
        "fused": ["entity", "semantic", "path"],
    }
    strategies = strategy_map.get(mode, ["entity", "semantic", "path"])

    result = await asyncio.to_thread(_run_pipeline_search, query.strip(), strategies, llm_flag)
    return {"status": "ok", "query": query.strip(), "mode": mode, "use_llm": llm_flag, **result}


@app.get("/api/documents")
async def list_documents():
    return {"documents": state.get("documents", [])}


@app.get("/api/graph")
async def get_graph():
    nodes = list(graph_data["nodes"].values())
    edges = [{"from": e["from"], "to": e["to"], "type": e["type"], "score": e["score"]}
             for e in graph_data["edges"]]
    return {"nodes": nodes, "edges": edges}


@app.get("/api/graph/neighbors/{node_id:path}")
async def get_neighbors(node_id: str):
    node_id = node_id.strip().lower()
    neighbor_ids = set()
    relevant_edges = []
    for e in graph_data["edges"]:
        if e["from"] == node_id or e["to"] == node_id:
            neighbor_ids.add(e["from"])
            neighbor_ids.add(e["to"])
            relevant_edges.append({"from": e["from"], "to": e["to"], "type": e["type"], "score": e["score"]})
    neighbor_ids.add(node_id)
    nodes = [graph_data["nodes"][nid] for nid in neighbor_ids if nid in graph_data["nodes"]]
    return {"nodes": nodes, "edges": relevant_edges, "center": node_id}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")