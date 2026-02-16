"""grapsit — End-to-end Graph RAG using Knowledgator technologies."""

# Register all construct processors on import.
from . import construct as _construct  # noqa: F401
# Register all query processors on import.
from . import query as _query  # noqa: F401

from .core.dag import DAGExecutor, PipeContext
from .core.factory import ProcessorFactory
from .core.builders import BuildConfigBuilder, QueryConfigBuilder
from .models import Document, Chunk, Entity, EntityMention, Relation, KGTriple, Subgraph, QueryResult
from .store.neo4j_store import Neo4jGraphStore
from .llm import BaseLLMClient, OpenAIClient

from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

__all__ = [
    # Public API
    "build_graph",
    "query_graph",
    # Core
    "DAGExecutor",
    "PipeContext",
    "ProcessorFactory",
    "BuildConfigBuilder",
    "QueryConfigBuilder",
    # Models
    "Document",
    "Chunk",
    "Entity",
    "EntityMention",
    "Relation",
    "KGTriple",
    "Subgraph",
    "QueryResult",
    # Store
    "Neo4jGraphStore",
    # LLM
    "BaseLLMClient",
    "OpenAIClient",
]


def build_graph(
    texts: List[str],
    *,
    entity_labels: List[str],
    relation_labels: List[str] = None,
    ner_model: str = "urchade/gliner_multi-v2.1",
    relex_model: str = "knowledgator/gliner-relex-large-v0.5",
    ner_threshold: float = 0.3,
    relex_threshold: float = 0.5,
    chunk_method: str = "sentence",
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "password",
    neo4j_database: str = "neo4j",
    device: str = "cpu",
    verbose: bool = False,
) -> PipeContext:
    """Build a knowledge graph from texts in one call.

    Args:
        texts: Input texts to process.
        entity_labels: Entity types for NER (e.g. ["person", "organization"]).
        relation_labels: Relation types for extraction. If None, relex is skipped.
        ner_model: GLiNER model for NER.
        relex_model: GLiNER-relex model for relation extraction.
        ner_threshold: NER confidence threshold.
        relex_threshold: Relation extraction threshold.
        chunk_method: "sentence", "paragraph", or "fixed".
        neo4j_uri: Neo4j bolt URI.
        neo4j_user: Neo4j user.
        neo4j_password: Neo4j password.
        neo4j_database: Neo4j database name.
        device: "cpu" or "cuda".
        verbose: Enable verbose logging.

    Returns:
        PipeContext with all intermediate results.
    """
    builder = BuildConfigBuilder(name="build_graph")
    builder.chunker(method=chunk_method)
    builder.ner_gliner(model=ner_model, labels=entity_labels, threshold=ner_threshold, device=device)

    if relation_labels:
        builder.relex_gliner(
            model=relex_model,
            entity_labels=entity_labels,
            relation_labels=relation_labels,
            threshold=relex_threshold,
            device=device,
        )

    builder.graph_writer(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        neo4j_database=neo4j_database,
    )

    executor = builder.build(verbose=verbose)
    return executor.execute({"texts": texts})


def query_graph(
    query: str,
    *,
    entity_labels: List[str],
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "password",
    neo4j_database: str = "neo4j",
    max_hops: int = 2,
    ner_method: str = "gliner",
    ner_model: str = None,
    api_key: str = None,
    model: str = "gpt-4o-mini",
    verbose: bool = False,
) -> QueryResult:
    """Query a knowledge graph in one call.

    Args:
        query: Natural language query.
        entity_labels: Entity types for NER on the query.
        neo4j_uri: Neo4j bolt URI.
        neo4j_user: Neo4j user.
        neo4j_password: Neo4j password.
        neo4j_database: Neo4j database name.
        max_hops: Subgraph expansion depth.
        ner_method: "gliner" or "llm" for query parsing.
        ner_model: Model name for the query parser.
        api_key: OpenAI API key. If provided, enables LLM reasoner.
        model: LLM model name for reasoner (and LLM parser if ner_method="llm").
        verbose: Enable verbose logging.

    Returns:
        QueryResult with subgraph, answer, and metadata.
    """
    builder = QueryConfigBuilder(name="query_graph")

    parser_kwargs = {"method": ner_method, "labels": entity_labels}
    if ner_model is not None:
        parser_kwargs["model"] = ner_model
    if ner_method == "llm" and api_key is not None:
        parser_kwargs["api_key"] = api_key
        parser_kwargs["model"] = model
    builder.query_parser(**parser_kwargs)

    builder.retriever(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        neo4j_database=neo4j_database,
        max_hops=max_hops,
    )
    builder.chunk_retriever()

    if api_key is not None:
        builder.reasoner(api_key=api_key, model=model)

    executor = builder.build(verbose=verbose)
    ctx = executor.execute({"query": query})

    # Extract QueryResult from the appropriate output
    if api_key is not None and ctx.has("reasoner_result"):
        return ctx.get("reasoner_result")["result"]
    elif ctx.has("chunk_result"):
        subgraph = ctx.get("chunk_result")["subgraph"]
        return QueryResult(query=query, subgraph=subgraph)
    else:
        return QueryResult(query=query)
