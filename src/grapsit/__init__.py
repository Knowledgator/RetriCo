"""grapsit — End-to-end Graph RAG using Knowledgator technologies."""

# Register all construct processors on import.
from . import construct as _construct  # noqa: F401
# Register all query processors on import.
from . import query as _query  # noqa: F401
# Register modeling processors on import.
from . import modeling as _modeling  # noqa: F401

from .core.dag import DAGExecutor, PipeContext
from .core.factory import ProcessorFactory
from .core.builders import BuildConfigBuilder, QueryConfigBuilder, IngestConfigBuilder, CommunityConfigBuilder
from .models import Document, Chunk, Entity, EntityMention, Relation, KGTriple, Subgraph, QueryResult
from .store.base import BaseGraphStore
from .store.neo4j_store import Neo4jGraphStore
from .store.falkordb_store import FalkorDBGraphStore
from .store.memgraph_store import MemgraphGraphStore
from .store import create_store
from .llm import BaseLLMClient, OpenAIClient

from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

__all__ = [
    # Public API
    "build_graph",
    "query_graph",
    "ingest_data",
    "detect_communities",
    # Core
    "DAGExecutor",
    "PipeContext",
    "ProcessorFactory",
    "BuildConfigBuilder",
    "QueryConfigBuilder",
    "IngestConfigBuilder",
    "CommunityConfigBuilder",
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
    "BaseGraphStore",
    "Neo4jGraphStore",
    "FalkorDBGraphStore",
    "MemgraphGraphStore",
    "create_store",
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
    linker_model: str = None,
    linker_entities: Any = None,
    linker_threshold: float = 0.5,
    linker_executor: Any = None,
    store_type: str = "neo4j",
    falkordb_host: str = "localhost",
    falkordb_port: int = 6379,
    falkordb_graph: str = "grapsit",
    memgraph_uri: str = "bolt://localhost:7687",
    memgraph_user: str = "",
    memgraph_password: str = "",
    memgraph_database: str = "memgraph",
    json_output: str = None,
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
        json_output: Path to save extracted data as JSON (ingest-ready format).

    Returns:
        PipeContext with all intermediate results.
    """
    builder = BuildConfigBuilder(name="build_graph")
    builder.chunker(method=chunk_method)
    builder.ner_gliner(model=ner_model, labels=entity_labels, threshold=ner_threshold, device=device)

    # Add linker if any linker param is provided
    if linker_executor or linker_model or linker_entities:
        linker_kwargs = {"threshold": linker_threshold}
        if linker_executor:
            linker_kwargs["executor"] = linker_executor
        if linker_model:
            linker_kwargs["model"] = linker_model
        if linker_entities:
            linker_kwargs["entities"] = linker_entities
        builder.linker(**linker_kwargs)

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
        store_type=store_type,
        falkordb_host=falkordb_host,
        falkordb_port=falkordb_port,
        falkordb_graph=falkordb_graph,
        memgraph_uri=memgraph_uri,
        memgraph_user=memgraph_user,
        memgraph_password=memgraph_password,
        memgraph_database=memgraph_database,
        json_output=json_output,
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
    linker_model: str = None,
    linker_entities: Any = None,
    linker_threshold: float = 0.5,
    linker_executor: Any = None,
    linker_neo4j: bool = False,
    store_type: str = "neo4j",
    falkordb_host: str = "localhost",
    falkordb_port: int = 6379,
    falkordb_graph: str = "grapsit",
    memgraph_uri: str = "bolt://localhost:7687",
    memgraph_user: str = "",
    memgraph_password: str = "",
    memgraph_database: str = "memgraph",
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

    # Add linker if any linker param is provided
    if linker_executor or linker_model or linker_entities or linker_neo4j:
        linker_kwargs = {"threshold": linker_threshold}
        if linker_executor:
            linker_kwargs["executor"] = linker_executor
        if linker_model:
            linker_kwargs["model"] = linker_model
        if linker_entities:
            linker_kwargs["entities"] = linker_entities
        if linker_neo4j:
            linker_kwargs["neo4j_uri"] = neo4j_uri
            linker_kwargs["neo4j_user"] = neo4j_user
            linker_kwargs["neo4j_password"] = neo4j_password
            linker_kwargs["neo4j_database"] = neo4j_database
        builder.linker(**linker_kwargs)

    builder.retriever(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        neo4j_database=neo4j_database,
        max_hops=max_hops,
        store_type=store_type,
        falkordb_host=falkordb_host,
        falkordb_port=falkordb_port,
        falkordb_graph=falkordb_graph,
        memgraph_uri=memgraph_uri,
        memgraph_user=memgraph_user,
        memgraph_password=memgraph_password,
        memgraph_database=memgraph_database,
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


def ingest_data(
    entities: List[Dict[str, Any]],
    relations: List[Dict[str, Any]] = None,
    *,
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "password",
    neo4j_database: str = "neo4j",
    store_type: str = "neo4j",
    falkordb_host: str = "localhost",
    falkordb_port: int = 6379,
    falkordb_graph: str = "grapsit",
    memgraph_uri: str = "bolt://localhost:7687",
    memgraph_user: str = "",
    memgraph_password: str = "",
    memgraph_database: str = "memgraph",
    json_output: str = None,
    verbose: bool = False,
) -> PipeContext:
    """Ingest pre-structured entities and relations into the graph database.

    Bypasses chunking, NER, and relation extraction — writes directly.

    Args:
        entities: List of entity dicts, each with ``text`` and ``label`` keys.
            Optional keys: ``id`` (explicit ID), ``properties`` (metadata dict).
        relations: List of relation dicts, each with ``head``, ``tail``, ``type``.
            Optional keys: ``score``, ``properties``.
        neo4j_uri: Neo4j bolt URI.
        neo4j_user: Neo4j user.
        neo4j_password: Neo4j password.
        neo4j_database: Neo4j database name.
        store_type: Graph store backend ("neo4j", "falkordb", "memgraph").
        json_output: Path to also save data as JSON (ingest-ready format).
        verbose: Enable verbose logging.

    Returns:
        PipeContext with writer_result containing entity_count, relation_count.

    Example::

        import grapsit

        result = grapsit.ingest_data(
            entities=[
                {"text": "Einstein", "label": "person"},
                {"text": "Ulm", "label": "location"},
            ],
            relations=[
                {"head": "Einstein", "tail": "Ulm", "type": "born_in"},
            ],
            neo4j_uri="bolt://localhost:7687",
        )
    """
    if relations is None:
        relations = []

    builder = IngestConfigBuilder(name="ingest_data")
    builder.graph_writer(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        neo4j_database=neo4j_database,
        store_type=store_type,
        falkordb_host=falkordb_host,
        falkordb_port=falkordb_port,
        falkordb_graph=falkordb_graph,
        memgraph_uri=memgraph_uri,
        memgraph_user=memgraph_user,
        memgraph_password=memgraph_password,
        memgraph_database=memgraph_database,
        json_output=json_output,
    )

    executor = builder.build(verbose=verbose)
    return executor.execute({"entities": entities, "relations": relations})


def detect_communities(
    *,
    method: str = "louvain",
    levels: int = 1,
    resolution: float = 1.0,
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "password",
    neo4j_database: str = "neo4j",
    store_type: str = "neo4j",
    falkordb_host: str = "localhost",
    falkordb_port: int = 6379,
    falkordb_graph: str = "grapsit",
    memgraph_uri: str = "bolt://localhost:7687",
    memgraph_user: str = "",
    memgraph_password: str = "",
    memgraph_database: str = "memgraph",
    api_key: str = None,
    model: str = "gpt-4o-mini",
    top_k: int = 10,
    embedding_method: str = "sentence_transformer",
    model_name: str = "all-MiniLM-L6-v2",
    vector_store_type: str = "in_memory",
    verbose: bool = False,
) -> PipeContext:
    """Detect communities in an existing knowledge graph.

    Optionally generates LLM summaries (if ``api_key`` is provided) and
    embeds community summaries into a vector store.

    Args:
        method: Community detection algorithm ("louvain" or "leiden").
        levels: Number of hierarchical levels.
        resolution: Resolution parameter for Louvain.
        neo4j_uri: Neo4j bolt URI.
        neo4j_user: Neo4j user.
        neo4j_password: Neo4j password.
        neo4j_database: Neo4j database name.
        store_type: Graph store backend ("neo4j", "falkordb", "memgraph").
        api_key: OpenAI API key. If provided, enables summarization + embedding.
        model: LLM model name for summarizer.
        top_k: Max entities per community for summarization context.
        embedding_method: "sentence_transformer" or "openai".
        model_name: Embedding model name.
        vector_store_type: "in_memory", "faiss", or "qdrant".
        verbose: Enable verbose logging.

    Returns:
        PipeContext with detector_result and optionally summarizer_result,
        embedder_result.
    """
    builder = CommunityConfigBuilder(name="detect_communities")
    builder.detector(
        method=method,
        levels=levels,
        resolution=resolution,
        store_type=store_type,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        neo4j_database=neo4j_database,
        falkordb_host=falkordb_host,
        falkordb_port=falkordb_port,
        falkordb_graph=falkordb_graph,
        memgraph_uri=memgraph_uri,
        memgraph_user=memgraph_user,
        memgraph_password=memgraph_password,
        memgraph_database=memgraph_database,
    )

    if api_key is not None:
        builder.summarizer(api_key=api_key, model=model, top_k=top_k)
        builder.embedder(
            embedding_method=embedding_method,
            model_name=model_name,
            vector_store_type=vector_store_type,
        )

    executor = builder.build(verbose=verbose)
    return executor.execute({})
