"""grapsit — End-to-end Graph RAG using Knowledgator technologies."""

# Register all construct processors on import.
from . import construct as _construct  # noqa: F401
# Register all query processors on import.
from . import query as _query  # noqa: F401
# Register modeling processors on import.
from . import modeling as _modeling  # noqa: F401

from .core.dag import DAGExecutor, PipeContext
from .core.factory import ProcessorFactory
from .core.builders import BuildConfigBuilder, QueryConfigBuilder, IngestConfigBuilder, CommunityConfigBuilder, KGModelingConfigBuilder
from .models import Document, Chunk, Entity, EntityMention, Relation, KGTriple, Subgraph, QueryResult
from .store import (
    BaseGraphStore, Neo4jGraphStore, FalkorDBGraphStore, MemgraphGraphStore,
    create_store, create_graph_store, graph_store_registry,
    BaseVectorStore, create_vector_store, vector_store_registry,
    BaseRelationalStore, create_relational_store, relational_store_registry,
    BaseStoreConfig, Neo4jConfig, FalkorDBConfig, MemgraphConfig,
    resolve_store_config,
    BaseVectorStoreConfig, InMemoryVectorConfig, FaissVectorConfig,
    QdrantVectorConfig, GraphDBVectorConfig,
    StorePool,
)
from .llm import BaseLLMClient, OpenAIClient

from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


# -- Store registration convenience functions --------------------------------

def register_graph_store(name: str, factory):
    """Register a custom graph store backend.

    Args:
        name: Store type name (used in ``store_type`` config key).
        factory: Callable that takes a config dict and returns a BaseGraphStore.
    """
    graph_store_registry.register(name, factory)


def register_vector_store(name: str, factory):
    """Register a custom vector store backend.

    Args:
        name: Store type name (used in ``vector_store_type`` config key).
        factory: Callable that takes a config dict and returns a BaseVectorStore.
    """
    vector_store_registry.register(name, factory)


def register_relational_store(name: str, factory):
    """Register a custom relational store backend.

    Args:
        name: Store type name (used in ``relational_store_type`` config key).
        factory: Callable that takes a config dict and returns a BaseRelationalStore.
    """
    relational_store_registry.register(name, factory)


__all__ = [
    # Public API
    "build_graph",
    "query_graph",
    "ingest_data",
    "detect_communities",
    "train_kg_model",
    # Core
    "DAGExecutor",
    "PipeContext",
    "ProcessorFactory",
    "BuildConfigBuilder",
    "QueryConfigBuilder",
    "IngestConfigBuilder",
    "CommunityConfigBuilder",
    "KGModelingConfigBuilder",
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
    "create_graph_store",
    "graph_store_registry",
    "BaseVectorStore",
    "create_vector_store",
    "vector_store_registry",
    "BaseRelationalStore",
    "create_relational_store",
    "relational_store_registry",
    "register_graph_store",
    "register_vector_store",
    "register_relational_store",
    "BaseStoreConfig",
    "Neo4jConfig",
    "FalkorDBConfig",
    "MemgraphConfig",
    "BaseVectorStoreConfig",
    "InMemoryVectorConfig",
    "FaissVectorConfig",
    "QdrantVectorConfig",
    "GraphDBVectorConfig",
    "StorePool",
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
    embed_chunks: bool = False,
    embed_entities: bool = False,
    embedding_method: str = "sentence_transformer",
    embedding_model_name: str = "all-MiniLM-L6-v2",
    vector_store_type: str = "in_memory",
    store_config: BaseStoreConfig = None,
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
        store_config: A BaseStoreConfig object (Neo4jConfig, FalkorDBConfig, etc.).
            If provided, overrides individual store params.

    Returns:
        PipeContext with all intermediate results.
    """
    if store_config is None:
        store_config = resolve_store_config(
            store_type=store_type, neo4j_uri=neo4j_uri, neo4j_user=neo4j_user,
            neo4j_password=neo4j_password, neo4j_database=neo4j_database,
            falkordb_host=falkordb_host, falkordb_port=falkordb_port,
            falkordb_graph=falkordb_graph, memgraph_uri=memgraph_uri,
            memgraph_user=memgraph_user, memgraph_password=memgraph_password,
            memgraph_database=memgraph_database,
        )

    builder = BuildConfigBuilder(name="build_graph")
    builder.store(store_config)
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

    builder.graph_writer(json_output=json_output)

    if embed_chunks:
        builder.chunk_embedder(
            embedding_method=embedding_method,
            model_name=embedding_model_name,
            vector_store_type=vector_store_type,
        )
    if embed_entities:
        builder.entity_embedder(
            embedding_method=embedding_method,
            model_name=embedding_model_name,
            vector_store_type=vector_store_type,
        )

    executor = builder.build(verbose=verbose)
    return executor.execute({"texts": texts})


def query_graph(
    query: str,
    *,
    entity_labels: List[str] = None,
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
    retrieval_strategy: str = "entity",
    retriever_kwargs: Dict[str, Any] = None,
    store_config: BaseStoreConfig = None,
) -> QueryResult:
    """Query a knowledge graph in one call.

    Args:
        query: Natural language query.
        entity_labels: Entity types for NER on the query. Required for entity-based
            strategies (entity, entity_embedding, path).
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
        retrieval_strategy: Strategy name — "entity" (default), "community",
            "chunk_embedding", "entity_embedding", "tool", or "path".
        retriever_kwargs: Extra kwargs passed to the retriever builder method
            (e.g. top_k, vector_index_name, max_tool_rounds).
        store_config: A BaseStoreConfig object (Neo4jConfig, FalkorDBConfig, etc.).
            If provided, overrides individual store params.

    Returns:
        QueryResult with subgraph, answer, and metadata.
    """
    if store_config is None:
        store_config = resolve_store_config(
            store_type=store_type, neo4j_uri=neo4j_uri, neo4j_user=neo4j_user,
            neo4j_password=neo4j_password, neo4j_database=neo4j_database,
            falkordb_host=falkordb_host, falkordb_port=falkordb_port,
            falkordb_graph=falkordb_graph, memgraph_uri=memgraph_uri,
            memgraph_user=memgraph_user, memgraph_password=memgraph_password,
            memgraph_database=memgraph_database,
        )

    builder = QueryConfigBuilder(name="query_graph")
    builder.store(store_config)

    if retriever_kwargs is None:
        retriever_kwargs = {}

    # Strategies that need a parser
    _ENTITY_STRATEGIES = {"entity", "entity_embedding", "path"}
    # kg_scored uses tool-calling parser (needs api_key, not entity_labels)
    _TOOL_PARSER_STRATEGIES = {"kg_scored"}

    if retrieval_strategy in _TOOL_PARSER_STRATEGIES:
        if api_key is None:
            raise ValueError(
                f"api_key required for '{retrieval_strategy}' strategy."
            )
        parser_kwargs = {
            "method": "tool",
            "api_key": api_key,
            "model": model,
        }
        if entity_labels:
            parser_kwargs["labels"] = entity_labels
        relation_labels = retriever_kwargs.get("relation_labels")
        if relation_labels:
            parser_kwargs["relation_labels"] = relation_labels
        builder.query_parser(**parser_kwargs)
    elif retrieval_strategy in _ENTITY_STRATEGIES:
        if entity_labels is None:
            raise ValueError(
                f"entity_labels required for '{retrieval_strategy}' strategy."
            )
        parser_kwargs = {"method": ner_method, "labels": entity_labels}
        if ner_model is not None:
            parser_kwargs["model"] = ner_model
        if ner_method == "llm" and api_key is not None:
            parser_kwargs["api_key"] = api_key
            parser_kwargs["model"] = model
        builder.query_parser(**parser_kwargs)

        # Add linker if any linker param is provided
        if linker_executor or linker_model or linker_entities or linker_neo4j:
            linker_kw = {"threshold": linker_threshold}
            if linker_executor:
                linker_kw["executor"] = linker_executor
            if linker_model:
                linker_kw["model"] = linker_model
            if linker_entities:
                linker_kw["entities"] = linker_entities
            if linker_neo4j:
                linker_kw["store_config"] = store_config
            builder.linker(**linker_kw)

    # Configure retriever based on strategy
    if retrieval_strategy == "entity":
        builder.retriever(max_hops=max_hops)
    elif retrieval_strategy == "community":
        builder.community_retriever(**retriever_kwargs)
    elif retrieval_strategy == "chunk_embedding":
        builder.chunk_embedding_retriever(**retriever_kwargs)
    elif retrieval_strategy == "entity_embedding":
        builder.entity_embedding_retriever(**retriever_kwargs)
    elif retrieval_strategy == "tool":
        tool_kw = dict(retriever_kwargs)
        if api_key is not None and "api_key" not in tool_kw:
            tool_kw["api_key"] = api_key
        if "model" not in tool_kw:
            tool_kw["model"] = model
        builder.tool_retriever(**tool_kw)
    elif retrieval_strategy == "path":
        builder.path_retriever(**retriever_kwargs)
    elif retrieval_strategy == "kg_scored":
        builder.kg_scorer(**retriever_kwargs)
    else:
        raise ValueError(
            f"Unknown retrieval_strategy: {retrieval_strategy!r}. "
            "Expected 'entity', 'community', 'chunk_embedding', "
            "'entity_embedding', 'tool', 'path', or 'kg_scored'."
        )

    chunk_kw = {}
    if "chunk_entity_source" in retriever_kwargs:
        chunk_kw["chunk_entity_source"] = retriever_kwargs["chunk_entity_source"]
    builder.chunk_retriever(**chunk_kw)

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
    elif ctx.has("kg_scorer_result"):
        subgraph = ctx.get("kg_scorer_result")["subgraph"]
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
    store_config: BaseStoreConfig = None,
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
        store_config: A BaseStoreConfig object. If provided, overrides individual store params.

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

    if store_config is None:
        store_config = resolve_store_config(
            store_type=store_type, neo4j_uri=neo4j_uri, neo4j_user=neo4j_user,
            neo4j_password=neo4j_password, neo4j_database=neo4j_database,
            falkordb_host=falkordb_host, falkordb_port=falkordb_port,
            falkordb_graph=falkordb_graph, memgraph_uri=memgraph_uri,
            memgraph_user=memgraph_user, memgraph_password=memgraph_password,
            memgraph_database=memgraph_database,
        )

    builder = IngestConfigBuilder(name="ingest_data")
    builder.store(store_config)
    builder.graph_writer(json_output=json_output)

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
    store_config: BaseStoreConfig = None,
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
        store_config: A BaseStoreConfig object. If provided, overrides individual store params.

    Returns:
        PipeContext with detector_result and optionally summarizer_result,
        embedder_result.
    """
    if store_config is None:
        store_config = resolve_store_config(
            store_type=store_type, neo4j_uri=neo4j_uri, neo4j_user=neo4j_user,
            neo4j_password=neo4j_password, neo4j_database=neo4j_database,
            falkordb_host=falkordb_host, falkordb_port=falkordb_port,
            falkordb_graph=falkordb_graph, memgraph_uri=memgraph_uri,
            memgraph_user=memgraph_user, memgraph_password=memgraph_password,
            memgraph_database=memgraph_database,
        )

    builder = CommunityConfigBuilder(name="detect_communities")
    builder.store(store_config)
    builder.detector(
        method=method,
        levels=levels,
        resolution=resolution,
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


def train_kg_model(
    *,
    source: str = "graph_store",
    tsv_path: str = None,
    model: str = "RotatE",
    embedding_dim: int = 128,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 0.001,
    device: str = "cpu",
    model_path: str = "kg_model",
    vector_store_type: str = "in_memory",
    store_to_graph: bool = False,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
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
    verbose: bool = False,
    store_config: BaseStoreConfig = None,
) -> PipeContext:
    """Train KG embeddings using PyKEEN in one call.

    Reads triples from the graph store (or TSV), trains a model, and stores
    the resulting embeddings.

    Args:
        source: "graph_store" or "tsv".
        tsv_path: Path to TSV file (if source="tsv").
        model: PyKEEN model name (e.g. "RotatE", "TransE", "ComplEx").
        embedding_dim: Embedding dimension.
        epochs: Training epochs.
        batch_size: Training batch size.
        lr: Learning rate.
        device: "cpu" or "cuda".
        model_path: Directory to save model weights.
        vector_store_type: "in_memory", "faiss", or "qdrant".
        store_to_graph: Write entity embeddings to graph DB nodes.
        train_ratio: Fraction of triples for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for test.
        neo4j_uri: Neo4j bolt URI.
        neo4j_user: Neo4j user.
        neo4j_password: Neo4j password.
        neo4j_database: Neo4j database name.
        store_type: Graph store backend.
        verbose: Enable verbose logging.
        store_config: A BaseStoreConfig object. If provided, overrides individual store params.

    Returns:
        PipeContext with reader_result, trainer_result, storer_result.
    """
    if store_config is None:
        store_config = resolve_store_config(
            store_type=store_type, neo4j_uri=neo4j_uri, neo4j_user=neo4j_user,
            neo4j_password=neo4j_password, neo4j_database=neo4j_database,
            falkordb_host=falkordb_host, falkordb_port=falkordb_port,
            falkordb_graph=falkordb_graph, memgraph_uri=memgraph_uri,
            memgraph_user=memgraph_user, memgraph_password=memgraph_password,
            memgraph_database=memgraph_database,
        )

    builder = KGModelingConfigBuilder(name="train_kg_model")
    builder.store(store_config)
    builder.triple_reader(
        source=source,
        tsv_path=tsv_path,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    builder.trainer(
        model=model,
        embedding_dim=embedding_dim,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
    )
    builder.storer(
        model_path=model_path,
        vector_store_type=vector_store_type,
        store_to_graph=store_to_graph,
    )
    executor = builder.build(verbose=verbose)
    return executor.execute({})
