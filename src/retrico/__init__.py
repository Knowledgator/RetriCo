"""retrico — End-to-end Graph RAG using Knowledgator technologies."""

# Register all construct processors on import.
from . import construct as _construct  # noqa: F401
# Register all query processors on import.
from . import query as _query  # noqa: F401
# Register modeling processors on import.
from . import modeling as _modeling  # noqa: F401

from .core.dag import DAGExecutor, PipeContext
from .core.factory import ProcessorFactory
from .core.builders import (
    RetriCoBuilder, RetriCoSearch, RetriCoIngest, RetriCoCommunity,
    RetriCoModeling, RetriCoFusedSearch,
    # Backward-compatible aliases
    BuildConfigBuilder, QueryConfigBuilder, IngestConfigBuilder,
    CommunityConfigBuilder, KGModelingConfigBuilder, FusedQueryBuilder,
)
from .core.registry import (
    processor_registry, construct_registry, query_registry, modeling_registry,
)
from .models import Document, Chunk, Entity, EntityMention, Relation, KGTriple, Subgraph, QueryResult
from .store import (
    BaseGraphStore, Neo4jGraphStore, FalkorDBGraphStore, FalkorDBLiteGraphStore, MemgraphGraphStore,
    create_store, create_graph_store, graph_store_registry,
    BaseVectorStore, create_vector_store, vector_store_registry,
    BaseRelationalStore, create_relational_store, relational_store_registry,
    BaseStoreConfig, FalkorDBLiteConfig, Neo4jConfig, FalkorDBConfig, MemgraphConfig,
    resolve_store_config,
    BaseVectorStoreConfig, InMemoryVectorConfig, FaissVectorConfig,
    QdrantVectorConfig, GraphDBVectorConfig,
    StorePool,
)
from .llm import BaseLLMClient, OpenAIClient
from .extraction import GLiNEREngine, LLMExtractionEngine, EntityLinkerEngine, ExtractionResult

from typing import Dict, List, Any, Optional, Union
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


# -- Processor registration convenience functions -----------------------------

def register_construct_processor(name: str, factory):
    """Register a custom construct (build pipeline) processor.

    Args:
        name: Processor name (used in pipeline config).
        factory: Callable ``(config_dict, pipeline) -> processor``.
    """
    construct_registry.register(name, factory)


def register_query_processor(name: str, factory):
    """Register a custom query pipeline processor.

    Args:
        name: Processor name (used in pipeline config).
        factory: Callable ``(config_dict, pipeline) -> processor``.
    """
    query_registry.register(name, factory)


def register_modeling_processor(name: str, factory):
    """Register a custom modeling processor.

    Args:
        name: Processor name (used in pipeline config).
        factory: Callable ``(config_dict, pipeline) -> processor``.
    """
    modeling_registry.register(name, factory)


__all__ = [
    # Public API
    "build_graph",
    "build_graph_from_store",
    "build_graph_from_pdf",
    "query_graph",
    "ingest_data",
    "detect_communities",
    "train_kg_model",
    "extract",
    # Extraction engines
    "GLiNEREngine",
    "LLMExtractionEngine",
    "EntityLinkerEngine",
    "ExtractionResult",
    # Core
    "DAGExecutor",
    "PipeContext",
    "ProcessorFactory",
    "RetriCoBuilder",
    "RetriCoSearch",
    "RetriCoIngest",
    "RetriCoCommunity",
    "RetriCoModeling",
    "RetriCoFusedSearch",
    # Backward-compatible aliases
    "BuildConfigBuilder",
    "QueryConfigBuilder",
    "IngestConfigBuilder",
    "CommunityConfigBuilder",
    "KGModelingConfigBuilder",
    "FusedQueryBuilder",
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
    "FalkorDBLiteGraphStore",
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
    # Processor registries
    "processor_registry",
    "construct_registry",
    "query_registry",
    "modeling_registry",
    "register_construct_processor",
    "register_query_processor",
    "register_modeling_processor",
    "BaseStoreConfig",
    "FalkorDBLiteConfig",
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
    store_type: str = "falkordb_lite",
    falkordb_host: str = "localhost",
    falkordb_port: int = 6379,
    falkordb_graph: str = "retrico",
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
    write_reversed_relations: bool = False,
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
        write_reversed_relations: If True, write reversed relations for bidirectional
            path traversal.

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

    builder = RetriCoBuilder(name="build_graph")
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

    builder.graph_writer(json_output=json_output, write_reversed_relations=write_reversed_relations)

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
    return executor.run(texts=texts)


def build_graph_from_pdf(
    pdf_paths: List[str],
    *,
    entity_labels: List[str],
    relation_labels: List[str] = None,
    extract_text: bool = True,
    extract_tables: bool = True,
    page_ids: List[int] = None,
    ner_model: str = "urchade/gliner_multi-v2.1",
    relex_model: str = "knowledgator/gliner-relex-large-v0.5",
    ner_threshold: float = 0.3,
    relex_threshold: float = 0.5,
    device: str = "cpu",
    verbose: bool = False,
    json_output: str = None,
    store_type: str = "falkordb_lite",
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "password",
    neo4j_database: str = "neo4j",
    falkordb_host: str = "localhost",
    falkordb_port: int = 6379,
    falkordb_graph: str = "retrico",
    store_config: BaseStoreConfig = None,
    write_reversed_relations: bool = False,
) -> PipeContext:
    """Build a knowledge graph from PDF files.

    Each PDF page becomes one chunk.  Tables are extracted and converted
    to Markdown format.  Chunk metadata includes ``page_number`` and
    ``source_pdf``.

    Args:
        pdf_paths: List of paths to PDF files.
        entity_labels: Entity types for NER.
        relation_labels: Relation types. If None, relex is skipped.
        extract_text: Extract regular text from pages.
        extract_tables: Extract tables and convert to Markdown.
        page_ids: Specific page numbers to extract (0-indexed). None = all.
        ner_model: GLiNER model for NER.
        relex_model: GLiNER-relex model for relation extraction.
        ner_threshold: NER confidence threshold.
        relex_threshold: Relation extraction threshold.
        device: "cpu" or "cuda".
        verbose: Enable verbose logging.
        json_output: Path to save extracted data as JSON.
        store_config: A BaseStoreConfig object. Overrides individual store params.
        write_reversed_relations: If True, write reversed relations.

    Returns:
        PipeContext with all intermediate results.
    """
    if store_config is None:
        store_config = resolve_store_config(
            store_type=store_type, neo4j_uri=neo4j_uri, neo4j_user=neo4j_user,
            neo4j_password=neo4j_password, neo4j_database=neo4j_database,
            falkordb_host=falkordb_host, falkordb_port=falkordb_port,
            falkordb_graph=falkordb_graph,
        )

    builder = RetriCoBuilder(name="build_graph_from_pdf")
    builder.store(store_config)
    builder.pdf_reader(
        extract_text=extract_text,
        extract_tables=extract_tables,
        page_ids=page_ids,
    )
    builder.ner_gliner(
        model=ner_model, labels=entity_labels,
        threshold=ner_threshold, device=device,
    )

    if relation_labels:
        builder.relex_gliner(
            model=relex_model,
            entity_labels=entity_labels,
            relation_labels=relation_labels,
            threshold=relex_threshold,
            device=device,
        )

    builder.graph_writer(
        json_output=json_output,
        write_reversed_relations=write_reversed_relations,
    )

    executor = builder.build(verbose=verbose)
    return executor.run(pdf_paths=pdf_paths)


def build_graph_from_store(
    *,
    table: str = "documents",
    text_field: str = "text",
    id_field: str = "id",
    metadata_fields: List[str] = None,
    limit: int = 0,
    offset: int = 0,
    filter_empty: bool = True,
    entity_labels: List[str],
    relation_labels: List[str] = None,
    relational_store_type: str = "sqlite",
    sqlite_path: str = ":memory:",
    postgres_host: str = "localhost",
    postgres_port: int = 5432,
    postgres_user: str = "postgres",
    postgres_password: str = "",
    postgres_database: str = "retrico",
    elasticsearch_url: str = "http://localhost:9200",
    elasticsearch_api_key: str = None,
    elasticsearch_index_prefix: str = "retrico_",
    ner_model: str = "urchade/gliner_multi-v2.1",
    relex_model: str = "knowledgator/gliner-relex-large-v0.5",
    ner_threshold: float = 0.3,
    relex_threshold: float = 0.5,
    chunk_method: str = "sentence",
    device: str = "cpu",
    verbose: bool = False,
    json_output: str = None,
    store_type: str = "falkordb_lite",
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "password",
    neo4j_database: str = "neo4j",
    store_config: BaseStoreConfig = None,
) -> PipeContext:
    """Build a knowledge graph from texts stored in a relational database.

    Reads records from the configured relational store, extracts texts,
    and runs the standard build pipeline (chunker -> NER -> relex -> graph_writer).

    Args:
        table: Table/collection to read from.
        text_field: Column containing the text.
        id_field: Column used as document source/ID.
        metadata_fields: Extra columns to include in Document metadata.
        limit: Max records to fetch (0 = all).
        offset: Records to skip.
        filter_empty: Skip records with empty/missing text.
        entity_labels: Entity types for NER.
        relation_labels: Relation types for extraction. If None, relex is skipped.
        relational_store_type: Relational backend ("sqlite", "postgres", "elasticsearch").
        ner_model: GLiNER model for NER.
        relex_model: GLiNER-relex model for relation extraction.
        chunk_method: "sentence", "paragraph", or "fixed".
        device: "cpu" or "cuda".
        verbose: Enable verbose logging.
        json_output: Path to save extracted data as JSON.
        store_config: A BaseStoreConfig for the graph store. Overrides individual params.

    Returns:
        PipeContext with all intermediate results.
    """
    if store_config is None:
        store_config = resolve_store_config(
            store_type=store_type, neo4j_uri=neo4j_uri, neo4j_user=neo4j_user,
            neo4j_password=neo4j_password, neo4j_database=neo4j_database,
        )

    builder = RetriCoBuilder(name="build_graph_from_store")
    builder.store(store_config)

    # Configure relational store for store_reader
    relational_kwargs = {"relational_store_type": relational_store_type}
    if relational_store_type == "sqlite":
        relational_kwargs["sqlite_path"] = sqlite_path
    elif relational_store_type == "postgres":
        relational_kwargs.update({
            "postgres_host": postgres_host, "postgres_port": postgres_port,
            "postgres_user": postgres_user, "postgres_password": postgres_password,
            "postgres_database": postgres_database,
        })
    elif relational_store_type == "elasticsearch":
        relational_kwargs.update({
            "elasticsearch_url": elasticsearch_url,
            "elasticsearch_api_key": elasticsearch_api_key,
            "elasticsearch_index_prefix": elasticsearch_index_prefix,
        })
    builder.chunk_store(**relational_kwargs)

    builder.store_reader(
        table=table, text_field=text_field, id_field=id_field,
        metadata_fields=metadata_fields, limit=limit, offset=offset,
        filter_empty=filter_empty,
    )
    builder.chunker(method=chunk_method)
    builder.ner_gliner(
        model=ner_model, labels=entity_labels,
        threshold=ner_threshold, device=device,
    )

    if relation_labels:
        builder.relex_gliner(
            model=relex_model, entity_labels=entity_labels,
            relation_labels=relation_labels, threshold=relex_threshold,
            device=device,
        )

    builder.graph_writer(json_output=json_output)

    executor = builder.build(verbose=verbose)
    return executor.run()


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
    store_type: str = "falkordb_lite",
    falkordb_host: str = "localhost",
    falkordb_port: int = 6379,
    falkordb_graph: str = "retrico",
    memgraph_uri: str = "bolt://localhost:7687",
    memgraph_user: str = "",
    memgraph_password: str = "",
    memgraph_database: str = "memgraph",
    retrieval_strategy: Union[str, List[str]] = "entity",
    retriever_kwargs: Dict[str, Any] = None,
    fusion_strategy: str = "union",
    fusion_top_k: int = 0,
    fusion_weights: List[float] = None,
    fusion_min_sources: int = 2,
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
        retrieval_strategy: Strategy name or list of names. Single string:
            "entity" (default), "community", "chunk_embedding",
            "entity_embedding", "tool", "path", or "keyword".
            List of strings: triggers multi-retriever fusion.
        retriever_kwargs: Extra kwargs passed to the retriever builder method
            (e.g. top_k, vector_index_name, max_tool_rounds).
        fusion_strategy: Fusion strategy when using multiple retrievers
            ("union", "rrf", "weighted", "intersection"). Default: "union".
        fusion_top_k: Max entities after fusion (0 = all).
        fusion_weights: Per-retriever weights for "weighted" fusion.
        fusion_min_sources: Min sources for "intersection" fusion.
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

    if retriever_kwargs is None:
        retriever_kwargs = {}

    # Strategy name → builder method name mapping
    _STRATEGY_TO_METHOD = {
        "entity": "retriever",
        "community": "community_retriever",
        "path": "path_retriever",
        "chunk_embedding": "chunk_embedding_retriever",
        "entity_embedding": "entity_embedding_retriever",
        "tool": "tool_retriever",
        "keyword": "keyword_retriever",
    }

    # Strategies that need a parser
    _ENTITY_STRATEGIES = {"entity", "entity_embedding", "path"}
    # kg_scored uses tool-calling parser (needs api_key, not entity_labels)
    _TOOL_PARSER_STRATEGIES = {"kg_scored"}

    # Normalize to list for uniform handling
    strategies = (
        retrieval_strategy
        if isinstance(retrieval_strategy, list)
        else [retrieval_strategy]
    )
    is_multi = len(strategies) > 1

    # Determine if any strategy needs entity-based parser
    needs_entity_parser = any(s in _ENTITY_STRATEGIES for s in strategies)
    needs_tool_parser = any(s in _TOOL_PARSER_STRATEGIES for s in strategies)

    def _configure_retriever(builder, strategy):
        """Configure a single retriever on a builder."""
        if strategy == "entity":
            builder.retriever(max_hops=max_hops)
        elif strategy == "community":
            builder.community_retriever(**retriever_kwargs)
        elif strategy == "chunk_embedding":
            builder.chunk_embedding_retriever(**retriever_kwargs)
        elif strategy == "entity_embedding":
            builder.entity_embedding_retriever(**retriever_kwargs)
        elif strategy == "tool":
            tool_kw = dict(retriever_kwargs)
            if api_key is not None and "api_key" not in tool_kw:
                tool_kw["api_key"] = api_key
            if "model" not in tool_kw:
                tool_kw["model"] = model
            builder.tool_retriever(**tool_kw)
        elif strategy == "path":
            builder.path_retriever(**retriever_kwargs)
        elif strategy == "keyword":
            builder.keyword_retriever(**retriever_kwargs)
        elif strategy == "kg_scored":
            builder.kg_scorer(**retriever_kwargs)
        elif strategy in _STRATEGY_TO_METHOD:
            getattr(builder, _STRATEGY_TO_METHOD[strategy])(**retriever_kwargs)
        else:
            raise ValueError(
                f"Unknown retrieval_strategy: {strategy!r}. "
                "Expected 'entity', 'community', 'chunk_embedding', "
                "'entity_embedding', 'tool', 'path', 'keyword', or 'kg_scored'."
            )

    def _add_parser_and_linker(builder):
        """Add parser and linker to a builder based on strategy needs."""
        if needs_tool_parser:
            if api_key is None:
                raise ValueError("api_key required for 'kg_scored' strategy.")
            parser_kwargs = {"method": "tool", "api_key": api_key, "model": model}
            if entity_labels:
                parser_kwargs["labels"] = entity_labels
            relation_labels = retriever_kwargs.get("relation_labels")
            if relation_labels:
                parser_kwargs["relation_labels"] = relation_labels
            builder.query_parser(**parser_kwargs)
        elif needs_entity_parser:
            if entity_labels is None:
                raise ValueError(
                    "entity_labels required for entity-based retrieval strategies "
                    f"({', '.join(s for s in strategies if s in _ENTITY_STRATEGIES)})."
                )
            parser_kwargs = {"method": ner_method, "labels": entity_labels}
            if ner_model is not None:
                parser_kwargs["model"] = ner_model
            if ner_method == "llm" and api_key is not None:
                parser_kwargs["api_key"] = api_key
                parser_kwargs["model"] = model
            builder.query_parser(**parser_kwargs)

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

    if is_multi:
        # Multi-strategy: create separate builders, combine via RetriCoFusedSearch
        sub_builders = []
        for strategy in strategies:
            sb = RetriCoSearch(name=f"query_{strategy}")
            sb.store(store_config)
            _add_parser_and_linker(sb)
            _configure_retriever(sb, strategy)
            sub_builders.append(sb)

        fusion_kw = {"strategy": fusion_strategy, "top_k": fusion_top_k}
        if fusion_weights:
            fusion_kw["weights"] = fusion_weights
        if fusion_strategy == "intersection":
            fusion_kw["min_sources"] = fusion_min_sources

        fused = RetriCoFusedSearch(*sub_builders, name="query_graph", **fusion_kw)

        chunk_kw = {}
        if "chunk_entity_source" in retriever_kwargs:
            chunk_kw["chunk_entity_source"] = retriever_kwargs["chunk_entity_source"]
        fused.chunk_retriever(**chunk_kw)

        if api_key is not None:
            fused.reasoner(api_key=api_key, model=model)

        executor = fused.build(verbose=verbose)
    else:
        # Single strategy: use RetriCoSearch directly (backward compatible)
        builder = RetriCoSearch(name="query_graph")
        builder.store(store_config)
        _add_parser_and_linker(builder)
        _configure_retriever(builder, strategies[0])

        chunk_kw = {}
        if "chunk_entity_source" in retriever_kwargs:
            chunk_kw["chunk_entity_source"] = retriever_kwargs["chunk_entity_source"]
        builder.chunk_retriever(**chunk_kw)

        if api_key is not None:
            builder.reasoner(api_key=api_key, model=model)

        executor = builder.build(verbose=verbose)

    ctx = executor.run(query=query)

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
    data: List[Dict[str, Any]],
    *,
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "password",
    neo4j_database: str = "neo4j",
    store_type: str = "falkordb_lite",
    falkordb_host: str = "localhost",
    falkordb_port: int = 6379,
    falkordb_graph: str = "retrico",
    memgraph_uri: str = "bolt://localhost:7687",
    memgraph_user: str = "",
    memgraph_password: str = "",
    memgraph_database: str = "memgraph",
    json_output: str = None,
    verbose: bool = False,
    store_config: BaseStoreConfig = None,
    write_reversed_relations: bool = False,
) -> PipeContext:
    """Ingest pre-structured entities and relations into the graph database.

    Bypasses chunking, NER, and relation extraction — writes directly.

    Args:
        data: List of dicts, each with ``entities`` (required), ``relations``
            (optional), and ``text`` (optional) keys.  Each dict groups
            related entities/relations together, optionally with a source text.
        neo4j_uri: Neo4j bolt URI.
        neo4j_user: Neo4j user.
        neo4j_password: Neo4j password.
        neo4j_database: Neo4j database name.
        store_type: Graph store backend ("neo4j", "falkordb", "memgraph").
        json_output: Path to also save data as JSON (ingest-ready format).
        verbose: Enable verbose logging.
        store_config: A BaseStoreConfig object. If provided, overrides individual store params.
        write_reversed_relations: If True, write reversed relations for bidirectional
            path traversal.

    Returns:
        PipeContext with writer_result containing entity_count, relation_count.

    Example::

        import retrico

        result = retrico.ingest_data(
            data=[
                {
                    "entities": [
                        {"text": "Einstein", "label": "person"},
                        {"text": "Ulm", "label": "location"},
                    ],
                    "relations": [
                        {"head": "Einstein", "tail": "Ulm", "type": "born_in"},
                    ],
                    "text": "Einstein was born in Ulm.",
                },
            ],
            neo4j_uri="bolt://localhost:7687",
        )
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

    builder = RetriCoIngest(name="ingest_data")
    builder.store(store_config)
    builder.graph_writer(json_output=json_output, write_reversed_relations=write_reversed_relations)

    executor = builder.build(verbose=verbose)
    return executor.run(data=data)


def detect_communities(
    *,
    method: str = "louvain",
    levels: int = 1,
    resolution: float = 1.0,
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "password",
    neo4j_database: str = "neo4j",
    store_type: str = "falkordb_lite",
    falkordb_host: str = "localhost",
    falkordb_port: int = 6379,
    falkordb_graph: str = "retrico",
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

    builder = RetriCoCommunity(name="detect_communities")
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
    return executor.run()


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
    store_type: str = "falkordb_lite",
    falkordb_host: str = "localhost",
    falkordb_port: int = 6379,
    falkordb_graph: str = "retrico",
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

    builder = RetriCoModeling(name="train_kg_model")
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
    return executor.run()


def extract(
    texts: List[str],
    *,
    entity_labels: List[str],
    relation_labels: List[str] = None,
    method: str = "gliner",
    ner_model: str = None,
    relex_model: str = None,
    device: str = "cpu",
    api_key: str = None,
    model: str = "gpt-4o-mini",
    threshold: float = 0.3,
    relation_threshold: float = 0.5,
) -> ExtractionResult:
    """Extract entities and relations without storing to a database.

    Args:
        texts: Input texts to process.
        entity_labels: Entity types for NER.
        relation_labels: Relation types. If None, only NER is performed.
        method: "gliner" or "llm".
        ner_model: Model name for NER (GLiNER model or LLM model).
        relex_model: Model name for relex (GLiNER-relex model). Only used
            when method="gliner" and relation_labels are provided.
        device: "cpu" or "cuda" (GLiNER only).
        api_key: OpenAI API key (required for method="llm").
        model: LLM model name (for method="llm").
        threshold: NER confidence threshold.
        relation_threshold: Relation extraction threshold (GLiNER only).

    Returns:
        ExtractionResult with entities and relations.
    """
    if method == "gliner":
        if relation_labels:
            engine = GLiNEREngine(
                model=relex_model or "knowledgator/gliner-relex-large-v0.5",
                labels=entity_labels,
                relation_labels=relation_labels,
                threshold=threshold,
                relation_threshold=relation_threshold,
                device=device,
            )
        else:
            engine = GLiNEREngine(
                model=ner_model or "urchade/gliner_multi-v2.1",
                labels=entity_labels,
                threshold=threshold,
                device=device,
            )
        return engine.extract(texts)
    elif method == "llm":
        engine = LLMExtractionEngine(
            api_key=api_key,
            model=model,
            labels=entity_labels,
            relation_labels=relation_labels or [],
        )
        return engine.extract(texts)
    else:
        raise ValueError(f"Unknown method: {method!r}. Expected 'gliner' or 'llm'.")
