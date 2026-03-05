"""Configuration builders for graph-building and query pipelines."""

from typing import Any, Dict, List, Optional

import yaml
from pathlib import Path

from .factory import ProcessorFactory
from .dag import DAGExecutor
from ..store.config import (
    BaseStoreConfig, Neo4jConfig,
    BaseVectorStoreConfig,
    BaseRelationalStoreConfig,
    resolve_store_config, extract_store_kwargs, _STORE_FLAT_KEYS,
)
from ..store.pool import StorePool

# Types that are safe to serialize to YAML
_YAML_SAFE_TYPES = (str, int, float, bool, type(None))


def _strip_non_serializable(obj):
    """Recursively remove non-serializable values from a config dict."""
    if isinstance(obj, dict):
        return {
            k: _strip_non_serializable(v)
            for k, v in obj.items()
            if isinstance(v, (*_YAML_SAFE_TYPES, dict, list))
        }
    if isinstance(obj, list):
        return [_strip_non_serializable(item) for item in obj]
    return obj


class _BuilderBase:
    """Common infrastructure for all pipeline builders.

    Provides three builder-level store setters — one per database category:

    * ``graph_store(config)`` — graph database (Neo4j, FalkorDB, Memgraph)
    * ``vector_store(type, **kw)`` — vector database (in_memory, faiss, qdrant)
    * ``chunk_store(type, **kw)`` — relational chunk storage (sqlite, postgres — planned)

    ``store()`` is a convenience alias for ``graph_store()``.

    Component methods inherit the builder-level config when no explicit
    override is provided.
    """

    def __init__(self, name: str, description: str = None):
        self.name = name
        self.description = description or f"{name} — auto-generated"
        self._store_config: Optional[BaseStoreConfig] = None
        self._vector_store_config: Optional[Dict[str, Any]] = None
        self._chunk_store_config: Optional[Dict[str, Any]] = None
        # Named store registries for the pool
        self._graph_stores: Dict[str, dict] = {}
        self._vector_stores: Dict[str, dict] = {}
        self._relational_stores: Dict[str, dict] = {}
        # Custom nodes added via add_node()
        self._custom_nodes: List[dict] = []

    # -- store setters -------------------------------------------------------

    def store(self, config: BaseStoreConfig = None, **kwargs) -> "_BuilderBase":
        """Set builder-level graph store config (alias for ``graph_store()``)."""
        return self.graph_store(config, **kwargs)

    def graph_store(self, config: BaseStoreConfig = None, name: str = None, **kwargs) -> "_BuilderBase":
        """Set builder-level graph database config.

        All components that need a graph store inherit this unless overridden.

        Args:
            config: A ``BaseStoreConfig`` subclass (``Neo4jConfig``, ``FalkorDBConfig``, etc.).
            name: Pool name for the store (default: from config or ``"default"``).
            **kwargs: Legacy flat-key overrides (``neo4j_uri=...``, ``store_type=...``, etc.).
        """
        resolved = resolve_store_config(config, **kwargs)
        pool_name = name or getattr(resolved, "name", "default")
        self._store_config = resolved
        self._graph_stores[pool_name] = resolved.to_flat_dict()
        return self

    def vector_store(
        self,
        config: BaseVectorStoreConfig = None,
        type: str = "in_memory",
        name: str = "default",
        **kwargs,
    ) -> "_BuilderBase":
        """Set builder-level vector store config.

        Components that need a vector store (embedders, embedding retrievers)
        inherit this unless overridden.

        Args:
            config: A ``BaseVectorStoreConfig`` subclass.
            type: Vector store backend — ``"in_memory"``, ``"faiss"``, or ``"qdrant"``.
            name: Pool name for the store (default: from config or ``"default"``).
            **kwargs: Backend-specific params (e.g. ``qdrant_url``, ``use_gpu``).
        """
        if isinstance(config, BaseVectorStoreConfig):
            pool_name = name if name != "default" else config.name
            flat = config.to_flat_dict()
        else:
            pool_name = name
            flat = {"vector_store_type": type, **kwargs}
        self._vector_store_config = flat
        self._vector_stores[pool_name] = flat
        return self

    def chunk_store(
        self,
        config: BaseRelationalStoreConfig = None,
        type: str = "sqlite",
        name: str = "default",
        **kwargs,
    ) -> "_BuilderBase":
        """Set builder-level chunk/document store config.

        Components that need a chunk store (graph_writer, keyword_retriever)
        inherit this unless overridden.

        Args:
            config: A ``BaseRelationalStoreConfig`` subclass.
            type: Relational store backend — ``"sqlite"``, ``"postgres"``, or ``"elasticsearch"``.
            name: Pool name for the store (default: from config or ``"default"``).
            **kwargs: Backend-specific params (e.g. ``sqlite_path``, ``postgres_host``).
        """
        if isinstance(config, BaseRelationalStoreConfig):
            pool_name = name if name != "default" else config.name
            flat = config.to_flat_dict()
        else:
            pool_name = name
            flat = {"relational_store_type": type, **kwargs}
        self._chunk_store_config = flat
        self._relational_stores[pool_name] = flat
        return self

    # -- internal helpers ----------------------------------------------------

    def _effective_store(self, store_config=None, **kwargs) -> Optional[BaseStoreConfig]:
        """Resolve graph store: explicit > builder-level > None."""
        store_kwargs = extract_store_kwargs(kwargs)
        if store_config is not None or store_kwargs:
            return resolve_store_config(store_config, **store_kwargs)
        return self._store_config

    def _effective_store_flat(self, store_config=None, **kwargs) -> dict:
        """Resolve graph store and return flat dict (with Neo4j defaults as fallback)."""
        store = self._effective_store(store_config, **kwargs)
        if store is not None:
            return store.to_flat_dict()
        return Neo4jConfig().to_flat_dict()

    def _effective_vector_store(self, **overrides) -> dict:
        """Resolve vector store config: explicit overrides > builder-level > empty."""
        base = dict(self._vector_store_config) if self._vector_store_config else {}
        base.update(overrides)
        return base

    def _effective_relational_store(self, **overrides) -> dict:
        """Resolve relational store config: explicit overrides > builder-level > empty."""
        base = dict(self._chunk_store_config) if self._chunk_store_config else {}
        base.update(overrides)
        return base

    def _inherit_store(self, source_config: dict, target_config: dict):
        """Inherit store keys from source into target (if not already set)."""
        for key in _STORE_FLAT_KEYS:
            if key not in target_config and key in source_config:
                target_config[key] = source_config[key]

    # -- build / save --------------------------------------------------------

    def build(self, verbose: bool = False) -> DAGExecutor:
        """Build and return a DAGExecutor."""
        config = self.get_config()
        pool = self._build_pool()
        return ProcessorFactory.create_from_dict(
            config, verbose=verbose, store_pool=pool,
        )

    def _build_pool(self) -> Optional[StorePool]:
        """Build a StorePool from registered named stores (if any)."""
        if not self._graph_stores and not self._vector_stores and not self._relational_stores:
            return None
        pool = StorePool()
        for name, cfg in self._graph_stores.items():
            pool.register_graph(name, cfg)
        for name, cfg in self._vector_stores.items():
            pool.register_vector(name, cfg)
        for name, cfg in self._relational_stores.items():
            pool.register_relational(name, cfg)
        return pool

    def add_node(
        self,
        processor: str,
        *,
        id: str = None,
        inputs: Dict[str, str] = None,
        output: str = None,
        requires: List[str] = None,
        **config,
    ) -> "_BuilderBase":
        """Add a custom processor node to the pipeline.

        Args:
            processor: Registry name (e.g. ``"ner_spacy"``).
            id: Node ID (defaults to processor name).
            inputs: Override input mappings ``{"param": "source.field"}``.
                Omit to use the processor's ``default_inputs``.
            output: Override output key.
                Omit to use the processor's ``default_output``.
            requires: Explicit dependency node IDs.
            **config: Processor configuration kwargs.
        """
        node: Dict[str, Any] = {"processor": processor, "config": config}
        if id is not None:
            node["id"] = id
        if inputs is not None:
            node["inputs"] = {
                k: (
                    {"source": v.split(".", 1)[0], "fields": v.split(".", 1)[1]}
                    if "." in v
                    else {"source": v}
                )
                for k, v in inputs.items()
            }
        if output is not None:
            node["output"] = {"key": output}
        if requires is not None:
            node["requires"] = requires
        self._custom_nodes.append(node)
        return self

    def save(self, filepath: str) -> None:
        """Save config to YAML."""
        config = _strip_non_serializable(self.get_config())
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    def _stores_section(self) -> Optional[Dict[str, Any]]:
        """Build the ``stores`` section for get_config() output."""
        stores: Dict[str, Any] = {}
        if self._graph_stores:
            stores["graph"] = dict(self._graph_stores)
        if self._vector_stores:
            stores["vector"] = dict(self._vector_stores)
        if self._relational_stores:
            stores["relational"] = dict(self._relational_stores)
        return stores or None

    def _build_result(self, nodes: List[dict]) -> Dict[str, Any]:
        """Assemble final config dict from nodes, custom nodes, and stores."""
        nodes.extend(self._custom_nodes)
        result: Dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "nodes": nodes,
        }
        stores = self._stores_section()
        if stores:
            result["stores"] = stores
        return result

    def get_config(self) -> Dict[str, Any]:
        raise NotImplementedError


class _LinkerMixin:
    """Adds linker() method to builders."""

    _has_linker: bool
    _linker_config: Optional[Dict[str, Any]]

    def linker(
        self,
        executor: Any = None,
        model: str = "knowledgator/gliner-linker-large-v1.0",
        threshold: float = 0.5,
        entities: Any = None,
        store_config: BaseStoreConfig = None,
        **kwargs,
    ):
        self._has_linker = True
        self._linker_config = {
            "model": model,
            "threshold": threshold,
        }
        if executor is not None:
            self._linker_config["executor"] = executor
        if entities is not None:
            self._linker_config["entities"] = entities
        # Only inject store config if explicitly provided at this level
        store = self._effective_store(store_config, **kwargs)
        if store_config is not None or extract_store_kwargs(dict(kwargs)):
            self._linker_config.update(store.to_flat_dict())
        # Pass remaining non-store kwargs
        self._linker_config.update(kwargs)
        return self


class _GraphWriterMixin:
    """Adds graph_writer() method to builders."""

    _writer_config: Optional[Dict[str, Any]]

    def graph_writer(
        self,
        store_config: BaseStoreConfig = None,
        setup_indexes: bool = True,
        json_output: str = None,
        chunk_table: str = None,
        document_table: str = None,
        write_reversed_relations: bool = False,
        **kwargs,
    ):
        store = self._effective_store(store_config, **kwargs) or Neo4jConfig()
        self._writer_config = store.to_flat_dict()
        self._writer_config["setup_indexes"] = setup_indexes
        if json_output is not None:
            self._writer_config["json_output"] = json_output
        if chunk_table is not None:
            self._writer_config["chunk_table"] = chunk_table
        if document_table is not None:
            self._writer_config["document_table"] = document_table
        if write_reversed_relations:
            self._writer_config["write_reversed_relations"] = True
        # Merge relational store config so graph_writer can create/resolve it
        relational = self._effective_relational_store()
        if relational:
            for k, v in relational.items():
                if k not in self._writer_config:
                    self._writer_config[k] = v
        self._writer_config.update(kwargs)
        return self


class _EmbedderMixin:
    """Adds chunk_embedder() and entity_embedder() methods."""

    _chunk_embedder_config: Optional[Dict[str, Any]]
    _entity_embedder_config: Optional[Dict[str, Any]]

    def chunk_embedder(
        self,
        embedding_method: str = "sentence_transformer",
        model_name: str = "all-MiniLM-L6-v2",
        vector_store_type: str = None,
        vector_index_name: str = "chunk_embeddings",
        **extra,
    ):
        vs = self._effective_vector_store()
        self._chunk_embedder_config = {
            "embedding_method": embedding_method,
            "model_name": model_name,
            "vector_store_type": vector_store_type or vs.get("vector_store_type", "in_memory"),
            "vector_index_name": vector_index_name,
            **extra,
        }
        return self

    def entity_embedder(
        self,
        embedding_method: str = "sentence_transformer",
        model_name: str = "all-MiniLM-L6-v2",
        vector_store_type: str = None,
        vector_index_name: str = "entity_embeddings",
        **extra,
    ):
        vs = self._effective_vector_store()
        self._entity_embedder_config = {
            "embedding_method": embedding_method,
            "model_name": model_name,
            "vector_store_type": vector_store_type or vs.get("vector_store_type", "in_memory"),
            "vector_index_name": vector_index_name,
            **extra,
        }
        return self

class BuildConfigBuilder(_BuilderBase, 
                         _LinkerMixin, 
                         _GraphWriterMixin, 
                         _EmbedderMixin):
    """Declarative builder for graph-building pipeline configs.

    Usage::

        builder = BuildConfigBuilder(name="my_graph")
        builder.graph_store(Neo4jConfig(uri="bolt://localhost:7687"))
        builder.chunker(method="sentence")
        builder.ner_gliner(labels=["person", "org"])
        builder.graph_writer()  # inherits store from builder

        executor = builder.build()
        builder.save("configs/my_graph.yaml")
    """

    def __init__(self, name: str = "build_pipeline", description: str = None):
        super().__init__(name, description)
        self._store_reader_config: Optional[Dict[str, Any]] = None
        self._chunker_config: Optional[Dict[str, Any]] = None
        self._ner_config: Optional[Dict[str, Any]] = None
        self._ner_type: str = "ner_gliner"
        self._linker_config: Optional[Dict[str, Any]] = None
        self._has_linker: bool = False
        self._relex_config: Optional[Dict[str, Any]] = None
        self._relex_type: str = "relex_gliner"
        self._writer_config: Optional[Dict[str, Any]] = None
        self._chunk_embedder_config: Optional[Dict[str, Any]] = None
        self._entity_embedder_config: Optional[Dict[str, Any]] = None

    def store_reader(
        self,
        table: str = "documents",
        text_field: str = "text",
        id_field: str = "id",
        metadata_fields: List[str] = None,
        limit: int = 0,
        offset: int = 0,
        filter_empty: bool = True,
        **kwargs,
    ) -> "BuildConfigBuilder":
        """Configure a store_reader node that pulls texts from a relational store.

        When set, the chunker reads from the store_reader output instead of ``$input``.

        Args:
            table: Table/collection name to read from.
            text_field: Column containing the text.
            id_field: Column used as document source/ID.
            metadata_fields: Extra columns to include in Document metadata.
            limit: Max records (0 = all).
            offset: Records to skip.
            filter_empty: Skip records with empty/missing text.
            **kwargs: Relational store overrides (``relational_store_type``, etc.).
        """
        relational = self._effective_relational_store(**kwargs)
        self._store_reader_config = {
            "table": table,
            "text_field": text_field,
            "id_field": id_field,
            "metadata_fields": metadata_fields or [],
            "limit": limit,
            "offset": offset,
            "filter_empty": filter_empty,
            **relational,
        }
        return self

    def chunker(
        self,
        method: str = "sentence",
        chunk_size: int = 512,
        overlap: int = 50,
    ) -> "BuildConfigBuilder":
        self._chunker_config = {
            "method": method,
            "chunk_size": chunk_size,
            "overlap": overlap,
        }
        return self

    def ner_gliner(
        self,
        model: str = "knowledgator/gliner-multitask-large-v0.5",
        labels: List[str] = None,
        threshold: float = 0.3,
        batch_size: int = 8,
        device: str = "cpu",
        flat_ner: bool = True,
    ) -> "BuildConfigBuilder":
        self._ner_type = "ner_gliner"
        self._ner_config = {
            "model": model,
            "labels": labels or [],
            "threshold": threshold,
            "batch_size": batch_size,
            "device": device,
            "flat_ner": flat_ner,
        }
        return self

    def ner_llm(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = "gpt-4o-mini",
        labels: List[str] = None,
        temperature: float = 0.1,
        max_completion_tokens: int = 4096,
        timeout: float = 60.0,
    ) -> "BuildConfigBuilder":
        self._ner_type = "ner_llm"
        self._ner_config = {
            "model": model,
            "labels": labels or [],
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
            "timeout": timeout,
        }
        if api_key is not None:
            self._ner_config["api_key"] = api_key
        if base_url is not None:
            self._ner_config["base_url"] = base_url
        return self

    def relex_gliner(
        self,
        model: str = "knowledgator/gliner-relex-large-v0.5",
        entity_labels: List[str] = None,
        relation_labels: List[str] = None,
        threshold: float = 0.5,
        relation_threshold: float = 0.5,
        adjacency_threshold: float = 0.55,
        batch_size: int = 8,
        device: str = "cpu",
    ) -> "BuildConfigBuilder":
        self._relex_type = "relex_gliner"
        self._relex_config = {
            "model": model,
            "entity_labels": entity_labels or [],
            "relation_labels": relation_labels or [],
            "threshold": threshold,
            "relation_threshold": relation_threshold,
            "adjacency_threshold": adjacency_threshold,
            "batch_size": batch_size,
            "device": device,
        }
        return self

    def relex_llm(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = "gpt-5-mini",
        entity_labels: List[str] = None,
        relation_labels: List[str] = None,
        temperature: float = 0.1,
        max_completion_tokens: int = 4096,
        timeout: float = 60.0,
    ) -> "BuildConfigBuilder":
        self._relex_type = "relex_llm"
        self._relex_config = {
            "model": model,
            "entity_labels": entity_labels or [],
            "relation_labels": relation_labels or [],
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
            "timeout": timeout,
        }
        if api_key is not None:
            self._relex_config["api_key"] = api_key
        if base_url is not None:
            self._relex_config["base_url"] = base_url
        return self

    def get_config(self) -> Dict[str, Any]:
        """Build configuration dict."""
        if not self._chunker_config:
            self._chunker_config = {"method": "sentence"}
        # NER/linker/relex are optional — a chunks-only pipeline
        # (chunker → graph_writer) is valid for relational-store-only use cases.
        if not self._writer_config:
            self._writer_config = {}

        has_store_reader = self._store_reader_config is not None
        nodes: List[Dict[str, Any]] = []

        if has_store_reader:
            nodes.append({
                "id": "store_reader",
                "processor": "store_reader",
                "inputs": {},
                "output": {"key": "store_reader_result"},
                "config": self._store_reader_config,
            })

        # Chunker reads from store_reader when present, else from $input
        if has_store_reader:
            chunker_inputs = {
                "texts": {"source": "store_reader_result", "fields": "texts"},
                "documents": {"source": "store_reader_result", "fields": "documents"},
            }
            chunker_requires = ["store_reader"]
        else:
            chunker_inputs = {
                "texts": {"source": "$input", "fields": "texts"},
            }
            chunker_requires = []

        chunker_node: Dict[str, Any] = {
            "id": "chunker",
            "processor": "chunker",
            "inputs": chunker_inputs,
            "output": {"key": "chunker_result"},
            "config": self._chunker_config,
        }
        if chunker_requires:
            chunker_node["requires"] = chunker_requires
        nodes.append(chunker_node)

        has_ner = self._ner_config is not None
        has_linker = self._has_linker
        has_relex = self._relex_config is not None

        if has_ner:
            nodes.append({
                "id": "ner",
                "processor": self._ner_type,
                "requires": ["chunker"],
                "inputs": {
                    "chunks": {"source": "chunker_result", "fields": "chunks"},
                },
                "output": {"key": "ner_result"},
                "config": self._ner_config,
            })

        if has_linker:
            linker_inputs = {
                "chunks": {"source": "chunker_result", "fields": "chunks"},
            }
            linker_requires = ["chunker"]
            if has_ner:
                linker_inputs["entities"] = {"source": "ner_result", "fields": "entities"}
                linker_requires.append("ner")
            nodes.append({
                "id": "linker",
                "processor": "entity_linker",
                "requires": linker_requires,
                "inputs": linker_inputs,
                "output": {"key": "linker_result"},
                "config": self._linker_config,
            })

        # Determine the entity source after NER/linker
        if has_linker:
            entity_source_before_relex = "linker_result"
        elif has_ner:
            entity_source_before_relex = "ner_result"
        else:
            entity_source_before_relex = None

        if has_relex:
            relex_inputs = {
                "chunks": {"source": "chunker_result", "fields": "chunks"},
            }
            relex_requires = ["chunker"]
            if entity_source_before_relex:
                relex_inputs["entities"] = {"source": entity_source_before_relex, "fields": "entities"}
                relex_requires.append("linker" if has_linker else "ner")
            nodes.append({
                "id": "relex",
                "processor": self._relex_type,
                "requires": relex_requires,
                "inputs": relex_inputs,
                "output": {"key": "relex_result"},
                "config": self._relex_config,
            })

        # Determine entity/relation sources for the graph writer
        if has_relex:
            entity_source = "relex_result"
            writer_requires = ["chunker", "relex"]
        elif has_linker:
            entity_source = "linker_result"
            writer_requires = ["chunker", "linker"]
        elif has_ner:
            entity_source = "ner_result"
            writer_requires = ["chunker", "ner"]
        else:
            entity_source = None
            writer_requires = ["chunker"]

        writer_inputs = {
            "chunks": {"source": "chunker_result", "fields": "chunks"},
            "documents": {"source": "chunker_result", "fields": "documents"},
        }

        if entity_source:
            writer_inputs["entities"] = {"source": entity_source, "fields": "entities"}

        if has_relex:
            writer_inputs["relations"] = {"source": "relex_result", "fields": "relations"}

        nodes.append({
            "id": "graph_writer",
            "processor": "graph_writer",
            "requires": writer_requires,
            "inputs": writer_inputs,
            "output": {"key": "writer_result"},
            "config": self._writer_config,
        })

        # Optional embedder nodes (run after graph_writer)
        if self._chunk_embedder_config is not None:
            embedder_config = dict(self._writer_config)
            embedder_config.update(self._chunk_embedder_config)
            nodes.append({
                "id": "chunk_embedder",
                "processor": "chunk_embedder",
                "requires": ["graph_writer"],
                "inputs": {
                    "chunks": {"source": "chunker_result", "fields": "chunks"},
                },
                "output": {"key": "chunk_embedder_result"},
                "config": embedder_config,
            })

        if self._entity_embedder_config is not None:
            embedder_config = dict(self._writer_config)
            embedder_config.update(self._entity_embedder_config)
            nodes.append({
                "id": "entity_embedder",
                "processor": "entity_embedder",
                "requires": ["graph_writer"],
                "inputs": {
                    "entity_map": {"source": "writer_result", "fields": "entity_map"},
                },
                "output": {"key": "entity_embedder_result"},
                "config": embedder_config,
            })

        return self._build_result(nodes)


class IngestConfigBuilder(_BuilderBase, _GraphWriterMixin, _EmbedderMixin):
    """Declarative builder for raw data ingest pipelines.

    Writes pre-structured entities and relations directly to the graph database,
    bypassing NER and relation extraction.

    Usage::

        builder = IngestConfigBuilder(name="my_ingest")
        builder.graph_store(Neo4jConfig(uri="bolt://localhost:7687"))
        builder.graph_writer()
        executor = builder.build()
        result = executor.execute({
            "data": [
                {
                    "entities": [{"text": "Einstein", "label": "person"}],
                    "relations": [{"head": "Einstein", "tail": "Ulm", "type": "born_in"}],
                    "text": "Einstein was born in Ulm.",  # optional
                },
            ]
        })
    """

    def __init__(self, name: str = "ingest_pipeline", description: str = None):
        super().__init__(name, description)
        self._writer_config: Optional[Dict[str, Any]] = None
        self._ingest_config: Dict[str, Any] = {}
        self._chunk_embedder_config: Optional[Dict[str, Any]] = None
        self._entity_embedder_config: Optional[Dict[str, Any]] = None

    def chunker(
        self,
        method: str = "sentence",
        chunk_size: int = 512,
        overlap: int = 50,
    ) -> "IngestConfigBuilder":
        """Configure chunking for optional texts input."""
        self._ingest_config["chunk_method"] = method
        self._ingest_config["chunk_size"] = chunk_size
        self._ingest_config["chunk_overlap"] = overlap
        return self

    def get_config(self) -> Dict[str, Any]:
        """Build configuration dict."""
        if not self._writer_config:
            self._writer_config = {}

        ingest_inputs = {
            "data": {"source": "$input", "fields": "data"},
        }

        nodes = [
            {
                "id": "data_ingest",
                "processor": "data_ingest",
                "inputs": ingest_inputs,
                "output": {"key": "ingest_result"},
                "config": dict(self._ingest_config),
            },
            {
                "id": "graph_writer",
                "processor": "graph_writer",
                "requires": ["data_ingest"],
                "inputs": {
                    "entities": {"source": "ingest_result", "fields": "entities"},
                    "relations": {"source": "ingest_result", "fields": "relations"},
                    "chunks": {"source": "ingest_result", "fields": "chunks"},
                    "documents": {"source": "ingest_result", "fields": "documents"},
                },
                "output": {"key": "writer_result"},
                "config": self._writer_config,
            },
        ]

        if self._chunk_embedder_config is not None:
            embedder_config = dict(self._writer_config)
            embedder_config.update(self._chunk_embedder_config)
            nodes.append({
                "id": "chunk_embedder",
                "processor": "chunk_embedder",
                "requires": ["graph_writer"],
                "inputs": {
                    "chunks": {"source": "ingest_result", "fields": "chunks"},
                },
                "output": {"key": "chunk_embedder_result"},
                "config": embedder_config,
            })

        if self._entity_embedder_config is not None:
            embedder_config = dict(self._writer_config)
            embedder_config.update(self._entity_embedder_config)
            nodes.append({
                "id": "entity_embedder",
                "processor": "entity_embedder",
                "requires": ["graph_writer"],
                "inputs": {
                    "entity_map": {"source": "writer_result", "fields": "entity_map"},
                },
                "output": {"key": "entity_embedder_result"},
                "config": embedder_config,
            })

        return self._build_result(nodes)


class QueryConfigBuilder(_BuilderBase, _LinkerMixin):
    """Declarative builder for query pipeline configs.

    Usage::

        builder = QueryConfigBuilder(name="my_query")
        builder.graph_store(Neo4jConfig(uri="bolt://localhost:7687"))
        builder.query_parser(method="gliner", labels=["person", "location"])
        builder.retriever(max_hops=2)
        builder.chunk_retriever()
        builder.reasoner(api_key="...", model="gpt-4o-mini")

        executor = builder.build()
        result = executor.execute({"query": "Where was Einstein born?"})
    """

    def __init__(self, name: str = "query_pipeline", description: str = None):
        super().__init__(name, description)
        self._parser_config: Optional[Dict[str, Any]] = None
        self._linker_config: Optional[Dict[str, Any]] = None
        self._has_linker: bool = False
        self._retriever_nodes: List[Dict[str, Any]] = []  # [{"type": str, "config": dict}]
        self._fusion_config: Optional[Dict[str, Any]] = None
        self._chunk_config: Optional[Dict[str, Any]] = None
        self._reasoner_config: Optional[Dict[str, Any]] = None
        self._kg_scorer_config: Optional[Dict[str, Any]] = None

    # -- backward compat properties ------------------------------------------

    @property
    def _retriever_config(self) -> Optional[Dict[str, Any]]:
        if self._retriever_nodes:
            return self._retriever_nodes[0]["config"]
        return None

    @_retriever_config.setter
    def _retriever_config(self, value):
        # Only used by legacy code paths that assign directly
        if value is not None:
            if self._retriever_nodes:
                self._retriever_nodes[0]["config"] = value
            else:
                self._retriever_nodes.append({"type": "retriever", "config": value})

    @property
    def _retriever_type(self) -> str:
        if self._retriever_nodes:
            return self._retriever_nodes[0]["type"]
        return "retriever"

    @_retriever_type.setter
    def _retriever_type(self, value):
        if self._retriever_nodes:
            self._retriever_nodes[0]["type"] = value

    def query_parser(
        self,
        method: str = "gliner",
        model: str = None,
        labels: List[str] = None,
        relation_labels: List[str] = None,
        threshold: float = 0.3,
        device: str = "cpu",
        api_key: str = None,
        base_url: str = None,
        temperature: float = 0.1,
    ) -> "QueryConfigBuilder":
        self._parser_config = {
            "method": method,
            "labels": labels or [],
            "threshold": threshold,
            "device": device,
            "temperature": temperature,
        }
        if relation_labels is not None:
            self._parser_config["relation_labels"] = relation_labels
        if model is not None:
            self._parser_config["model"] = model
        if api_key is not None:
            self._parser_config["api_key"] = api_key
        if base_url is not None:
            self._parser_config["base_url"] = base_url
        return self

    def retriever(self, max_hops: int = 2, store_config: BaseStoreConfig = None, **kwargs) -> "QueryConfigBuilder":
        config = self._effective_store_flat(store_config, **kwargs)
        config["max_hops"] = max_hops
        config.update(kwargs)
        self._retriever_nodes.append({"type": "retriever", "config": config})
        return self

    def community_retriever(
        self,
        top_k: int = 3,
        max_hops: int = 1,
        vector_index_name: str = "community_embeddings",
        embedding_method: str = "sentence_transformer",
        model_name: str = "all-MiniLM-L6-v2",
        vector_store_type: str = None,
        store_config: BaseStoreConfig = None,
        **kwargs,
    ) -> "QueryConfigBuilder":
        vs = self._effective_vector_store()
        config = self._effective_store_flat(store_config, **kwargs)
        config.update({
            "top_k": top_k,
            "max_hops": max_hops,
            "vector_index_name": vector_index_name,
            "embedding_method": embedding_method,
            "model_name": model_name,
            "vector_store_type": vector_store_type or vs.get("vector_store_type", "in_memory"),
        })
        config.update(kwargs)
        self._retriever_nodes.append({"type": "community_retriever", "config": config})
        return self

    def chunk_embedding_retriever(
        self,
        top_k: int = 5,
        max_hops: int = 1,
        vector_index_name: str = "chunk_embeddings",
        embedding_method: str = "sentence_transformer",
        model_name: str = "all-MiniLM-L6-v2",
        vector_store_type: str = None,
        store_config: BaseStoreConfig = None,
        **kwargs,
    ) -> "QueryConfigBuilder":
        vs = self._effective_vector_store()
        config = self._effective_store_flat(store_config, **kwargs)
        config.update({
            "top_k": top_k,
            "max_hops": max_hops,
            "vector_index_name": vector_index_name,
            "embedding_method": embedding_method,
            "model_name": model_name,
            "vector_store_type": vector_store_type or vs.get("vector_store_type", "in_memory"),
        })
        config.update(kwargs)
        self._retriever_nodes.append({"type": "chunk_embedding_retriever", "config": config})
        return self

    def entity_embedding_retriever(
        self,
        top_k: int = 5,
        max_hops: int = 2,
        vector_index_name: str = "entity_embeddings",
        embedding_method: str = "sentence_transformer",
        model_name: str = "all-MiniLM-L6-v2",
        vector_store_type: str = None,
        store_config: BaseStoreConfig = None,
        **kwargs,
    ) -> "QueryConfigBuilder":
        vs = self._effective_vector_store()
        config = self._effective_store_flat(store_config, **kwargs)
        config.update({
            "top_k": top_k,
            "max_hops": max_hops,
            "vector_index_name": vector_index_name,
            "embedding_method": embedding_method,
            "model_name": model_name,
            "vector_store_type": vector_store_type or vs.get("vector_store_type", "in_memory"),
        })
        config.update(kwargs)
        self._retriever_nodes.append({"type": "entity_embedding_retriever", "config": config})
        return self

    def tool_retriever(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_completion_tokens: int = 4096,
        timeout: float = 60.0,
        entity_types: List[str] = None,
        relation_types: List[str] = None,
        max_tool_rounds: int = 3,
        chunk_source: str = "entity",
        store_config: BaseStoreConfig = None,
        **kwargs,
    ) -> "QueryConfigBuilder":
        config = self._effective_store_flat(store_config, **kwargs)
        config.update({
            "model": model,
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
            "timeout": timeout,
            "entity_types": entity_types or [],
            "relation_types": relation_types or [],
            "max_tool_rounds": max_tool_rounds,
            "chunk_source": chunk_source,
        })
        config.update(kwargs)
        if api_key is not None:
            config["api_key"] = api_key
        if base_url is not None:
            config["base_url"] = base_url
        self._retriever_nodes.append({"type": "tool_retriever", "config": config})
        return self

    def path_retriever(
        self,
        max_path_length: int = 5,
        top_k: int = 3,
        chunk_source: str = "entity",
        store_config: BaseStoreConfig = None,
        **kwargs,
    ) -> "QueryConfigBuilder":
        config = self._effective_store_flat(store_config, **kwargs)
        config.update({
            "max_path_length": max_path_length,
            "top_k": top_k,
            "chunk_source": chunk_source,
        })
        config.update(kwargs)
        self._retriever_nodes.append({"type": "path_retriever", "config": config})
        return self

    def keyword_retriever(
        self,
        top_k: int = 10,
        search_source: str = "relational",
        chunk_table: str = "chunks",
        relational_store_type: str = "sqlite",
        expand_entities: bool = None,
        max_hops: int = 1,
        fulltext_index: str = "chunk_text_idx",
        store_config: BaseStoreConfig = None,
        **kwargs,
    ) -> "QueryConfigBuilder":
        """Configure keyword (full-text search) retrieval strategy.

        Supports two search backends via ``search_source``:

        - ``"relational"`` (default) — searches a relational store (SQLite FTS5,
          PostgreSQL tsvector, Elasticsearch).  Requires a relational store.
        - ``"graph"`` — uses the graph database's native full-text index
          (Neo4j Lucene, FalkorDB FTS, Memgraph Tantivy).  The index is
          created automatically during ``setup_indexes()``.

        Set ``expand_entities=True`` to also look up entities in matched chunks
        and build a full subgraph via the graph store.  Defaults to ``False``
        for relational source, ``True`` for graph source.

        Args:
            top_k: Number of chunks to retrieve.
            search_source: ``"relational"`` or ``"graph"``.
            chunk_table: Table name for chunk records (relational source only).
            relational_store_type: Backend type (``"sqlite"``, ``"postgres"``,
                ``"elasticsearch"``).  Relational source only.
            expand_entities: Look up entities in matched chunks and build a
                subgraph via graph store.  Defaults to ``False`` for
                relational, ``True`` for graph.
            max_hops: Subgraph expansion depth when ``expand_entities=True``.
            fulltext_index: Name of the FTS index (graph source only,
                default: ``"chunk_text_idx"``).
            store_config: Explicit graph store config (used when
                ``expand_entities=True`` or ``search_source="graph"``).
            **kwargs: Additional config keys (e.g. ``sqlite_path``, graph store overrides).
        """
        # Resolve expand_entities default based on search source
        if expand_entities is None:
            expand_entities = search_source == "graph"
        needs_graph = expand_entities or search_source == "graph"
        if needs_graph:
            config = self._effective_store_flat(store_config, **kwargs)
        else:
            config = {}
        config.update({
            "top_k": top_k,
            "search_source": search_source,
            "chunk_table": chunk_table,
            "relational_store_type": relational_store_type,
            "expand_entities": expand_entities,
            "max_hops": max_hops,
            "fulltext_index": fulltext_index,
        })
        config.update(kwargs)
        self._retriever_nodes.append({"type": "keyword_retriever", "config": config})
        return self

    def fusion(
        self,
        strategy: str = "union",
        top_k: int = 0,
        weights: List[float] = None,
        min_sources: int = 2,
        rrf_k: int = 60,
    ) -> "QueryConfigBuilder":
        """Configure fusion of multiple retriever results.

        Automatically inserted when multiple retrievers are configured.
        Call explicitly to customize the fusion strategy.

        Args:
            strategy: "union", "rrf", "weighted", or "intersection".
            top_k: Max entities to keep (0 = all).
            weights: Per-retriever weights for "weighted" strategy.
            min_sources: Min retrievers for "intersection" strategy.
            rrf_k: RRF constant for "rrf" strategy.
        """
        self._fusion_config = {
            "strategy": strategy,
            "top_k": top_k,
            "rrf_k": rrf_k,
        }
        if weights:
            self._fusion_config["weights"] = weights
        if strategy == "intersection":
            self._fusion_config["min_sources"] = min_sources
        return self

    def chunk_retriever(
        self,
        max_chunks: int = 0,
        chunk_entity_source: str = "all",
        store_config: BaseStoreConfig = None,
        **kwargs,
    ) -> "QueryConfigBuilder":
        self._chunk_config = {"max_chunks": max_chunks, "chunk_entity_source": chunk_entity_source}
        if store_config is not None:
            self._chunk_config.update(store_config.to_flat_dict())
        store_kw = extract_store_kwargs(kwargs)
        if store_kw:
            self._chunk_config.update(store_kw)
        self._chunk_config.update(kwargs)
        return self

    def reasoner(
        self,
        method: str = "llm",
        api_key: str = None,
        base_url: str = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_completion_tokens: int = 4096,
        timeout: float = 60.0,
    ) -> "QueryConfigBuilder":
        self._reasoner_config = {
            "method": method,
            "model": model,
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
            "timeout": timeout,
        }
        if api_key is not None:
            self._reasoner_config["api_key"] = api_key
        if base_url is not None:
            self._reasoner_config["base_url"] = base_url
        return self

    def kg_scorer(
        self,
        model_path: str = None,
        model_name: str = "RotatE",
        embedding_dim: int = 128,
        top_k: int = 10,
        score_threshold: float = None,
        predict_tails: bool = True,
        predict_heads: bool = False,
        device: str = "cpu",
        store_config: BaseStoreConfig = None,
        **kwargs,
    ) -> "QueryConfigBuilder":
        self._kg_scorer_config = {
            "model_name": model_name,
            "embedding_dim": embedding_dim,
            "top_k": top_k,
            "predict_tails": predict_tails,
            "predict_heads": predict_heads,
            "device": device,
        }
        if model_path is not None:
            self._kg_scorer_config["model_path"] = model_path
        if score_threshold is not None:
            self._kg_scorer_config["score_threshold"] = score_threshold
        # Only inject store if explicitly provided at this level
        store = self._effective_store(store_config, **kwargs)
        if store_config is not None or extract_store_kwargs(dict(kwargs)):
            self._kg_scorer_config.update(store.to_flat_dict())
        self._kg_scorer_config.update(kwargs)
        return self

    def get_config(self) -> Dict[str, Any]:
        """Build configuration dict."""
        has_parser = self._parser_config is not None
        has_linker = self._has_linker

        # Check if this is a kg_scored pipeline (tool parser + kg_scorer, no separate retriever)
        is_kg_scored = (
            has_parser
            and self._parser_config.get("method") == "tool"
            and self._kg_scorer_config is not None
            and len(self._retriever_nodes) == 0
        )

        # Strategies that require entities from a parser
        _ENTITY_STRATEGIES = {"retriever", "entity_embedding_retriever", "path_retriever"}

        if not is_kg_scored:
            if len(self._retriever_nodes) > 0:
                first_type = self._retriever_nodes[0]["type"]
                needs_parser = first_type in _ENTITY_STRATEGIES
                if needs_parser and not has_parser and not has_linker:
                    raise ValueError(
                        f"Parser or linker config required for '{first_type}' strategy. "
                        "Call .query_parser() or .linker() first."
                    )
            else:
                raise ValueError("Retriever config required. Call a retriever method first.")

        # Default chunk retriever inherits store config
        if self._chunk_config is None:
            self._chunk_config = {}
        inherit_source = (
            self._retriever_nodes[0]["config"] if self._retriever_nodes
            else self._kg_scorer_config or {}
        )
        self._inherit_store(inherit_source, self._chunk_config)

        nodes = []

        # kg_scored pipeline: parser(tool) → kg_scorer → chunk_retriever → reasoner
        if is_kg_scored:
            nodes.append({
                "id": "query_parser",
                "processor": "query_parser",
                "inputs": {
                    "query": {"source": "$input", "fields": "query"},
                },
                "output": {"key": "parser_result"},
                "config": self._parser_config,
            })

            nodes.append({
                "id": "kg_scorer",
                "processor": "kg_scorer",
                "requires": ["query_parser"],
                "inputs": {
                    "triple_queries": {"source": "parser_result", "fields": "triple_queries"},
                },
                "output": {"key": "kg_scorer_result"},
                "config": self._kg_scorer_config,
            })

            chunk_inputs = {
                "subgraph": {"source": "kg_scorer_result", "fields": "subgraph"},
            }
            if self._chunk_config.get("chunk_entity_source") not in ("all", "both", None):
                chunk_inputs["scored_triples"] = {"source": "kg_scorer_result", "fields": "scored_triples"}

            nodes.append({
                "id": "chunk_retriever",
                "processor": "chunk_retriever",
                "requires": ["kg_scorer"],
                "inputs": chunk_inputs,
                "output": {"key": "chunk_result"},
                "config": self._chunk_config,
            })

            if self._reasoner_config is not None:
                nodes.append({
                    "id": "reasoner",
                    "processor": "reasoner",
                    "requires": ["chunk_retriever"],
                    "inputs": {
                        "query": {"source": "$input", "fields": "query"},
                        "subgraph": {"source": "chunk_result", "fields": "subgraph"},
                    },
                    "output": {"key": "reasoner_result"},
                    "config": self._reasoner_config,
                })

            return self._build_result(nodes)

        # Standard pipeline
        if has_parser:
            nodes.append({
                "id": "query_parser",
                "processor": "query_parser",
                "inputs": {
                    "query": {"source": "$input", "fields": "query"},
                },
                "output": {"key": "parser_result"},
                "config": self._parser_config,
            })

        if has_linker:
            linker_inputs = {"query": {"source": "$input", "fields": "query"}}
            linker_requires = []
            if has_parser:
                linker_inputs["entities"] = {"source": "parser_result", "fields": "entities"}
                linker_requires.append("query_parser")
            nodes.append({
                "id": "linker",
                "processor": "entity_linker",
                "requires": linker_requires,
                "inputs": linker_inputs,
                "output": {"key": "linker_result"},
                "config": self._linker_config,
            })

        # Determine entity source for entity-based strategies
        if has_linker:
            entity_source = "linker_result"
            entity_requires = ["linker"]
        elif has_parser:
            entity_source = "parser_result"
            entity_requires = ["query_parser"]
        else:
            entity_source = None
            entity_requires = []

        # Single retriever — backward compatible, same DAG as before
        if len(self._retriever_nodes) == 1:
            rnode = self._retriever_nodes[0]
            rtype = rnode["type"]
            rconfig = rnode["config"]

            if rtype in _ENTITY_STRATEGIES:
                nodes.append({
                    "id": "retriever",
                    "processor": rtype,
                    "requires": list(entity_requires),
                    "inputs": {
                        "entities": {"source": entity_source, "fields": "entities"},
                    },
                    "output": {"key": "retriever_result"},
                    "config": rconfig,
                })
            else:
                nodes.append({
                    "id": "retriever",
                    "processor": rtype,
                    "requires": [],
                    "inputs": {
                        "query": {"source": "$input", "fields": "query"},
                    },
                    "output": {"key": "retriever_result"},
                    "config": rconfig,
                })

            subgraph_source = "retriever_result"
            chunk_requires = ["retriever"]

        # Multiple retrievers — emit retriever_0..N + fusion node
        else:
            retriever_ids = []
            for idx, rnode in enumerate(self._retriever_nodes):
                rtype = rnode["type"]
                rconfig = rnode["config"]
                node_id = f"retriever_{idx}"
                output_key = f"retriever_result_{idx}"

                if rtype in _ENTITY_STRATEGIES:
                    nodes.append({
                        "id": node_id,
                        "processor": rtype,
                        "requires": list(entity_requires),
                        "inputs": {
                            "entities": {"source": entity_source, "fields": "entities"},
                        },
                        "output": {"key": output_key},
                        "config": rconfig,
                    })
                else:
                    nodes.append({
                        "id": node_id,
                        "processor": rtype,
                        "requires": [],
                        "inputs": {
                            "query": {"source": "$input", "fields": "query"},
                        },
                        "output": {"key": output_key},
                        "config": rconfig,
                    })
                retriever_ids.append(node_id)

            # Fusion node
            fusion_inputs = {}
            for idx in range(len(self._retriever_nodes)):
                fusion_inputs[f"subgraph_{idx}"] = {
                    "source": f"retriever_result_{idx}",
                    "fields": "subgraph",
                }
            fusion_config = self._fusion_config or {"strategy": "union"}
            nodes.append({
                "id": "fusion",
                "processor": "fusion",
                "requires": retriever_ids,
                "inputs": fusion_inputs,
                "output": {"key": "fusion_result"},
                "config": fusion_config,
            })

            subgraph_source = "fusion_result"
            chunk_requires = ["fusion"]

        nodes.append({
            "id": "chunk_retriever",
            "processor": "chunk_retriever",
            "requires": chunk_requires,
            "inputs": {
                "subgraph": {"source": subgraph_source, "fields": "subgraph"},
            },
            "output": {"key": "chunk_result"},
            "config": self._chunk_config,
        })

        if self._reasoner_config is not None:
            nodes.append({
                "id": "reasoner",
                "processor": "reasoner",
                "requires": ["chunk_retriever"],
                "inputs": {
                    "query": {"source": "$input", "fields": "query"},
                    "subgraph": {"source": "chunk_result", "fields": "subgraph"},
                },
                "output": {"key": "reasoner_result"},
                "config": self._reasoner_config,
            })

        if self._kg_scorer_config is not None:
            scorer_requires = chunk_requires[:1]  # depends on first retriever or fusion
            subgraph_src = subgraph_source
            if has_parser:
                scorer_inputs = {
                    "entities": {"source": "parser_result", "fields": "entities"},
                    "subgraph": {"source": subgraph_src, "fields": "subgraph"},
                }
            else:
                scorer_inputs = {
                    "subgraph": {"source": subgraph_src, "fields": "subgraph"},
                }
            nodes.append({
                "id": "kg_scorer",
                "processor": "kg_scorer",
                "requires": scorer_requires,
                "inputs": scorer_inputs,
                "output": {"key": "kg_scorer_result"},
                "config": self._kg_scorer_config,
            })

        return self._build_result(nodes)


class FusedQueryBuilder(_BuilderBase, _LinkerMixin):
    """Wrapper that combines multiple ``QueryConfigBuilder`` instances into a fused pipeline.

    Each sub-builder defines one retrieval strategy.  ``FusedQueryBuilder``
    collects their retriever nodes, merges store configs, and emits a single
    DAG with a fusion node.

    Usage::

        b1 = QueryConfigBuilder(name="entity")
        b1.query_parser(labels=["person", "location"])
        b1.retriever(neo4j_uri="bolt://localhost:7687", max_hops=2)

        b2 = QueryConfigBuilder(name="community")
        b2.community_retriever(neo4j_uri="bolt://localhost:7687")

        fused = FusedQueryBuilder(b1, b2, strategy="rrf", top_k=20)
        fused.chunk_retriever()
        fused.reasoner(api_key="sk-...", model="gpt-4o-mini")
        executor = fused.build()
        ctx = executor.execute({"query": "Where was Einstein born?"})
    """

    def __init__(
        self,
        *builders: QueryConfigBuilder,
        strategy: str = "union",
        top_k: int = 0,
        weights: List[float] = None,
        min_sources: int = 2,
        rrf_k: int = 60,
        name: str = "fused_query",
    ):
        super().__init__(name)
        self._sub_builders: List[QueryConfigBuilder] = list(builders)

        # Fusion config
        self._fusion_config: Dict[str, Any] = {
            "strategy": strategy,
            "top_k": top_k,
            "rrf_k": rrf_k,
        }
        if weights:
            self._fusion_config["weights"] = weights
        if strategy == "intersection":
            self._fusion_config["min_sources"] = min_sources

        # Own pipeline stages (override sub-builders)
        self._parser_config: Optional[Dict[str, Any]] = None
        self._linker_config: Optional[Dict[str, Any]] = None
        self._has_linker: bool = False
        self._chunk_config: Optional[Dict[str, Any]] = None
        self._reasoner_config: Optional[Dict[str, Any]] = None
        self._kg_scorer_config: Optional[Dict[str, Any]] = None

        # Inherit store config from first sub-builder that has one
        for b in self._sub_builders:
            if b._store_config is not None:
                self._store_config = b._store_config
                break

        # Merge named stores from all sub-builders
        for b in self._sub_builders:
            for k, v in b._graph_stores.items():
                if k not in self._graph_stores:
                    self._graph_stores[k] = v
            for k, v in b._vector_stores.items():
                if k not in self._vector_stores:
                    self._vector_stores[k] = v
            for k, v in b._relational_stores.items():
                if k not in self._relational_stores:
                    self._relational_stores[k] = v

    def query_parser(
        self,
        method: str = "gliner",
        model: str = None,
        labels: List[str] = None,
        relation_labels: List[str] = None,
        threshold: float = 0.3,
        device: str = "cpu",
        api_key: str = None,
        base_url: str = None,
        temperature: float = 0.1,
    ) -> "FusedQueryBuilder":
        """Override the auto-inherited parser from sub-builders."""
        self._parser_config = {
            "method": method,
            "labels": labels or [],
            "threshold": threshold,
            "device": device,
            "temperature": temperature,
        }
        if relation_labels is not None:
            self._parser_config["relation_labels"] = relation_labels
        if model is not None:
            self._parser_config["model"] = model
        if api_key is not None:
            self._parser_config["api_key"] = api_key
        if base_url is not None:
            self._parser_config["base_url"] = base_url
        return self

    def chunk_retriever(
        self,
        max_chunks: int = 0,
        chunk_entity_source: str = "all",
        store_config: BaseStoreConfig = None,
        **kwargs,
    ) -> "FusedQueryBuilder":
        self._chunk_config = {"max_chunks": max_chunks, "chunk_entity_source": chunk_entity_source}
        if store_config is not None:
            self._chunk_config.update(store_config.to_flat_dict())
        store_kw = extract_store_kwargs(kwargs)
        if store_kw:
            self._chunk_config.update(store_kw)
        self._chunk_config.update(kwargs)
        return self

    def reasoner(
        self,
        method: str = "llm",
        api_key: str = None,
        base_url: str = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_completion_tokens: int = 4096,
        timeout: float = 60.0,
    ) -> "FusedQueryBuilder":
        self._reasoner_config = {
            "method": method,
            "model": model,
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
            "timeout": timeout,
        }
        if api_key is not None:
            self._reasoner_config["api_key"] = api_key
        if base_url is not None:
            self._reasoner_config["base_url"] = base_url
        return self

    def fusion(
        self,
        strategy: str = "union",
        top_k: int = 0,
        weights: List[float] = None,
        min_sources: int = 2,
        rrf_k: int = 60,
    ) -> "FusedQueryBuilder":
        """Override fusion config after init."""
        self._fusion_config = {
            "strategy": strategy,
            "top_k": top_k,
            "rrf_k": rrf_k,
        }
        if weights:
            self._fusion_config["weights"] = weights
        if strategy == "intersection":
            self._fusion_config["min_sources"] = min_sources
        return self

    def get_config(self) -> Dict[str, Any]:
        """Build configuration dict by collecting retriever nodes from all sub-builders."""
        # Collect all retriever nodes from sub-builders
        all_retriever_nodes = [
            node
            for b in self._sub_builders
            for node in b._retriever_nodes
        ]
        if not all_retriever_nodes:
            raise ValueError(
                "No retriever nodes found in sub-builders. "
                "Each sub-builder should configure at least one retriever."
            )

        # Strategies that require entities from a parser
        _ENTITY_STRATEGIES = {"retriever", "entity_embedding_retriever", "path_retriever"}

        # Resolve parser: explicit > first sub-builder that has one
        parser_config = self._parser_config
        if parser_config is None:
            for b in self._sub_builders:
                if b._parser_config is not None:
                    parser_config = b._parser_config
                    break

        # Resolve linker: explicit > first sub-builder that has one
        has_linker = self._has_linker
        linker_config = self._linker_config
        if not has_linker:
            for b in self._sub_builders:
                if b._has_linker:
                    has_linker = True
                    linker_config = b._linker_config
                    break

        has_parser = parser_config is not None

        # Validate: entity-based strategies need parser or linker
        needs_parser = any(n["type"] in _ENTITY_STRATEGIES for n in all_retriever_nodes)
        if needs_parser and not has_parser and not has_linker:
            raise ValueError(
                "Parser or linker config required for entity-based retrieval strategies. "
                "Call .query_parser() on the FusedQueryBuilder or on a sub-builder."
            )

        # Default chunk retriever
        if self._chunk_config is None:
            self._chunk_config = {}
        # Inherit store keys from first retriever
        self._inherit_store(all_retriever_nodes[0]["config"], self._chunk_config)

        nodes: List[Dict[str, Any]] = []

        # Parser node
        if has_parser:
            nodes.append({
                "id": "query_parser",
                "processor": "query_parser",
                "inputs": {
                    "query": {"source": "$input", "fields": "query"},
                },
                "output": {"key": "parser_result"},
                "config": parser_config,
            })

        # Linker node
        if has_linker:
            linker_inputs = {"query": {"source": "$input", "fields": "query"}}
            linker_requires = []
            if has_parser:
                linker_inputs["entities"] = {"source": "parser_result", "fields": "entities"}
                linker_requires.append("query_parser")
            nodes.append({
                "id": "linker",
                "processor": "entity_linker",
                "requires": linker_requires,
                "inputs": linker_inputs,
                "output": {"key": "linker_result"},
                "config": linker_config,
            })

        # Determine entity source
        if has_linker:
            entity_source = "linker_result"
            entity_requires = ["linker"]
        elif has_parser:
            entity_source = "parser_result"
            entity_requires = ["query_parser"]
        else:
            entity_source = None
            entity_requires = []

        # Emit retriever nodes: retriever_0, retriever_1, ...
        retriever_ids = []
        for idx, rnode in enumerate(all_retriever_nodes):
            rtype = rnode["type"]
            rconfig = rnode["config"]
            node_id = f"retriever_{idx}"
            output_key = f"retriever_result_{idx}"

            if rtype in _ENTITY_STRATEGIES:
                nodes.append({
                    "id": node_id,
                    "processor": rtype,
                    "requires": list(entity_requires),
                    "inputs": {
                        "entities": {"source": entity_source, "fields": "entities"},
                    },
                    "output": {"key": output_key},
                    "config": rconfig,
                })
            else:
                nodes.append({
                    "id": node_id,
                    "processor": rtype,
                    "requires": [],
                    "inputs": {
                        "query": {"source": "$input", "fields": "query"},
                    },
                    "output": {"key": output_key},
                    "config": rconfig,
                })
            retriever_ids.append(node_id)

        # Fusion node (always emitted — user explicitly chose FusedQueryBuilder)
        fusion_inputs = {}
        for idx in range(len(all_retriever_nodes)):
            fusion_inputs[f"subgraph_{idx}"] = {
                "source": f"retriever_result_{idx}",
                "fields": "subgraph",
            }
        nodes.append({
            "id": "fusion",
            "processor": "fusion",
            "requires": retriever_ids,
            "inputs": fusion_inputs,
            "output": {"key": "fusion_result"},
            "config": self._fusion_config,
        })

        # Chunk retriever
        nodes.append({
            "id": "chunk_retriever",
            "processor": "chunk_retriever",
            "requires": ["fusion"],
            "inputs": {
                "subgraph": {"source": "fusion_result", "fields": "subgraph"},
            },
            "output": {"key": "chunk_result"},
            "config": self._chunk_config,
        })

        # Reasoner
        if self._reasoner_config is not None:
            nodes.append({
                "id": "reasoner",
                "processor": "reasoner",
                "requires": ["chunk_retriever"],
                "inputs": {
                    "query": {"source": "$input", "fields": "query"},
                    "subgraph": {"source": "chunk_result", "fields": "subgraph"},
                },
                "output": {"key": "reasoner_result"},
                "config": self._reasoner_config,
            })

        return self._build_result(nodes)


class CommunityConfigBuilder(_BuilderBase):
    """Declarative builder for community detection pipeline configs.

    Usage::

        builder = CommunityConfigBuilder(name="my_communities")
        builder.graph_store(Neo4jConfig(uri="bolt://localhost:7687"))
        builder.detector(method="louvain", levels=2)
        builder.summarizer(api_key="sk-...", model="gpt-4o-mini")
        builder.embedder(embedding_method="sentence_transformer")
        executor = builder.build()
        result = executor.execute({})
    """

    def __init__(self, name: str = "community_pipeline", description: str = None):
        super().__init__(name, description)
        self._detector_config: Optional[Dict[str, Any]] = None
        self._summarizer_config: Optional[Dict[str, Any]] = None
        self._embedder_config: Optional[Dict[str, Any]] = None

    def detector(
        self,
        method: str = "louvain",
        levels: int = 1,
        resolution: float = 1.0,
        store_config: BaseStoreConfig = None,
        **kwargs,
    ) -> "CommunityConfigBuilder":
        store = self._effective_store(store_config, **kwargs) or Neo4jConfig()
        self._detector_config = {
            "method": method,
            "levels": levels,
            "resolution": resolution,
        }
        self._detector_config.update(store.to_flat_dict())
        self._detector_config.update(kwargs)
        return self

    def summarizer(
        self,
        top_k: int = 10,
        api_key: str = None,
        base_url: str = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_completion_tokens: int = 4096,
        **store_overrides,
    ) -> "CommunityConfigBuilder":
        self._summarizer_config = {
            "top_k": top_k,
            "model": model,
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
        }
        if api_key is not None:
            self._summarizer_config["api_key"] = api_key
        if base_url is not None:
            self._summarizer_config["base_url"] = base_url
        self._summarizer_config.update(store_overrides)
        return self

    def embedder(
        self,
        embedding_method: str = "sentence_transformer",
        model_name: str = "all-MiniLM-L6-v2",
        vector_store_type: str = None,
        **extra,
    ) -> "CommunityConfigBuilder":
        vs = self._effective_vector_store()
        self._embedder_config = {
            "embedding_method": embedding_method,
            "model_name": model_name,
            "vector_store_type": vector_store_type or vs.get("vector_store_type", "in_memory"),
        }
        self._embedder_config.update(extra)
        return self

    def get_config(self) -> Dict[str, Any]:
        """Build configuration dict."""
        if not self._detector_config:
            raise ValueError("Detector config required. Call .detector() first.")

        nodes = [
            {
                "id": "community_detector",
                "processor": "community_detector",
                "inputs": {},
                "output": {"key": "detector_result"},
                "config": self._detector_config,
            },
        ]

        if self._summarizer_config is not None:
            summarizer_full = dict(self._summarizer_config)
            self._inherit_store(self._detector_config, summarizer_full)

            nodes.append({
                "id": "community_summarizer",
                "processor": "community_summarizer",
                "requires": ["community_detector"],
                "inputs": {},
                "output": {"key": "summarizer_result"},
                "config": summarizer_full,
            })

        if self._embedder_config is not None:
            embedder_full = dict(self._embedder_config)
            self._inherit_store(self._detector_config, embedder_full)

            embedder_requires = ["community_detector"]
            if self._summarizer_config is not None:
                embedder_requires.append("community_summarizer")

            nodes.append({
                "id": "community_embedder",
                "processor": "community_embedder",
                "requires": embedder_requires,
                "inputs": {},
                "output": {"key": "embedder_result"},
                "config": embedder_full,
            })

        return self._build_result(nodes)


class KGModelingConfigBuilder(_BuilderBase):
    """Declarative builder for KG embedding training pipelines.

    Usage::

        builder = KGModelingConfigBuilder(name="my_kg")
        builder.graph_store(Neo4jConfig(uri="bolt://localhost:7687"))
        builder.triple_reader()
        builder.trainer(model="RotatE", epochs=100)
        builder.storer(model_path="kg_model")
        executor = builder.build()
        result = executor.execute({})
    """

    def __init__(self, name: str = "kg_modeling_pipeline", description: str = None):
        super().__init__(name, description)
        self._reader_config: Optional[Dict[str, Any]] = None
        self._trainer_config: Optional[Dict[str, Any]] = None
        self._storer_config: Optional[Dict[str, Any]] = None

    def triple_reader(
        self,
        source: str = "graph_store",
        tsv_path: str = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42,
        store_config: BaseStoreConfig = None,
        **kwargs,
    ) -> "KGModelingConfigBuilder":
        store = self._effective_store(store_config, **kwargs) or Neo4jConfig()
        self._reader_config = {
            "source": source,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "random_seed": random_seed,
        }
        self._reader_config.update(store.to_flat_dict())
        self._reader_config.update(kwargs)
        if tsv_path is not None:
            self._reader_config["tsv_path"] = tsv_path
        return self

    def trainer(
        self,
        model: str = "RotatE",
        embedding_dim: int = 128,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 0.001,
        negative_sampler: str = "basic",
        num_negatives: int = 1,
        device: str = "cpu",
        use_tqdm: bool = True,
    ) -> "KGModelingConfigBuilder":
        self._trainer_config = {
            "model": model,
            "embedding_dim": embedding_dim,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "negative_sampler": negative_sampler,
            "num_negatives": num_negatives,
            "device": device,
            "use_tqdm": use_tqdm,
        }
        return self

    def storer(
        self,
        model_path: str = "kg_model",
        entity_index_name: str = "kg_entity_embeddings",
        relation_index_name: str = "kg_relation_embeddings",
        vector_store_type: str = None,
        store_to_graph: bool = False,
        **extra,
    ) -> "KGModelingConfigBuilder":
        vs = self._effective_vector_store()
        self._storer_config = {
            "model_path": model_path,
            "entity_index_name": entity_index_name,
            "relation_index_name": relation_index_name,
            "vector_store_type": vector_store_type or vs.get("vector_store_type", "in_memory"),
            "store_to_graph": store_to_graph,
        }
        self._storer_config.update(extra)
        return self

    def get_config(self) -> Dict[str, Any]:
        """Build configuration dict."""
        if not self._reader_config:
            raise ValueError("Triple reader config required. Call .triple_reader() first.")
        if not self._trainer_config:
            self._trainer_config = {}

        nodes = [
            {
                "id": "kg_triple_reader",
                "processor": "kg_triple_reader",
                "inputs": {},
                "output": {"key": "reader_result"},
                "config": self._reader_config,
            },
            {
                "id": "kg_trainer",
                "processor": "kg_trainer",
                "requires": ["kg_triple_reader"],
                "inputs": {
                    "training": {"source": "reader_result", "fields": "training"},
                    "validation": {"source": "reader_result", "fields": "validation"},
                    "testing": {"source": "reader_result", "fields": "testing"},
                },
                "output": {"key": "trainer_result"},
                "config": self._trainer_config,
            },
        ]

        if self._storer_config is not None:
            storer_full = dict(self._storer_config)
            self._inherit_store(self._reader_config, storer_full)

            nodes.append({
                "id": "kg_embedding_storer",
                "processor": "kg_embedding_storer",
                "requires": ["kg_trainer"],
                "inputs": {
                    "model": {"source": "trainer_result", "fields": "model"},
                    "entity_to_id": {"source": "reader_result", "fields": "entity_to_id"},
                    "relation_to_id": {"source": "reader_result", "fields": "relation_to_id"},
                },
                "output": {"key": "storer_result"},
                "config": storer_full,
            })

        return self._build_result(nodes)
