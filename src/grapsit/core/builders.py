"""Configuration builders for graph-building and query pipelines."""

from typing import Any, Dict, List, Optional

import yaml
from pathlib import Path

from .factory import ProcessorFactory
from .dag import DAGExecutor

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


class BuildConfigBuilder:
    """Declarative builder for graph-building pipeline configs.

    Usage::

        builder = BuildConfigBuilder(name="my_graph")
        builder.chunker(method="sentence")
        builder.ner_gliner(model="urchade/gliner_multi-v2.1", labels=["person", "org"])
        builder.relex_gliner(
            model="knowledgator/gliner-relex-large-v0.5",
            entity_labels=["person", "org"],
            relation_labels=["works_at", "founded"],
        )
        builder.graph_writer(neo4j_uri="bolt://localhost:7687")

        # Get executor directly
        executor = builder.build()

        # Or save to YAML
        builder.save("configs/my_graph.yaml")
    """

    def __init__(self, name: str = "build_pipeline", description: str = None):
        self.name = name
        self.description = description or f"{name} — auto-generated"
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
        model: str = "urchade/gliner_multi-v2.1",
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

    def linker(
        self,
        executor: Any = None,
        model: str = "knowledgator/gliner-linker-large-v1.0",
        threshold: float = 0.5,
        entities: Any = None,
        store_type: str = None,
        neo4j_uri: str = None,
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        neo4j_database: str = "neo4j",
        falkordb_host: str = None,
        falkordb_port: int = 6379,
        falkordb_graph: str = "grapsit",
        memgraph_uri: str = None,
        memgraph_user: str = "",
        memgraph_password: str = "",
        memgraph_database: str = "memgraph",
    ) -> "BuildConfigBuilder":
        self._has_linker = True
        self._linker_config = {
            "model": model,
            "threshold": threshold,
        }
        if executor is not None:
            self._linker_config["executor"] = executor
        if entities is not None:
            self._linker_config["entities"] = entities
        if store_type is not None:
            self._linker_config["store_type"] = store_type
        if neo4j_uri is not None:
            self._linker_config["neo4j_uri"] = neo4j_uri
            self._linker_config["neo4j_user"] = neo4j_user
            self._linker_config["neo4j_password"] = neo4j_password
            self._linker_config["neo4j_database"] = neo4j_database
        if falkordb_host is not None:
            self._linker_config["falkordb_host"] = falkordb_host
            self._linker_config["falkordb_port"] = falkordb_port
            self._linker_config["falkordb_graph"] = falkordb_graph
        if memgraph_uri is not None:
            self._linker_config["memgraph_uri"] = memgraph_uri
            self._linker_config["memgraph_user"] = memgraph_user
            self._linker_config["memgraph_password"] = memgraph_password
            self._linker_config["memgraph_database"] = memgraph_database
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
        model: str = "gpt-4o-mini",
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

    def graph_writer(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        neo4j_database: str = "neo4j",
        setup_indexes: bool = True,
        store_type: str = "neo4j",
        falkordb_host: str = "localhost",
        falkordb_port: int = 6379,
        falkordb_graph: str = "grapsit",
        memgraph_uri: str = "bolt://localhost:7687",
        memgraph_user: str = "",
        memgraph_password: str = "",
        memgraph_database: str = "memgraph",
        json_output: str = None,
    ) -> "BuildConfigBuilder":
        self._writer_config = {
            "store_type": store_type,
            "neo4j_uri": neo4j_uri,
            "neo4j_user": neo4j_user,
            "neo4j_password": neo4j_password,
            "neo4j_database": neo4j_database,
            "setup_indexes": setup_indexes,
            "falkordb_host": falkordb_host,
            "falkordb_port": falkordb_port,
            "falkordb_graph": falkordb_graph,
            "memgraph_uri": memgraph_uri,
            "memgraph_user": memgraph_user,
            "memgraph_password": memgraph_password,
            "memgraph_database": memgraph_database,
        }
        if json_output is not None:
            self._writer_config["json_output"] = json_output
        return self

    def chunk_embedder(
        self,
        embedding_method: str = "sentence_transformer",
        model_name: str = "all-MiniLM-L6-v2",
        vector_store_type: str = "in_memory",
        vector_index_name: str = "chunk_embeddings",
        **extra,
    ) -> "BuildConfigBuilder":
        self._chunk_embedder_config = {
            "embedding_method": embedding_method,
            "model_name": model_name,
            "vector_store_type": vector_store_type,
            "vector_index_name": vector_index_name,
            **extra,
        }
        return self

    def entity_embedder(
        self,
        embedding_method: str = "sentence_transformer",
        model_name: str = "all-MiniLM-L6-v2",
        vector_store_type: str = "in_memory",
        vector_index_name: str = "entity_embeddings",
        **extra,
    ) -> "BuildConfigBuilder":
        self._entity_embedder_config = {
            "embedding_method": embedding_method,
            "model_name": model_name,
            "vector_store_type": vector_store_type,
            "vector_index_name": vector_index_name,
            **extra,
        }
        return self

    def get_config(self) -> Dict[str, Any]:
        """Build configuration dict."""
        if not self._chunker_config:
            self._chunker_config = {"method": "sentence"}
        if not self._ner_config and not self._relex_config and not self._has_linker:
            raise ValueError(
                "NER, linker, or relex config required. "
                "Call .ner_gliner()/.ner_llm(), .linker(), or .relex_gliner()/.relex_llm() first."
            )
        if not self._writer_config:
            self._writer_config = {}

        nodes = [
            {
                "id": "chunker",
                "processor": "chunker",
                "inputs": {
                    "texts": {"source": "$input", "fields": "texts"},
                },
                "output": {"key": "chunker_result"},
                "config": self._chunker_config,
            },
        ]

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
        # Priority: linker > ner (linker output has linked_entity_id)
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
        else:
            entity_source = "ner_result"
            writer_requires = ["chunker", "ner"]

        writer_inputs = {
            "chunks": {"source": "chunker_result", "fields": "chunks"},
            "documents": {"source": "chunker_result", "fields": "documents"},
            "entities": {"source": entity_source, "fields": "entities"},
        }

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
            # Inherit store params from writer config
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

        return {
            "name": self.name,
            "description": self.description,
            "nodes": nodes,
        }

    def build(self, verbose: bool = False) -> DAGExecutor:
        """Build and return a DAGExecutor."""
        return ProcessorFactory.create_from_dict(self.get_config(), verbose=verbose)

    def save(self, filepath: str) -> None:
        """Save config to YAML."""
        config = _strip_non_serializable(self.get_config())
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)


class IngestConfigBuilder:
    """Declarative builder for raw data ingest pipelines.

    Writes pre-structured entities and relations directly to the graph database,
    bypassing chunking, NER, and relation extraction.

    Usage::

        builder = IngestConfigBuilder(name="my_ingest")
        builder.graph_writer(neo4j_uri="bolt://localhost:7687")
        executor = builder.build()
        result = executor.execute({
            "entities": [
                {"text": "Einstein", "label": "person"},
                {"text": "Ulm", "label": "location"},
            ],
            "relations": [
                {"head": "Einstein", "tail": "Ulm", "type": "born_in"},
            ],
        })
    """

    def __init__(self, name: str = "ingest_pipeline", description: str = None):
        self.name = name
        self.description = description or f"{name} — auto-generated"
        self._writer_config: Optional[Dict[str, Any]] = None

    def graph_writer(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        neo4j_database: str = "neo4j",
        setup_indexes: bool = True,
        store_type: str = "neo4j",
        falkordb_host: str = "localhost",
        falkordb_port: int = 6379,
        falkordb_graph: str = "grapsit",
        memgraph_uri: str = "bolt://localhost:7687",
        memgraph_user: str = "",
        memgraph_password: str = "",
        memgraph_database: str = "memgraph",
        json_output: str = None,
    ) -> "IngestConfigBuilder":
        self._writer_config = {
            "store_type": store_type,
            "neo4j_uri": neo4j_uri,
            "neo4j_user": neo4j_user,
            "neo4j_password": neo4j_password,
            "neo4j_database": neo4j_database,
            "setup_indexes": setup_indexes,
            "falkordb_host": falkordb_host,
            "falkordb_port": falkordb_port,
            "falkordb_graph": falkordb_graph,
            "memgraph_uri": memgraph_uri,
            "memgraph_user": memgraph_user,
            "memgraph_password": memgraph_password,
            "memgraph_database": memgraph_database,
        }
        if json_output is not None:
            self._writer_config["json_output"] = json_output
        return self

    def get_config(self) -> Dict[str, Any]:
        """Build configuration dict."""
        if not self._writer_config:
            self._writer_config = {}

        nodes = [
            {
                "id": "data_ingest",
                "processor": "data_ingest",
                "inputs": {
                    "entities": {"source": "$input", "fields": "entities"},
                    "relations": {"source": "$input", "fields": "relations"},
                },
                "output": {"key": "ingest_result"},
                "config": {},
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

        return {
            "name": self.name,
            "description": self.description,
            "nodes": nodes,
        }

    def build(self, verbose: bool = False) -> DAGExecutor:
        """Build and return a DAGExecutor."""
        return ProcessorFactory.create_from_dict(self.get_config(), verbose=verbose)

    def save(self, filepath: str) -> None:
        """Save config to YAML."""
        config = _strip_non_serializable(self.get_config())
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)


class QueryConfigBuilder:
    """Declarative builder for query pipeline configs.

    Usage::

        builder = QueryConfigBuilder(name="my_query")
        builder.query_parser(method="gliner", labels=["person", "location"])
        builder.retriever(neo4j_uri="bolt://localhost:7687", max_hops=2)
        builder.chunk_retriever()
        builder.reasoner(api_key="...", model="gpt-4o-mini")  # optional

        executor = builder.build()
        result = executor.execute({"query": "Where was Einstein born?"})
    """

    def __init__(self, name: str = "query_pipeline", description: str = None):
        self.name = name
        self.description = description or f"{name} — auto-generated"
        self._parser_config: Optional[Dict[str, Any]] = None
        self._linker_config: Optional[Dict[str, Any]] = None
        self._has_linker: bool = False
        self._retriever_config: Optional[Dict[str, Any]] = None
        self._retriever_type: str = "retriever"
        self._chunk_config: Optional[Dict[str, Any]] = None
        self._reasoner_config: Optional[Dict[str, Any]] = None
        self._kg_scorer_config: Optional[Dict[str, Any]] = None

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

    def linker(
        self,
        executor: Any = None,
        model: str = "knowledgator/gliner-linker-large-v1.0",
        threshold: float = 0.5,
        entities: Any = None,
        neo4j_uri: str = None,
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        neo4j_database: str = "neo4j",
        store_type: str = None,
        falkordb_host: str = None,
        falkordb_port: int = 6379,
        falkordb_graph: str = "grapsit",
        memgraph_uri: str = None,
        memgraph_user: str = "",
        memgraph_password: str = "",
        memgraph_database: str = "memgraph",
    ) -> "QueryConfigBuilder":
        self._has_linker = True
        self._linker_config = {
            "model": model,
            "threshold": threshold,
        }
        if executor is not None:
            self._linker_config["executor"] = executor
        if entities is not None:
            self._linker_config["entities"] = entities
        if store_type is not None:
            self._linker_config["store_type"] = store_type
        if neo4j_uri is not None:
            self._linker_config["neo4j_uri"] = neo4j_uri
            self._linker_config["neo4j_user"] = neo4j_user
            self._linker_config["neo4j_password"] = neo4j_password
            self._linker_config["neo4j_database"] = neo4j_database
        if falkordb_host is not None:
            self._linker_config["falkordb_host"] = falkordb_host
            self._linker_config["falkordb_port"] = falkordb_port
            self._linker_config["falkordb_graph"] = falkordb_graph
        if memgraph_uri is not None:
            self._linker_config["memgraph_uri"] = memgraph_uri
            self._linker_config["memgraph_user"] = memgraph_user
            self._linker_config["memgraph_password"] = memgraph_password
            self._linker_config["memgraph_database"] = memgraph_database
        return self

    def retriever(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        neo4j_database: str = "neo4j",
        max_hops: int = 2,
        store_type: str = "neo4j",
        falkordb_host: str = "localhost",
        falkordb_port: int = 6379,
        falkordb_graph: str = "grapsit",
        memgraph_uri: str = "bolt://localhost:7687",
        memgraph_user: str = "",
        memgraph_password: str = "",
        memgraph_database: str = "memgraph",
    ) -> "QueryConfigBuilder":
        self._retriever_type = "retriever"
        self._retriever_config = {
            "store_type": store_type,
            "neo4j_uri": neo4j_uri,
            "neo4j_user": neo4j_user,
            "neo4j_password": neo4j_password,
            "neo4j_database": neo4j_database,
            "max_hops": max_hops,
            "falkordb_host": falkordb_host,
            "falkordb_port": falkordb_port,
            "falkordb_graph": falkordb_graph,
            "memgraph_uri": memgraph_uri,
            "memgraph_user": memgraph_user,
            "memgraph_password": memgraph_password,
            "memgraph_database": memgraph_database,
        }
        return self

    def community_retriever(
        self,
        top_k: int = 3,
        max_hops: int = 1,
        vector_index_name: str = "community_embeddings",
        embedding_method: str = "sentence_transformer",
        model_name: str = "all-MiniLM-L6-v2",
        vector_store_type: str = "in_memory",
        store_type: str = "neo4j",
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        neo4j_database: str = "neo4j",
        falkordb_host: str = "localhost",
        falkordb_port: int = 6379,
        falkordb_graph: str = "grapsit",
        memgraph_uri: str = "bolt://localhost:7687",
        memgraph_user: str = "",
        memgraph_password: str = "",
        memgraph_database: str = "memgraph",
    ) -> "QueryConfigBuilder":
        self._retriever_type = "community_retriever"
        self._retriever_config = {
            "top_k": top_k,
            "max_hops": max_hops,
            "vector_index_name": vector_index_name,
            "embedding_method": embedding_method,
            "model_name": model_name,
            "vector_store_type": vector_store_type,
            "store_type": store_type,
            "neo4j_uri": neo4j_uri,
            "neo4j_user": neo4j_user,
            "neo4j_password": neo4j_password,
            "neo4j_database": neo4j_database,
            "falkordb_host": falkordb_host,
            "falkordb_port": falkordb_port,
            "falkordb_graph": falkordb_graph,
            "memgraph_uri": memgraph_uri,
            "memgraph_user": memgraph_user,
            "memgraph_password": memgraph_password,
            "memgraph_database": memgraph_database,
        }
        return self

    def chunk_embedding_retriever(
        self,
        top_k: int = 5,
        max_hops: int = 1,
        vector_index_name: str = "chunk_embeddings",
        embedding_method: str = "sentence_transformer",
        model_name: str = "all-MiniLM-L6-v2",
        vector_store_type: str = "in_memory",
        store_type: str = "neo4j",
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        neo4j_database: str = "neo4j",
        falkordb_host: str = "localhost",
        falkordb_port: int = 6379,
        falkordb_graph: str = "grapsit",
        memgraph_uri: str = "bolt://localhost:7687",
        memgraph_user: str = "",
        memgraph_password: str = "",
        memgraph_database: str = "memgraph",
    ) -> "QueryConfigBuilder":
        self._retriever_type = "chunk_embedding_retriever"
        self._retriever_config = {
            "top_k": top_k,
            "max_hops": max_hops,
            "vector_index_name": vector_index_name,
            "embedding_method": embedding_method,
            "model_name": model_name,
            "vector_store_type": vector_store_type,
            "store_type": store_type,
            "neo4j_uri": neo4j_uri,
            "neo4j_user": neo4j_user,
            "neo4j_password": neo4j_password,
            "neo4j_database": neo4j_database,
            "falkordb_host": falkordb_host,
            "falkordb_port": falkordb_port,
            "falkordb_graph": falkordb_graph,
            "memgraph_uri": memgraph_uri,
            "memgraph_user": memgraph_user,
            "memgraph_password": memgraph_password,
            "memgraph_database": memgraph_database,
        }
        return self

    def entity_embedding_retriever(
        self,
        top_k: int = 5,
        max_hops: int = 2,
        vector_index_name: str = "entity_embeddings",
        embedding_method: str = "sentence_transformer",
        model_name: str = "all-MiniLM-L6-v2",
        vector_store_type: str = "in_memory",
        store_type: str = "neo4j",
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        neo4j_database: str = "neo4j",
        falkordb_host: str = "localhost",
        falkordb_port: int = 6379,
        falkordb_graph: str = "grapsit",
        memgraph_uri: str = "bolt://localhost:7687",
        memgraph_user: str = "",
        memgraph_password: str = "",
        memgraph_database: str = "memgraph",
    ) -> "QueryConfigBuilder":
        self._retriever_type = "entity_embedding_retriever"
        self._retriever_config = {
            "top_k": top_k,
            "max_hops": max_hops,
            "vector_index_name": vector_index_name,
            "embedding_method": embedding_method,
            "model_name": model_name,
            "vector_store_type": vector_store_type,
            "store_type": store_type,
            "neo4j_uri": neo4j_uri,
            "neo4j_user": neo4j_user,
            "neo4j_password": neo4j_password,
            "neo4j_database": neo4j_database,
            "falkordb_host": falkordb_host,
            "falkordb_port": falkordb_port,
            "falkordb_graph": falkordb_graph,
            "memgraph_uri": memgraph_uri,
            "memgraph_user": memgraph_user,
            "memgraph_password": memgraph_password,
            "memgraph_database": memgraph_database,
        }
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
        store_type: str = "neo4j",
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        neo4j_database: str = "neo4j",
        falkordb_host: str = "localhost",
        falkordb_port: int = 6379,
        falkordb_graph: str = "grapsit",
        memgraph_uri: str = "bolt://localhost:7687",
        memgraph_user: str = "",
        memgraph_password: str = "",
        memgraph_database: str = "memgraph",
    ) -> "QueryConfigBuilder":
        self._retriever_type = "tool_retriever"
        self._retriever_config = {
            "model": model,
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
            "timeout": timeout,
            "entity_types": entity_types or [],
            "relation_types": relation_types or [],
            "max_tool_rounds": max_tool_rounds,
            "store_type": store_type,
            "neo4j_uri": neo4j_uri,
            "neo4j_user": neo4j_user,
            "neo4j_password": neo4j_password,
            "neo4j_database": neo4j_database,
            "falkordb_host": falkordb_host,
            "falkordb_port": falkordb_port,
            "falkordb_graph": falkordb_graph,
            "memgraph_uri": memgraph_uri,
            "memgraph_user": memgraph_user,
            "memgraph_password": memgraph_password,
            "memgraph_database": memgraph_database,
        }
        if api_key is not None:
            self._retriever_config["api_key"] = api_key
        if base_url is not None:
            self._retriever_config["base_url"] = base_url
        return self

    def path_retriever(
        self,
        max_path_length: int = 5,
        max_pairs: int = 10,
        store_type: str = "neo4j",
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        neo4j_database: str = "neo4j",
        falkordb_host: str = "localhost",
        falkordb_port: int = 6379,
        falkordb_graph: str = "grapsit",
        memgraph_uri: str = "bolt://localhost:7687",
        memgraph_user: str = "",
        memgraph_password: str = "",
        memgraph_database: str = "memgraph",
    ) -> "QueryConfigBuilder":
        self._retriever_type = "path_retriever"
        self._retriever_config = {
            "max_path_length": max_path_length,
            "max_pairs": max_pairs,
            "store_type": store_type,
            "neo4j_uri": neo4j_uri,
            "neo4j_user": neo4j_user,
            "neo4j_password": neo4j_password,
            "neo4j_database": neo4j_database,
            "falkordb_host": falkordb_host,
            "falkordb_port": falkordb_port,
            "falkordb_graph": falkordb_graph,
            "memgraph_uri": memgraph_uri,
            "memgraph_user": memgraph_user,
            "memgraph_password": memgraph_password,
            "memgraph_database": memgraph_database,
        }
        return self

    def chunk_retriever(
        self,
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None,
        neo4j_database: str = None,
        max_chunks: int = 0,
        chunk_entity_source: str = "all",
        store_type: str = None,
        falkordb_host: str = None,
        falkordb_port: int = None,
        falkordb_graph: str = None,
        memgraph_uri: str = None,
        memgraph_user: str = None,
        memgraph_password: str = None,
        memgraph_database: str = None,
    ) -> "QueryConfigBuilder":
        self._chunk_config = {"max_chunks": max_chunks, "chunk_entity_source": chunk_entity_source}
        # Inherit store config from retriever if not explicitly provided
        if neo4j_uri is not None:
            self._chunk_config["neo4j_uri"] = neo4j_uri
        if neo4j_user is not None:
            self._chunk_config["neo4j_user"] = neo4j_user
        if neo4j_password is not None:
            self._chunk_config["neo4j_password"] = neo4j_password
        if neo4j_database is not None:
            self._chunk_config["neo4j_database"] = neo4j_database
        if store_type is not None:
            self._chunk_config["store_type"] = store_type
        if falkordb_host is not None:
            self._chunk_config["falkordb_host"] = falkordb_host
        if falkordb_port is not None:
            self._chunk_config["falkordb_port"] = falkordb_port
        if falkordb_graph is not None:
            self._chunk_config["falkordb_graph"] = falkordb_graph
        if memgraph_uri is not None:
            self._chunk_config["memgraph_uri"] = memgraph_uri
        if memgraph_user is not None:
            self._chunk_config["memgraph_user"] = memgraph_user
        if memgraph_password is not None:
            self._chunk_config["memgraph_password"] = memgraph_password
        if memgraph_database is not None:
            self._chunk_config["memgraph_database"] = memgraph_database
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
        store_type: str = None,
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None,
        neo4j_database: str = None,
        falkordb_host: str = None,
        falkordb_port: int = None,
        falkordb_graph: str = None,
        memgraph_uri: str = None,
        memgraph_user: str = None,
        memgraph_password: str = None,
        memgraph_database: str = None,
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
        if store_type is not None:
            self._kg_scorer_config["store_type"] = store_type
        if neo4j_uri is not None:
            self._kg_scorer_config["neo4j_uri"] = neo4j_uri
        if neo4j_user is not None:
            self._kg_scorer_config["neo4j_user"] = neo4j_user
        if neo4j_password is not None:
            self._kg_scorer_config["neo4j_password"] = neo4j_password
        if neo4j_database is not None:
            self._kg_scorer_config["neo4j_database"] = neo4j_database
        if falkordb_host is not None:
            self._kg_scorer_config["falkordb_host"] = falkordb_host
        if falkordb_port is not None:
            self._kg_scorer_config["falkordb_port"] = falkordb_port
        if falkordb_graph is not None:
            self._kg_scorer_config["falkordb_graph"] = falkordb_graph
        if memgraph_uri is not None:
            self._kg_scorer_config["memgraph_uri"] = memgraph_uri
        if memgraph_user is not None:
            self._kg_scorer_config["memgraph_user"] = memgraph_user
        if memgraph_password is not None:
            self._kg_scorer_config["memgraph_password"] = memgraph_password
        if memgraph_database is not None:
            self._kg_scorer_config["memgraph_database"] = memgraph_database
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
            and self._retriever_config is None
        )

        # Strategies that require entities from a parser
        _ENTITY_STRATEGIES = {"retriever", "entity_embedding_retriever", "path_retriever"}
        # Strategies that take query directly (no parser needed)
        _QUERY_STRATEGIES = {"community_retriever", "chunk_embedding_retriever", "tool_retriever"}

        if not is_kg_scored:
            needs_parser = self._retriever_type in _ENTITY_STRATEGIES

            if needs_parser and not has_parser and not has_linker:
                raise ValueError(
                    f"Parser or linker config required for '{self._retriever_type}' strategy. "
                    "Call .query_parser() or .linker() first."
                )
            if not self._retriever_config:
                raise ValueError("Retriever config required. Call a retriever method first.")

        # Default chunk retriever inherits store config
        if self._chunk_config is None:
            self._chunk_config = {}
        inherit_keys = (
            "store_type", "neo4j_uri", "neo4j_user", "neo4j_password", "neo4j_database",
            "falkordb_host", "falkordb_port", "falkordb_graph",
            "memgraph_uri", "memgraph_user", "memgraph_password", "memgraph_database",
        )
        # Inherit from retriever config or kg_scorer config
        inherit_source = self._retriever_config or self._kg_scorer_config or {}
        for key in inherit_keys:
            if key not in self._chunk_config and key in inherit_source:
                self._chunk_config[key] = inherit_source[key]

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

            return {
                "name": self.name,
                "description": self.description,
                "nodes": nodes,
            }

        # Standard pipeline
        # Add parser if configured (needed for entity-based strategies)
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

        # Build retriever node based on strategy type
        if self._retriever_type in _ENTITY_STRATEGIES:
            # Entity-based retriever: needs entities from parser/linker
            if has_linker:
                entity_source = "linker_result"
                retriever_requires = ["linker"]
            else:
                entity_source = "parser_result"
                retriever_requires = ["query_parser"]

            nodes.append({
                "id": "retriever",
                "processor": self._retriever_type,
                "requires": retriever_requires,
                "inputs": {
                    "entities": {"source": entity_source, "fields": "entities"},
                },
                "output": {"key": "retriever_result"},
                "config": self._retriever_config,
            })
        else:
            # Query-based retriever: takes query directly
            nodes.append({
                "id": "retriever",
                "processor": self._retriever_type,
                "requires": [],
                "inputs": {
                    "query": {"source": "$input", "fields": "query"},
                },
                "output": {"key": "retriever_result"},
                "config": self._retriever_config,
            })

        nodes.append({
            "id": "chunk_retriever",
            "processor": "chunk_retriever",
            "requires": ["retriever"],
            "inputs": {
                "subgraph": {"source": "retriever_result", "fields": "subgraph"},
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
            # Determine subgraph source and dependencies
            scorer_requires = ["retriever"]
            subgraph_source = "retriever_result"
            if has_parser:
                scorer_inputs = {
                    "entities": {"source": "parser_result", "fields": "entities"},
                    "subgraph": {"source": subgraph_source, "fields": "subgraph"},
                }
            else:
                scorer_inputs = {
                    "subgraph": {"source": subgraph_source, "fields": "subgraph"},
                }
            nodes.append({
                "id": "kg_scorer",
                "processor": "kg_scorer",
                "requires": scorer_requires,
                "inputs": scorer_inputs,
                "output": {"key": "kg_scorer_result"},
                "config": self._kg_scorer_config,
            })

        return {
            "name": self.name,
            "description": self.description,
            "nodes": nodes,
        }

    def build(self, verbose: bool = False) -> DAGExecutor:
        """Build and return a DAGExecutor."""
        return ProcessorFactory.create_from_dict(self.get_config(), verbose=verbose)

    def save(self, filepath: str) -> None:
        """Save config to YAML."""
        config = _strip_non_serializable(self.get_config())
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)


class CommunityConfigBuilder:
    """Declarative builder for community detection pipeline configs.

    Detects communities in an existing knowledge graph, optionally generates
    LLM summaries, and optionally embeds community summaries.

    Usage::

        builder = CommunityConfigBuilder(name="my_communities")
        builder.detector(method="louvain", levels=2, neo4j_uri="bolt://localhost:7687")
        builder.summarizer(api_key="sk-...", model="gpt-4o-mini")
        builder.embedder(embedding_method="sentence_transformer")
        executor = builder.build()
        result = executor.execute({})
    """

    def __init__(self, name: str = "community_pipeline", description: str = None):
        self.name = name
        self.description = description or f"{name} — auto-generated"
        self._detector_config: Optional[Dict[str, Any]] = None
        self._summarizer_config: Optional[Dict[str, Any]] = None
        self._embedder_config: Optional[Dict[str, Any]] = None

    def detector(
        self,
        method: str = "louvain",
        levels: int = 1,
        resolution: float = 1.0,
        store_type: str = "neo4j",
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        neo4j_database: str = "neo4j",
        falkordb_host: str = "localhost",
        falkordb_port: int = 6379,
        falkordb_graph: str = "grapsit",
        memgraph_uri: str = "bolt://localhost:7687",
        memgraph_user: str = "",
        memgraph_password: str = "",
        memgraph_database: str = "memgraph",
    ) -> "CommunityConfigBuilder":
        self._detector_config = {
            "method": method,
            "levels": levels,
            "resolution": resolution,
            "store_type": store_type,
            "neo4j_uri": neo4j_uri,
            "neo4j_user": neo4j_user,
            "neo4j_password": neo4j_password,
            "neo4j_database": neo4j_database,
            "falkordb_host": falkordb_host,
            "falkordb_port": falkordb_port,
            "falkordb_graph": falkordb_graph,
            "memgraph_uri": memgraph_uri,
            "memgraph_user": memgraph_user,
            "memgraph_password": memgraph_password,
            "memgraph_database": memgraph_database,
        }
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
        vector_store_type: str = "in_memory",
        **extra,
    ) -> "CommunityConfigBuilder":
        self._embedder_config = {
            "embedding_method": embedding_method,
            "model_name": model_name,
            "vector_store_type": vector_store_type,
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

        # Inherit store params from detector for downstream processors
        store_keys = (
            "store_type", "neo4j_uri", "neo4j_user", "neo4j_password", "neo4j_database",
            "falkordb_host", "falkordb_port", "falkordb_graph",
            "memgraph_uri", "memgraph_user", "memgraph_password", "memgraph_database",
        )

        if self._summarizer_config is not None:
            summarizer_full = dict(self._summarizer_config)
            for key in store_keys:
                if key not in summarizer_full and key in self._detector_config:
                    summarizer_full[key] = self._detector_config[key]

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
            for key in store_keys:
                if key not in embedder_full and key in self._detector_config:
                    embedder_full[key] = self._detector_config[key]

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

        return {
            "name": self.name,
            "description": self.description,
            "nodes": nodes,
        }

    def build(self, verbose: bool = False) -> DAGExecutor:
        """Build and return a DAGExecutor."""
        return ProcessorFactory.create_from_dict(self.get_config(), verbose=verbose)

    def save(self, filepath: str) -> None:
        """Save config to YAML."""
        config = _strip_non_serializable(self.get_config())
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)


class KGModelingConfigBuilder:
    """Declarative builder for KG embedding training pipelines.

    Reads triples, trains a PyKEEN model, and stores embeddings.

    Usage::

        builder = KGModelingConfigBuilder(name="my_kg")
        builder.triple_reader(neo4j_uri="bolt://localhost:7687")
        builder.trainer(model="RotatE", epochs=100)
        builder.storer(model_path="kg_model")
        executor = builder.build()
        result = executor.execute({})
    """

    def __init__(self, name: str = "kg_modeling_pipeline", description: str = None):
        self.name = name
        self.description = description or f"{name} — auto-generated"
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
        store_type: str = "neo4j",
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        neo4j_database: str = "neo4j",
        falkordb_host: str = "localhost",
        falkordb_port: int = 6379,
        falkordb_graph: str = "grapsit",
        memgraph_uri: str = "bolt://localhost:7687",
        memgraph_user: str = "",
        memgraph_password: str = "",
        memgraph_database: str = "memgraph",
    ) -> "KGModelingConfigBuilder":
        self._reader_config = {
            "source": source,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "random_seed": random_seed,
            "store_type": store_type,
            "neo4j_uri": neo4j_uri,
            "neo4j_user": neo4j_user,
            "neo4j_password": neo4j_password,
            "neo4j_database": neo4j_database,
            "falkordb_host": falkordb_host,
            "falkordb_port": falkordb_port,
            "falkordb_graph": falkordb_graph,
            "memgraph_uri": memgraph_uri,
            "memgraph_user": memgraph_user,
            "memgraph_password": memgraph_password,
            "memgraph_database": memgraph_database,
        }
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
        vector_store_type: str = "in_memory",
        store_to_graph: bool = False,
        **extra,
    ) -> "KGModelingConfigBuilder":
        self._storer_config = {
            "model_path": model_path,
            "entity_index_name": entity_index_name,
            "relation_index_name": relation_index_name,
            "vector_store_type": vector_store_type,
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
            # Inherit store params from reader for graph DB writes
            store_keys = (
                "store_type", "neo4j_uri", "neo4j_user", "neo4j_password", "neo4j_database",
                "falkordb_host", "falkordb_port", "falkordb_graph",
                "memgraph_uri", "memgraph_user", "memgraph_password", "memgraph_database",
            )
            storer_full = dict(self._storer_config)
            for key in store_keys:
                if key not in storer_full and key in self._reader_config:
                    storer_full[key] = self._reader_config[key]

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

        return {
            "name": self.name,
            "description": self.description,
            "nodes": nodes,
        }

    def build(self, verbose: bool = False) -> DAGExecutor:
        """Build and return a DAGExecutor."""
        return ProcessorFactory.create_from_dict(self.get_config(), verbose=verbose)

    def save(self, filepath: str) -> None:
        """Save config to YAML."""
        config = _strip_non_serializable(self.get_config())
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
