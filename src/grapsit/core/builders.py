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
        self._chunk_config: Optional[Dict[str, Any]] = None
        self._reasoner_config: Optional[Dict[str, Any]] = None

    def query_parser(
        self,
        method: str = "gliner",
        model: str = None,
        labels: List[str] = None,
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

    def chunk_retriever(
        self,
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None,
        neo4j_database: str = None,
        max_chunks: int = 0,
        store_type: str = None,
        falkordb_host: str = None,
        falkordb_port: int = None,
        falkordb_graph: str = None,
        memgraph_uri: str = None,
        memgraph_user: str = None,
        memgraph_password: str = None,
        memgraph_database: str = None,
    ) -> "QueryConfigBuilder":
        self._chunk_config = {"max_chunks": max_chunks}
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

    def get_config(self) -> Dict[str, Any]:
        """Build configuration dict."""
        has_parser = self._parser_config is not None
        has_linker = self._has_linker
        if not has_parser and not has_linker:
            raise ValueError("Parser or linker config required. Call .query_parser() or .linker() first.")
        if not self._retriever_config:
            raise ValueError("Retriever config required. Call .retriever() first.")

        # Default chunk retriever inherits store config from retriever
        if self._chunk_config is None:
            self._chunk_config = {}
        inherit_keys = (
            "store_type", "neo4j_uri", "neo4j_user", "neo4j_password", "neo4j_database",
            "falkordb_host", "falkordb_port", "falkordb_graph",
            "memgraph_uri", "memgraph_user", "memgraph_password", "memgraph_database",
        )
        for key in inherit_keys:
            if key not in self._chunk_config and key in self._retriever_config:
                self._chunk_config[key] = self._retriever_config[key]

        nodes = []

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

        # Determine entity source for retriever
        if has_linker:
            entity_source = "linker_result"
            retriever_requires = ["linker"]
        else:
            entity_source = "parser_result"
            retriever_requires = ["query_parser"]

        nodes.append({
            "id": "retriever",
            "processor": "retriever",
            "requires": retriever_requires,
            "inputs": {
                "entities": {"source": entity_source, "fields": "entities"},
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
