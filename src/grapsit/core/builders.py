"""Configuration builders for graph-building and query pipelines."""

from typing import Any, Dict, List, Optional

import yaml
from pathlib import Path

from .factory import ProcessorFactory
from .dag import DAGExecutor


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

    def graph_writer(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        neo4j_database: str = "neo4j",
        setup_indexes: bool = True,
    ) -> "BuildConfigBuilder":
        self._writer_config = {
            "neo4j_uri": neo4j_uri,
            "neo4j_user": neo4j_user,
            "neo4j_password": neo4j_password,
            "neo4j_database": neo4j_database,
            "setup_indexes": setup_indexes,
        }
        return self

    def get_config(self) -> Dict[str, Any]:
        """Build configuration dict."""
        if not self._chunker_config:
            self._chunker_config = {"method": "sentence"}
        if not self._ner_config and not self._relex_config:
            raise ValueError(
                "NER or relex config required. "
                "Call .ner_gliner() or .relex_gliner() first."
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

        if has_relex:
            relex_inputs = {
                "chunks": {"source": "chunker_result", "fields": "chunks"},
            }
            relex_requires = ["chunker"]
            if has_ner:
                relex_inputs["entities"] = {"source": "ner_result", "fields": "entities"}
                relex_requires.append("ner")
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
            # relex always produces entities (standalone or with pre-extracted)
            entity_source = "relex_result"
            writer_requires = ["chunker", "relex"]
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
        config = self.get_config()
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
