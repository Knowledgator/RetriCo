"""Pipeline factory — create DAG executors from YAML or dicts."""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .dag import DAGExecutor, DAGPipeline, InputConfig, OutputConfig, PipeNode


def load_yaml(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class ProcessorFactory:
    """Factory for creating pipelines from configs."""

    @staticmethod
    def create_pipeline(config_path: str | Path, verbose: bool = False) -> DAGExecutor:
        """Create DAG pipeline from a YAML config file."""
        config = load_yaml(config_path)
        return ProcessorFactory.create_from_dict(config, verbose=verbose)

    @staticmethod
    def create_from_dict(config_dict: dict, verbose: bool = False) -> DAGExecutor:
        """Create pipeline from a Python dict."""
        nodes = []
        for node_cfg in config_dict["nodes"]:
            inputs = {}
            for name, data in node_cfg.get("inputs", {}).items():
                inputs[name] = InputConfig(**data)

            node = PipeNode(
                id=node_cfg["id"],
                processor=node_cfg["processor"],
                inputs=inputs,
                output=OutputConfig(**node_cfg["output"]),
                requires=node_cfg.get("requires", []),
                config=node_cfg.get("config", {}),
                schema=node_cfg.get("schema"),
                condition=node_cfg.get("condition"),
            )
            nodes.append(node)

        pipeline = DAGPipeline(
            name=config_dict["name"],
            description=config_dict.get("description"),
            nodes=nodes,
        )
        return DAGExecutor(pipeline, verbose=verbose)
