"""DAG pipeline engine — adapted from GLinker."""

from typing import Dict, List, Set, Any, Optional, Literal, Union
from collections import defaultdict, deque, OrderedDict
from pydantic import BaseModel, Field
from datetime import datetime
from pathlib import Path
import re
import json
import logging
import time

logger = logging.getLogger(__name__)


# ============================================================================
# INPUT/OUTPUT CONFIG
# ============================================================================


class ReshapeConfig(BaseModel):
    """Configuration for data reshaping."""
    by: str = Field(..., description="Reference structure path: 'l1_result.entities'")
    mode: Literal["flatten_per_group", "preserve_structure"] = Field(
        "flatten_per_group",
        description="Reshape mode",
    )


class InputConfig(BaseModel):
    """Unified input data specification."""
    source: str = Field(
        ...,
        description="Data source: key ('l1_result'), index ('outputs[-1]'), or '$input'",
    )
    fields: Union[str, List[str], None] = Field(
        None,
        description="JSONPath fields: 'entities[*].text' or ['label', 'score']",
    )
    reduce: Literal["all", "first", "last", "flatten"] = Field(
        "all",
        description="Reduction mode for lists",
    )
    reshape: Optional[ReshapeConfig] = Field(None)
    template: Optional[str] = Field(None)
    filter: Optional[str] = Field(None)
    default: Any = None


class OutputConfig(BaseModel):
    """Output specification."""
    key: str = Field(..., description="Key for storing in context")
    fields: Union[str, List[str], None] = Field(None)


# ============================================================================
# PIPE NODE
# ============================================================================


class PipeNode(BaseModel):
    """Single node in DAG pipeline."""
    id: str = Field(..., description="Unique node identifier")
    processor: str = Field(..., description="Processor name from registry")
    inputs: Dict[str, InputConfig] = Field(default_factory=dict)
    output: OutputConfig = Field(..., description="Output specification")
    requires: List[str] = Field(default_factory=list)
    config: Dict[str, Any] = Field(default_factory=dict)
    schema_: Optional[Dict[str, Any]] = Field(None, alias="schema")
    condition: Optional[str] = Field(None)

    model_config = {"populate_by_name": True}


# ============================================================================
# PIPE CONTEXT
# ============================================================================


class PipeContext:
    """Pipeline execution context — stores all outputs."""

    def __init__(self, pipeline_input: Any = None):
        self._outputs: OrderedDict[str, Any] = OrderedDict()
        self._execution_order: List[str] = []
        self._pipeline_input = pipeline_input
        self._metadata: Dict[str, Any] = {}

    def set(self, key: str, value: Any, metadata: Optional[Dict] = None):
        self._outputs[key] = value
        self._execution_order.append(key)
        if metadata:
            self._metadata[key] = metadata

    def get(self, source: str) -> Any:
        if source == "$input":
            return self._pipeline_input
        if source.startswith("outputs["):
            index_str = source.replace("outputs[", "").replace("]", "")
            index = int(index_str)
            if index < 0:
                index = len(self._execution_order) + index
            if 0 <= index < len(self._execution_order):
                key = self._execution_order[index]
                return self._outputs[key]
            return None
        return self._outputs.get(source)

    def has(self, key: str) -> bool:
        return key in self._outputs

    def get_all_outputs(self) -> Dict[str, Any]:
        return dict(self._outputs)

    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        return self._metadata.get(key)

    def get_execution_order(self) -> List[str]:
        return list(self._execution_order)

    @property
    def data(self) -> Dict[str, Any]:
        return dict(self._outputs)

    def to_dict(self) -> Dict[str, Any]:
        def serialize(value):
            if hasattr(value, "model_dump"):
                return {"__type__": "pydantic", "data": value.model_dump()}
            elif hasattr(value, "dict"):
                return {"__type__": "pydantic", "data": value.dict()}
            elif isinstance(value, list):
                return [serialize(item) for item in value]
            elif isinstance(value, dict):
                return {k: serialize(v) for k, v in value.items()}
            elif isinstance(value, (str, int, float, bool, type(None))):
                return value
            else:
                return {"__type__": "object", "repr": repr(value)}

        return {
            "outputs": {k: serialize(v) for k, v in self._outputs.items()},
            "execution_order": self._execution_order,
            "pipeline_input": serialize(self._pipeline_input),
            "metadata": self._metadata,
            "saved_at": datetime.now().isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipeContext":
        def deserialize(value):
            if isinstance(value, dict):
                if "__type__" in value:
                    if value["__type__"] == "pydantic":
                        return value["data"]
                    elif value["__type__"] == "object":
                        return value["repr"]
                return {k: deserialize(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [deserialize(item) for item in value]
            return value

        pipeline_input = deserialize(data.get("pipeline_input"))
        context = cls(pipeline_input)
        for key, value in data.get("outputs", {}).items():
            context._outputs[key] = deserialize(value)
        context._execution_order = data.get("execution_order", [])
        context._metadata = data.get("metadata", {})
        return context

    def to_json(self, filepath: str = None, indent: int = 2) -> str:
        data = self.to_dict()
        json_str = json.dumps(data, indent=indent, ensure_ascii=False)
        if filepath:
            Path(filepath).write_text(json_str, encoding="utf-8")
        return json_str

    @classmethod
    def from_json(cls, json_data: str = None, filepath: str = None) -> "PipeContext":
        if filepath:
            json_data = Path(filepath).read_text(encoding="utf-8")
        if not json_data:
            raise ValueError("Either json_data or filepath must be provided")
        return cls.from_dict(json.loads(json_data))


# ============================================================================
# FIELD RESOLVER
# ============================================================================


class FieldResolver:
    """Resolve fields from data using path expressions."""

    @staticmethod
    def resolve(context: PipeContext, config: InputConfig) -> Any:
        data = context.get(config.source)
        if data is None:
            return config.default
        if config.fields:
            if isinstance(config.fields, str):
                data = FieldResolver._extract_fields(data, config.fields)
            else:
                # Multiple fields → dict
                result = {}
                for f in config.fields:
                    result[f] = FieldResolver._extract_fields(data, f)
                data = result
        if config.template:
            data = FieldResolver._format_template(data, config.template)
        if isinstance(data, list) and config.reduce and config.reduce != "all":
            data = FieldResolver._apply_reduce(data, config.reduce)
        if config.filter:
            data = FieldResolver._apply_filter(data, config.filter)
        return data

    @staticmethod
    def _extract_fields(data: Any, path: str) -> Any:
        parts = path.split(".")
        current = data
        for part in parts:
            if current is None:
                return None
            if "[" in part:
                current = FieldResolver._handle_brackets(current, part)
            else:
                current = FieldResolver._get_attr(current, part)
        return current

    @staticmethod
    def _handle_brackets(data: Any, part: str) -> Any:
        if part.startswith("["):
            field_name = None
            brackets = part
        else:
            bracket_idx = part.index("[")
            field_name = part[:bracket_idx]
            brackets = part[bracket_idx:]

        current = data
        if field_name:
            current = FieldResolver._get_attr(current, field_name)
        if current is None:
            return None

        while "[" in brackets:
            start = brackets.index("[")
            end = brackets.index("]")
            content = brackets[start + 1 : end]
            brackets = brackets[end + 1 :]

            if content == "*":
                if not isinstance(current, list):
                    current = [current]
            elif ":" in content:
                s_str, e_str = content.split(":")
                s = int(s_str) if s_str else None
                e = int(e_str) if e_str else None
                current = current[s:e]
            else:
                current = current[int(content)]
        return current

    @staticmethod
    def _get_attr(data: Any, field: str) -> Any:
        if isinstance(data, list):
            return [FieldResolver._get_attr(item, field) for item in data]
        if isinstance(data, dict):
            return data.get(field)
        return getattr(data, field, None)

    @staticmethod
    def _format_template(data, template):
        if isinstance(data, list):
            results = []
            for item in data:
                try:
                    if hasattr(item, "model_dump"):
                        results.append(template.format(**item.model_dump()))
                    elif isinstance(item, dict):
                        results.append(template.format(**item))
                    else:
                        results.append(str(item))
                except Exception:
                    results.append(str(item))
            return results
        try:
            if hasattr(data, "model_dump"):
                return template.format(**data.model_dump())
            elif isinstance(data, dict):
                return template.format(**data)
            return str(data)
        except Exception:
            return str(data)

    @staticmethod
    def _apply_reduce(data: list, mode: str) -> Any:
        if mode == "first":
            return data[0] if data else None
        elif mode == "last":
            return data[-1] if data else None
        elif mode == "flatten":
            def flatten(lst):
                result = []
                for item in lst:
                    if isinstance(item, list):
                        result.extend(flatten(item))
                    else:
                        result.append(item)
                return result
            return flatten(data)
        return data

    @staticmethod
    def _apply_filter(data, filter_expr: str):
        if not isinstance(data, list):
            return data
        match = re.match(r"(\w+)\s*(>=|<=|>|<|==|!=)\s*(.+)", filter_expr)
        if not match:
            return data
        field, operator, value = match.groups()
        try:
            if value.startswith(("'", '"')):
                value = value.strip("'\"")
            elif "." in value:
                value = float(value)
            else:
                value = int(value)
        except Exception:
            pass

        ops = {
            ">": lambda a, b: a > b,
            ">=": lambda a, b: a >= b,
            "<": lambda a, b: a < b,
            "<=": lambda a, b: a <= b,
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
        }
        op_fn = ops[operator]
        result = []
        for item in data:
            try:
                item_value = item.get(field) if isinstance(item, dict) else getattr(item, field, None)
                if item_value is not None and op_fn(item_value, value):
                    result.append(item)
            except Exception:
                continue
        return result


# ============================================================================
# DAG EXECUTOR
# ============================================================================


class DAGPipeline(BaseModel):
    """DAG pipeline configuration."""
    name: str
    nodes: List[PipeNode]
    description: Optional[str] = None


class DAGExecutor:
    """Executes DAG pipeline with topological sort."""

    def __init__(self, pipeline: DAGPipeline, verbose: bool = False, store_pool=None):
        self.pipeline = pipeline
        self.verbose = verbose
        self.store_pool = store_pool
        self.nodes_map: Dict[str, PipeNode] = {}
        self.dependency_graph: Dict[str, List[str]] = defaultdict(list)
        self.reverse_graph: Dict[str, Set[str]] = defaultdict(set)
        self.processors: Dict[str, Any] = {}
        self._build_dependency_graph()
        self._initialize_processors()

    def _build_dependency_graph(self):
        for node in self.pipeline.nodes:
            self.nodes_map[node.id] = node

        for node in self.pipeline.nodes:
            for dep_id in node.requires:
                self.dependency_graph[dep_id].append(node.id)
                self.reverse_graph[node.id].add(dep_id)

            for input_config in node.inputs.values():
                source = input_config.source
                if source == "$input" or source.startswith("outputs["):
                    continue
                if source in self.nodes_map:
                    self.dependency_graph[source].append(node.id)
                    self.reverse_graph[node.id].add(source)

    def _initialize_processors(self):
        from .registry import processor_registry

        for node_id, node in self.nodes_map.items():
            try:
                config = dict(node.config)
                # Inject store pool into every processor's config
                if self.store_pool is not None:
                    config["__store_pool__"] = self.store_pool
                factory = processor_registry.get(node.processor)
                processor = factory(config_dict=config, pipeline=None)
                self.processors[node_id] = processor
            except Exception as e:
                raise RuntimeError(
                    f"Failed to create processor '{node.processor}' for node '{node_id}': {e}"
                )

    def _topological_sort(self) -> List[List[str]]:
        in_degree = {nid: len(self.reverse_graph.get(nid, set())) for nid in self.nodes_map}
        queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
        levels = []
        visited: set[str] = set()

        while queue:
            current_level = list(queue)
            levels.append(current_level)
            next_queue: deque[str] = deque()
            for node_id in current_level:
                visited.add(node_id)
                for dep_id in self.dependency_graph.get(node_id, []):
                    in_degree[dep_id] -= 1
                    if in_degree[dep_id] == 0:
                        next_queue.append(dep_id)
            queue = next_queue

        if len(visited) != len(self.nodes_map):
            unvisited = set(self.nodes_map.keys()) - visited
            raise ValueError(f"Cycle detected in pipeline DAG! Unvisited nodes: {unvisited}")
        return levels

    def execute(self, pipeline_input: Any) -> PipeContext:
        context = PipeContext(pipeline_input)
        execution_levels = self._topological_sort()

        if self.verbose:
            logger.info(f"Executing pipeline: {self.pipeline.name}")
            logger.info(f"Total nodes: {len(self.nodes_map)}, levels: {len(execution_levels)}")

        pipeline_start = time.perf_counter()
        for level_idx, level_nodes in enumerate(execution_levels):
            if self.verbose:
                logger.info(f"Level {level_idx + 1}/{len(execution_levels)}: {level_nodes}")
            for node_id in level_nodes:
                self._run_node(node_id, context)

        if self.verbose:
            total = time.perf_counter() - pipeline_start
            logger.info(f"Pipeline completed in {total:.2f}s")
        return context

    def _run_node(self, node_id: str, context: PipeContext):
        node = self.nodes_map[node_id]

        if self.verbose:
            logger.info(f"Executing: {node.id} (processor: {node.processor})")

        if node.condition and not self._evaluate_condition(node.condition, context):
            if self.verbose:
                logger.info(f"  Skipped (condition not met)")
            return

        kwargs = {}
        for param_name, input_config in node.inputs.items():
            try:
                value = FieldResolver.resolve(context, input_config)
                kwargs[param_name] = value
            except Exception as e:
                raise ValueError(
                    f"Failed to resolve input '{param_name}' for node '{node_id}': {e}"
                )

        processor = self.processors[node_id]

        if node.schema_ and hasattr(processor, "schema"):
            processor.schema = node.schema_

        node_start = time.perf_counter()
        try:
            result = processor(**kwargs)
        except Exception as e:
            raise RuntimeError(f"Node '{node_id}' failed: {e}")
        elapsed = time.perf_counter() - node_start

        if node.output.fields:
            if isinstance(node.output.fields, str):
                result = FieldResolver._extract_fields(result, node.output.fields)

        context.set(node.output.key, result)

        if self.verbose:
            logger.info(f"  Output: '{node.output.key}' — {elapsed:.2f}s")

    def _evaluate_condition(self, condition: str, context: PipeContext) -> bool:
        return True

    def close(self):
        """Close the store pool (if any), releasing all shared connections."""
        if self.store_pool is not None:
            self.store_pool.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
