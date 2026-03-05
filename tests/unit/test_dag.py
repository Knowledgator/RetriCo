"""Tests for the DAG engine."""

import pytest
from retrico.core.dag import (
    PipeContext,
    FieldResolver,
    InputConfig,
    DAGPipeline,
    DAGExecutor,
    PipeNode,
    OutputConfig,
)
from retrico.core.registry import processor_registry
from retrico.core.base import BaseProcessor


class TestPipeContext:
    def test_set_and_get(self):
        ctx = PipeContext(pipeline_input={"texts": ["hello"]})
        ctx.set("step1", {"result": 42})

        assert ctx.get("$input") == {"texts": ["hello"]}
        assert ctx.get("step1") == {"result": 42}

    def test_get_by_index(self):
        ctx = PipeContext()
        ctx.set("a", 1)
        ctx.set("b", 2)

        assert ctx.get("outputs[0]") == 1
        assert ctx.get("outputs[-1]") == 2

    def test_serialization(self):
        ctx = PipeContext(pipeline_input="test")
        ctx.set("key", {"value": [1, 2, 3]})
        json_str = ctx.to_json()
        restored = PipeContext.from_json(json_data=json_str)
        assert restored.get("key") == {"value": [1, 2, 3]}


class TestFieldResolver:
    def test_simple_field(self):
        ctx = PipeContext()
        ctx.set("data", {"name": "Alice", "items": [1, 2, 3]})
        config = InputConfig(source="data", fields="name")
        result = FieldResolver.resolve(ctx, config)
        assert result == "Alice"

    def test_nested_field(self):
        ctx = PipeContext()
        ctx.set("data", {"entities": [{"text": "A"}, {"text": "B"}]})
        config = InputConfig(source="data", fields="entities[*].text")
        result = FieldResolver.resolve(ctx, config)
        assert result == ["A", "B"]

    def test_default_value(self):
        ctx = PipeContext()
        config = InputConfig(source="missing", default="fallback")
        result = FieldResolver.resolve(ctx, config)
        assert result == "fallback"

    def test_filter(self):
        ctx = PipeContext()
        ctx.set("data", [{"score": 0.8}, {"score": 0.3}, {"score": 0.6}])
        config = InputConfig(source="data", filter="score > 0.5")
        result = FieldResolver.resolve(ctx, config)
        assert len(result) == 2


class TestDAGExecutor:
    def test_simple_pipeline(self):
        # Register a test processor
        @processor_registry.register("test_double")
        def create_double(config_dict, pipeline=None):
            class DoubleProcessor(BaseProcessor):
                def __call__(self, **kwargs):
                    val = kwargs.get("value", 0)
                    return {"result": val * 2}
            return DoubleProcessor(config_dict, pipeline)

        pipeline = DAGPipeline(
            name="test",
            nodes=[
                PipeNode(
                    id="step1",
                    processor="test_double",
                    inputs={"value": InputConfig(source="$input", fields="value")},
                    output=OutputConfig(key="step1_result"),
                    config={},
                ),
            ],
        )

        executor = DAGExecutor(pipeline)
        ctx = executor.run(value=5)
        assert ctx.get("step1_result") == {"result": 10}

    def test_dependency_chain(self):
        @processor_registry.register("test_add_one")
        def create_add_one(config_dict, pipeline=None):
            class AddOneProcessor(BaseProcessor):
                def __call__(self, **kwargs):
                    val = kwargs.get("value", 0)
                    return {"result": val + 1}
            return AddOneProcessor(config_dict, pipeline)

        pipeline = DAGPipeline(
            name="chain_test",
            nodes=[
                PipeNode(
                    id="a",
                    processor="test_add_one",
                    inputs={"value": InputConfig(source="$input", fields="value")},
                    output=OutputConfig(key="a_result"),
                    config={},
                ),
                PipeNode(
                    id="b",
                    processor="test_add_one",
                    requires=["a"],
                    inputs={"value": InputConfig(source="a_result", fields="result")},
                    output=OutputConfig(key="b_result"),
                    config={},
                ),
            ],
        )

        executor = DAGExecutor(pipeline)
        ctx = executor.run(value=0)
        assert ctx.get("b_result") == {"result": 2}
