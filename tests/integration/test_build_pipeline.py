"""Integration test — full pipeline with mocked GLiNER and Neo4j."""

import json
import pytest
from unittest.mock import MagicMock, patch

from retrico.core.factory import ProcessorFactory
from retrico.core.builders import RetriCoBuilder


class TestBuildPipeline:
    @patch("retrico.construct.graph_writer.resolve_from_pool_or_create")
    def test_full_pipeline_ner_only(self, mock_create_store):
        """Test chunker -> NER -> graph_writer (no relex)."""
        mock_store = MagicMock()
        mock_create_store.return_value = mock_store

        builder = RetriCoBuilder(name="integration_test")
        builder.chunker(method="sentence")
        builder.ner_gliner(model="test-model", labels=["person", "location"], threshold=0.3)
        builder.graph_writer(neo4j_uri="bolt://localhost:7687")

        executor = builder.build(verbose=False)

        # Mock the NER model via engine
        ner_proc = executor.processors["ner"]
        ner_proc._engine._model = MagicMock()
        ner_proc._engine._model.inference.return_value = [
            [{"text": "Einstein", "label": "person", "start": 0, "end": 8, "score": 0.95}],
        ]

        result = executor.run(texts=["Einstein was a physicist."])

        assert result.has("chunker_result")
        assert result.has("ner_result")
        assert result.has("writer_result")

        writer_result = result.get("writer_result")
        assert writer_result["entity_count"] >= 1
        assert writer_result["chunk_count"] >= 1

    @patch("retrico.construct.graph_writer.resolve_from_pool_or_create")
    def test_full_pipeline_with_relex(self, mock_create_store):
        """Test chunker -> NER -> relex -> graph_writer."""
        mock_store = MagicMock()
        mock_create_store.return_value = mock_store

        builder = RetriCoBuilder(name="integration_relex")
        builder.chunker(method="sentence")
        builder.ner_gliner(model="test-ner", labels=["person", "location"])
        builder.relex_gliner(
            model="test-relex",
            entity_labels=["person", "location"],
            relation_labels=["born in"],
        )
        builder.graph_writer()

        executor = builder.build()

        # Mock NER via engine
        ner_proc = executor.processors["ner"]
        ner_proc._engine._model = MagicMock()
        ner_proc._engine._model.inference.return_value = [
            [
                {"text": "Einstein", "label": "person", "start": 0, "end": 8, "score": 0.95},
                {"text": "Ulm", "label": "location", "start": 21, "end": 24, "score": 0.85},
            ],
        ]

        # Mock relex via engine
        relex_proc = executor.processors["relex"]
        relex_proc._engine._model = MagicMock()
        relex_proc._engine._model.inference.return_value = (
            [[{"text": "Einstein", "label": "person"}, {"text": "Ulm", "label": "location"}]],
            [[{
                "head": {"entity_idx": 0},
                "tail": {"entity_idx": 1},
                "relation": "born in",
                "score": 0.8,
            }]],
        )

        result = executor.run(texts=["Einstein was born in Ulm."])

        assert result.has("relex_result")

        # Verify relex received NER entities as input_spans
        call_kwargs = relex_proc._engine._model.inference.call_args[1]
        assert "input_spans" in call_kwargs
        spans = call_kwargs["input_spans"][0]
        assert any(s["text"] == "Einstein" for s in spans)
        assert any(s["text"] == "Ulm" for s in spans)

        writer_result = result.get("writer_result")
        assert writer_result["entity_count"] >= 2
        assert writer_result["relation_count"] >= 1

    @patch("retrico.construct.graph_writer.resolve_from_pool_or_create")
    def test_llm_pipeline_ner_and_relex(self, mock_create_store):
        """Test chunker -> ner_llm -> relex_llm -> graph_writer."""
        mock_store = MagicMock()
        mock_create_store.return_value = mock_store

        builder = RetriCoBuilder(name="llm_pipeline")
        builder.chunker(method="sentence")
        builder.ner_llm(api_key="test", model="test-model", labels=["person", "location"])
        builder.relex_llm(
            api_key="test",
            model="test-model",
            entity_labels=["person", "location"],
            relation_labels=["born in"],
        )
        builder.graph_writer()

        executor = builder.build()

        # Mock NER LLM client via engine
        ner_proc = executor.processors["ner"]
        ner_proc._engine._client = MagicMock()
        ner_proc._engine._client.complete.return_value = json.dumps({"entities": [
            {"text": "Einstein", "label": "person", "start": 0, "end": 8},
            {"text": "Ulm", "label": "location", "start": 21, "end": 24},
        ]})

        # Mock relex LLM client via engine
        relex_proc = executor.processors["relex"]
        relex_proc._engine._client = MagicMock()
        relex_proc._engine._client.complete.return_value = json.dumps({"relations": [
            {"head": "Einstein", "tail": "Ulm", "relation": "born in"},
        ]})

        result = executor.run(texts=["Einstein was born in Ulm."])

        assert result.has("ner_result")
        assert result.has("relex_result")
        assert result.has("writer_result")

        writer_result = result.get("writer_result")
        assert writer_result["entity_count"] >= 2
        assert writer_result["relation_count"] >= 1

    @patch("retrico.construct.graph_writer.resolve_from_pool_or_create")
    def test_mixed_pipeline_gliner_ner_llm_relex(self, mock_create_store):
        """Test chunker -> ner_gliner -> relex_llm -> graph_writer."""
        mock_store = MagicMock()
        mock_create_store.return_value = mock_store

        builder = RetriCoBuilder(name="mixed_pipeline")
        builder.chunker(method="sentence")
        builder.ner_gliner(model="test-ner", labels=["person", "location"])
        builder.relex_llm(
            api_key="test",
            model="test-model",
            entity_labels=["person", "location"],
            relation_labels=["born in"],
        )
        builder.graph_writer()

        executor = builder.build()

        # Mock GLiNER NER via engine
        ner_proc = executor.processors["ner"]
        ner_proc._engine._model = MagicMock()
        ner_proc._engine._model.inference.return_value = [
            [
                {"text": "Einstein", "label": "person", "start": 0, "end": 8, "score": 0.95},
                {"text": "Ulm", "label": "location", "start": 21, "end": 24, "score": 0.85},
            ],
        ]

        # Mock relex LLM client via engine
        relex_proc = executor.processors["relex"]
        relex_proc._engine._client = MagicMock()
        relex_proc._engine._client.complete.return_value = json.dumps({"relations": [
            {"head": "Einstein", "tail": "Ulm", "relation": "born in"},
        ]})

        result = executor.run(texts=["Einstein was born in Ulm."])

        assert result.has("relex_result")
        relex_result = result.get("relex_result")
        assert len(relex_result["relations"][0]) == 1
        assert relex_result["relations"][0][0].head_text == "Einstein"

        # Entities should be the GLiNER ones passed through
        assert len(relex_result["entities"][0]) == 2

        writer_result = result.get("writer_result")
        assert writer_result["entity_count"] >= 2
        assert writer_result["relation_count"] >= 1

    @patch("retrico.construct.graph_writer.resolve_from_pool_or_create")
    def test_llm_relex_standalone(self, mock_create_store):
        """Test chunker -> relex_llm (standalone, no NER) -> graph_writer."""
        mock_store = MagicMock()
        mock_create_store.return_value = mock_store

        builder = RetriCoBuilder(name="standalone_relex")
        builder.chunker(method="sentence")
        builder.relex_llm(
            api_key="test",
            model="test-model",
            entity_labels=["person", "location"],
            relation_labels=["born in"],
        )
        builder.graph_writer()

        executor = builder.build()

        # Mock relex LLM client via engine
        relex_proc = executor.processors["relex"]
        relex_proc._engine._client = MagicMock()
        relex_proc._engine._client.complete.return_value = json.dumps({
            "entities": [
                {"text": "Einstein", "label": "person"},
                {"text": "Ulm", "label": "location"},
            ],
            "relations": [
                {"head": "Einstein", "tail": "Ulm", "relation": "born in"},
            ],
        })

        result = executor.run(texts=["Einstein was born in Ulm."])

        assert result.has("relex_result")
        assert result.has("writer_result")

        writer_result = result.get("writer_result")
        assert writer_result["entity_count"] >= 2
        assert writer_result["relation_count"] >= 1

    @patch("retrico.construct.graph_writer.resolve_from_pool_or_create")
    def test_write_reversed_relations(self, mock_create_store):
        """Test that write_reversed_relations writes both forward and reverse edges."""
        mock_store = MagicMock()
        mock_create_store.return_value = mock_store

        builder = RetriCoBuilder(name="reversed_test")
        builder.chunker(method="sentence")
        builder.ner_gliner(model="test-ner", labels=["person", "location"])
        builder.relex_gliner(
            model="test-relex",
            entity_labels=["person", "location"],
            relation_labels=["born in"],
        )
        builder.graph_writer(write_reversed_relations=True)

        executor = builder.build()

        # Mock NER
        ner_proc = executor.processors["ner"]
        ner_proc._engine._model = MagicMock()
        ner_proc._engine._model.inference.return_value = [
            [
                {"text": "Einstein", "label": "person", "start": 0, "end": 8, "score": 0.95},
                {"text": "Ulm", "label": "location", "start": 21, "end": 24, "score": 0.85},
            ],
        ]

        # Mock relex
        relex_proc = executor.processors["relex"]
        relex_proc._engine._model = MagicMock()
        relex_proc._engine._model.inference.return_value = (
            [[{"text": "Einstein", "label": "person"}, {"text": "Ulm", "label": "location"}]],
            [[{
                "head": {"entity_idx": 0},
                "tail": {"entity_idx": 1},
                "relation": "born in",
                "score": 0.8,
            }]],
        )

        result = executor.run(texts=["Einstein was born in Ulm."])
        writer_result = result.get("writer_result")

        assert writer_result["relation_count"] >= 1

        # write_relation should be called twice per relation (forward + reverse)
        write_calls = mock_store.write_relation.call_args_list
        assert len(write_calls) == 2

        # First call: forward relation (BORN_IN)
        fwd_rel = write_calls[0][0][0]
        assert fwd_rel.head_text == "Einstein"
        assert fwd_rel.tail_text == "Ulm"

        # Second call: reverse relation (REV_BORN_IN)
        rev_rel = write_calls[1][0][0]
        assert rev_rel.relation_type == "REV_BORN_IN"
        assert rev_rel.head_text == "Ulm"
        assert rev_rel.tail_text == "Einstein"

    def test_yaml_config_loading(self, tmp_path):
        """Test loading pipeline from YAML."""
        builder = RetriCoBuilder(name="yaml_test")
        builder.ner_gliner(labels=["person"])
        yaml_path = str(tmp_path / "test.yaml")
        builder.save(yaml_path)

        # Verify it loads without error (models are lazily loaded)
        executor = ProcessorFactory.create_pipeline(yaml_path)
        assert executor.pipeline.name == "yaml_test"
        assert len(executor.nodes_map) == 3
