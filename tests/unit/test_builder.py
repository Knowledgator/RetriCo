"""Tests for RetriCoBuilder."""

import pytest
from retrico.core.builders import RetriCoBuilder


class TestRetriCoBuilder:
    def test_minimal_config(self):
        builder = RetriCoBuilder(name="test")
        builder.ner_gliner(labels=["person"])
        config = builder.get_config()

        assert config["name"] == "test"
        assert len(config["nodes"]) == 3  # chunker, ner, graph_writer

        node_ids = [n["id"] for n in config["nodes"]]
        assert "chunker" in node_ids
        assert "ner" in node_ids
        assert "graph_writer" in node_ids

    def test_with_relex(self):
        builder = RetriCoBuilder()
        builder.ner_gliner(labels=["person", "location"])
        builder.relex_gliner(
            entity_labels=["person", "location"],
            relation_labels=["born in"],
        )
        config = builder.get_config()
        assert len(config["nodes"]) == 4  # chunker, ner, relex, graph_writer

    def test_chunks_only_pipeline_valid(self):
        """A pipeline with only chunker + graph_writer is valid (no NER/relex)."""
        builder = RetriCoBuilder()
        config = builder.get_config()
        node_ids = [n["id"] for n in config["nodes"]]
        assert "chunker" in node_ids
        assert "graph_writer" in node_ids
        writer_node = next(n for n in config["nodes"] if n["id"] == "graph_writer")
        assert "entities" not in writer_node["inputs"]

    def test_graph_writer_config(self):
        builder = RetriCoBuilder()
        builder.ner_gliner(labels=["person"])
        builder.graph_writer(neo4j_uri="bolt://custom:7687", neo4j_password="secret")
        config = builder.get_config()

        writer = [n for n in config["nodes"] if n["id"] == "graph_writer"][0]
        assert writer["config"]["neo4j_uri"] == "bolt://custom:7687"
        assert writer["config"]["neo4j_password"] == "secret"
