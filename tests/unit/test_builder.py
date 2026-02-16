"""Tests for BuildConfigBuilder."""

import pytest
from grapsit.core.builders import BuildConfigBuilder


class TestBuildConfigBuilder:
    def test_minimal_config(self):
        builder = BuildConfigBuilder(name="test")
        builder.ner_gliner(labels=["person"])
        config = builder.get_config()

        assert config["name"] == "test"
        assert len(config["nodes"]) == 3  # chunker, ner, graph_writer

        node_ids = [n["id"] for n in config["nodes"]]
        assert "chunker" in node_ids
        assert "ner" in node_ids
        assert "graph_writer" in node_ids

    def test_with_relex(self):
        builder = BuildConfigBuilder()
        builder.ner_gliner(labels=["person", "location"])
        builder.relex_gliner(
            entity_labels=["person", "location"],
            relation_labels=["born in"],
        )
        config = builder.get_config()
        assert len(config["nodes"]) == 4  # chunker, ner, relex, graph_writer

    def test_requires_ner_or_relex(self):
        builder = BuildConfigBuilder()
        with pytest.raises(ValueError, match="NER, linker, or relex config required"):
            builder.get_config()

    def test_graph_writer_config(self):
        builder = BuildConfigBuilder()
        builder.ner_gliner(labels=["person"])
        builder.graph_writer(neo4j_uri="bolt://custom:7687", neo4j_password="secret")
        config = builder.get_config()

        writer = [n for n in config["nodes"] if n["id"] == "graph_writer"][0]
        assert writer["config"]["neo4j_uri"] == "bolt://custom:7687"
        assert writer["config"]["neo4j_password"] == "secret"
