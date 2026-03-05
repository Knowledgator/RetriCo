"""Tests for data_ingest processor, IngestConfigBuilder, and JSON export."""

import json
import pytest
from unittest.mock import patch, MagicMock

from grapsit.construct.ingest import DataIngestProcessor
from grapsit.models.entity import EntityMention
from grapsit.models.relation import Relation
from grapsit.core.builders import IngestConfigBuilder


class TestDataIngestProcessor:
    """Unit tests for DataIngestProcessor."""

    def test_empty_input(self):
        proc = DataIngestProcessor({})
        result = proc(data=[])
        assert result["entities"] == [[]]
        assert result["relations"] == [[]]
        assert result["chunks"] == []
        assert result["documents"] == []

    def test_none_input_defaults_to_empty(self):
        proc = DataIngestProcessor({})
        result = proc()
        assert result["entities"] == [[]]
        assert result["relations"] == [[]]

    def test_entities_only(self):
        proc = DataIngestProcessor({})
        result = proc(data=[
            {
                "entities": [
                    {"text": "Einstein", "label": "person"},
                    {"text": "Ulm", "label": "location"},
                ],
            },
        ])
        mentions = result["entities"][0]
        assert len(mentions) == 2
        assert isinstance(mentions[0], EntityMention)
        assert mentions[0].text == "Einstein"
        assert mentions[0].label == "person"
        assert mentions[0].score == 1.0
        assert mentions[1].text == "Ulm"
        assert mentions[1].label == "location"

    def test_entities_with_id(self):
        proc = DataIngestProcessor({})
        result = proc(data=[
            {"entities": [{"text": "Einstein", "label": "person", "id": "Q937"}]},
        ])
        mention = result["entities"][0][0]
        assert mention.linked_entity_id == "Q937"

    def test_entities_with_score(self):
        proc = DataIngestProcessor({})
        result = proc(data=[
            {"entities": [{"text": "Einstein", "label": "person", "score": 0.95}]},
        ])
        assert result["entities"][0][0].score == 0.95

    def test_relations(self):
        proc = DataIngestProcessor({})
        result = proc(data=[
            {
                "entities": [
                    {"text": "Einstein", "label": "person"},
                    {"text": "Ulm", "label": "location"},
                ],
                "relations": [
                    {"head": "Einstein", "tail": "Ulm", "type": "born_in"},
                ],
            },
        ])
        rels = result["relations"][0]
        assert len(rels) == 1
        assert isinstance(rels[0], Relation)
        assert rels[0].head_text == "Einstein"
        assert rels[0].tail_text == "Ulm"
        assert rels[0].relation_type == "born_in"
        assert rels[0].score == 1.0

    def test_relations_with_metadata(self):
        proc = DataIngestProcessor({})
        result = proc(data=[
            {
                "entities": [
                    {"text": "Einstein", "label": "person"},
                    {"text": "Ulm", "label": "location"},
                ],
                "relations": [
                    {
                        "head": "Einstein",
                        "tail": "Ulm",
                        "type": "born_in",
                        "score": 0.9,
                        "head_label": "person",
                        "tail_label": "location",
                        "properties": {"year": 1879},
                    },
                ],
            },
        ])
        rel = result["relations"][0][0]
        assert rel.score == 0.9
        assert rel.head_label == "person"
        assert rel.tail_label == "location"
        assert rel.properties == {"year": 1879}

    def test_entity_properties(self):
        proc = DataIngestProcessor({})
        result = proc(data=[
            {
                "entities": [
                    {
                        "text": "Einstein",
                        "label": "person",
                        "properties": {"birth_year": 1879, "field": "physics"},
                    },
                ],
            },
        ])
        mention = result["entities"][0][0]
        assert mention.properties == {"birth_year": 1879, "field": "physics"}

    def test_entity_properties_flow_to_graph_writer(self):
        """Verify entity properties survive through the full pipeline."""
        with patch("grapsit.construct.graph_writer.resolve_from_pool_or_create") as mock_cs:
            mock_store = MagicMock()
            mock_cs.return_value = mock_store

            builder = IngestConfigBuilder()
            builder.graph_writer(setup_indexes=False)
            executor = builder.build()

            ctx = executor.execute({
                "data": [
                    {
                        "entities": [
                            {
                                "text": "Einstein",
                                "label": "person",
                                "properties": {"birth_year": 1879},
                            },
                        ],
                    },
                ],
            })

            result = ctx.get("writer_result")
            entity = list(result["entity_map"].values())[0]
            assert entity.properties == {"birth_year": 1879}

    def test_item_metadata_on_document(self):
        """Verify item-level metadata is passed to the Document."""
        proc = DataIngestProcessor({})
        result = proc(data=[
            {
                "entities": [{"text": "Einstein", "label": "person"}],
                "text": "Einstein was born in Ulm.",
                "metadata": {"source": "wikipedia", "language": "en"},
            },
        ])
        assert len(result["documents"]) == 1
        doc = result["documents"][0]
        assert doc.metadata == {"source": "wikipedia", "language": "en"}

    def test_item_metadata_ignored_without_text(self):
        """Metadata is ignored when no text is provided."""
        proc = DataIngestProcessor({})
        result = proc(data=[
            {
                "entities": [{"text": "Einstein", "label": "person"}],
                "metadata": {"source": "wikipedia"},
            },
        ])
        assert result["documents"] == []

    def test_multiple_items(self):
        proc = DataIngestProcessor({})
        result = proc(data=[
            {"entities": [{"text": "Einstein", "label": "person"}]},
            {"entities": [{"text": "Ulm", "label": "location"}]},
        ])
        assert len(result["entities"]) == 2
        assert result["entities"][0][0].text == "Einstein"
        assert result["entities"][1][0].text == "Ulm"

    def test_item_without_relations(self):
        proc = DataIngestProcessor({})
        result = proc(data=[
            {"entities": [{"text": "Einstein", "label": "person"}]},
        ])
        assert result["relations"][0] == []

    def test_output_shape_is_list_of_lists(self):
        proc = DataIngestProcessor({})
        result = proc(data=[
            {
                "entities": [{"text": "A", "label": "x"}],
                "relations": [{"head": "A", "tail": "B", "type": "r"}],
            },
        ])
        # graph_writer expects List[List[...]]
        assert isinstance(result["entities"], list)
        assert isinstance(result["entities"][0], list)
        assert isinstance(result["relations"], list)
        assert isinstance(result["relations"][0], list)


class TestIngestConfigBuilder:
    """Unit tests for IngestConfigBuilder."""

    def test_default_config(self):
        builder = IngestConfigBuilder(name="test")
        config = builder.get_config()
        assert config["name"] == "test"
        assert len(config["nodes"]) == 2
        assert config["nodes"][0]["processor"] == "data_ingest"
        assert config["nodes"][1]["processor"] == "graph_writer"

    def test_graph_writer_config(self):
        builder = IngestConfigBuilder()
        builder.graph_writer(neo4j_uri="bolt://myhost:7687", store_type="neo4j")
        config = builder.get_config()
        writer_node = config["nodes"][1]
        assert writer_node["config"]["neo4j_uri"] == "bolt://myhost:7687"
        assert writer_node["config"]["store_type"] == "neo4j"

    def test_dag_dependencies(self):
        builder = IngestConfigBuilder()
        config = builder.get_config()
        writer_node = config["nodes"][1]
        assert "data_ingest" in writer_node["requires"]

    def test_data_ingest_inputs(self):
        builder = IngestConfigBuilder()
        config = builder.get_config()
        ingest_node = config["nodes"][0]
        assert ingest_node["inputs"]["data"]["source"] == "$input"

    def test_writer_inputs_from_ingest(self):
        builder = IngestConfigBuilder()
        config = builder.get_config()
        writer_node = config["nodes"][1]
        assert writer_node["inputs"]["entities"]["source"] == "ingest_result"
        assert writer_node["inputs"]["relations"]["source"] == "ingest_result"

    def test_build_creates_executor(self):
        builder = IngestConfigBuilder()
        builder.graph_writer(setup_indexes=False)
        with patch("grapsit.construct.graph_writer.resolve_from_pool_or_create") as mock_cs:
            mock_store = MagicMock()
            mock_cs.return_value = mock_store
            executor = builder.build()
            assert executor is not None

    def test_save_and_load(self, tmp_path):
        builder = IngestConfigBuilder(name="save_test")
        builder.graph_writer(neo4j_uri="bolt://localhost:7687")
        filepath = str(tmp_path / "ingest.yaml")
        builder.save(filepath)

        import yaml
        with open(filepath) as f:
            config = yaml.safe_load(f)
        assert config["name"] == "save_test"
        assert len(config["nodes"]) == 2


class TestIngestIntegration:
    """Integration test: data_ingest -> graph_writer with mocked store."""

    def test_full_ingest_pipeline(self):
        with patch("grapsit.construct.graph_writer.resolve_from_pool_or_create") as mock_cs:
            mock_store = MagicMock()
            mock_cs.return_value = mock_store

            builder = IngestConfigBuilder(name="integration_test")
            builder.graph_writer(setup_indexes=False)
            executor = builder.build()

            ctx = executor.execute({
                "data": [
                    {
                        "entities": [
                            {"text": "Einstein", "label": "person"},
                            {"text": "Ulm", "label": "location"},
                        ],
                        "relations": [
                            {"head": "Einstein", "tail": "Ulm", "type": "born_in"},
                        ],
                    },
                    {
                        "entities": [
                            {"text": "Germany", "label": "location"},
                        ],
                        "relations": [
                            {"head": "Ulm", "tail": "Germany", "type": "located_in"},
                        ],
                    },
                ],
            })

            result = ctx.get("writer_result")
            assert result["entity_count"] == 3
            assert result["relation_count"] == 2
            assert result["chunk_count"] == 0

            # Verify store was called correctly
            assert mock_store.write_entity.call_count == 3
            assert mock_store.write_relation.call_count == 2

    def test_ingest_entities_only(self):
        with patch("grapsit.construct.graph_writer.resolve_from_pool_or_create") as mock_cs:
            mock_store = MagicMock()
            mock_cs.return_value = mock_store

            builder = IngestConfigBuilder()
            builder.graph_writer(setup_indexes=False)
            executor = builder.build()

            ctx = executor.execute({
                "data": [
                    {"entities": [{"text": "Einstein", "label": "person"}]},
                ],
            })

            result = ctx.get("writer_result")
            assert result["entity_count"] == 1
            assert result["relation_count"] == 0

    def test_ingest_with_entity_ids(self):
        with patch("grapsit.construct.graph_writer.resolve_from_pool_or_create") as mock_cs:
            mock_store = MagicMock()
            mock_cs.return_value = mock_store

            builder = IngestConfigBuilder()
            builder.graph_writer(setup_indexes=False)
            executor = builder.build()

            ctx = executor.execute({
                "data": [
                    {
                        "entities": [
                            {"text": "Einstein", "label": "person", "id": "Q937"},
                            {"text": "Albert Einstein", "label": "person", "id": "Q937"},
                        ],
                    },
                ],
            })

            result = ctx.get("writer_result")
            # Both mentions share the same linked_entity_id, so they dedup to 1 entity
            assert result["entity_count"] == 1

    def test_convenience_function(self):
        with patch("grapsit.construct.graph_writer.resolve_from_pool_or_create") as mock_cs:
            mock_store = MagicMock()
            mock_cs.return_value = mock_store

            import grapsit
            ctx = grapsit.ingest_data(
                data=[
                    {
                        "entities": [
                            {"text": "Einstein", "label": "person"},
                            {"text": "Ulm", "label": "location"},
                        ],
                        "relations": [
                            {"head": "Einstein", "tail": "Ulm", "type": "born_in"},
                        ],
                    },
                ],
            )

            result = ctx.get("writer_result")
            assert result["entity_count"] == 2
            assert result["relation_count"] == 1


class TestJsonExport:
    """Tests for graph_writer JSON export feature."""

    def test_json_output_creates_file(self, tmp_path):
        json_path = str(tmp_path / "output.json")
        with patch("grapsit.construct.graph_writer.resolve_from_pool_or_create") as mock_cs:
            mock_store = MagicMock()
            mock_cs.return_value = mock_store

            builder = IngestConfigBuilder()
            builder.graph_writer(setup_indexes=False, json_output=json_path)
            executor = builder.build()

            executor.execute({
                "data": [
                    {
                        "entities": [
                            {"text": "Einstein", "label": "person"},
                            {"text": "Ulm", "label": "location"},
                        ],
                        "relations": [
                            {"head": "Einstein", "tail": "Ulm", "type": "born_in", "score": 0.9},
                        ],
                    },
                ],
            })

            with open(json_path) as f:
                data = json.load(f)

            # Output is a list of items (ingest-ready format)
            assert isinstance(data, list)
            assert len(data) == 1
            item = data[0]
            assert len(item["entities"]) == 2
            assert len(item["relations"]) == 1
            # Entities in ingest format
            texts = {e["text"] for e in item["entities"]}
            assert "Einstein" in texts
            assert "Ulm" in texts
            # Relations in ingest format
            rel = item["relations"][0]
            assert rel["head"] == "Einstein"
            assert rel["tail"] == "Ulm"
            assert rel["type"] == "born_in"
            assert rel["score"] == 0.9

    def test_json_output_entities_only(self, tmp_path):
        json_path = str(tmp_path / "entities_only.json")
        with patch("grapsit.construct.graph_writer.resolve_from_pool_or_create") as mock_cs:
            mock_store = MagicMock()
            mock_cs.return_value = mock_store

            builder = IngestConfigBuilder()
            builder.graph_writer(setup_indexes=False, json_output=json_path)
            executor = builder.build()

            executor.execute({
                "data": [
                    {"entities": [{"text": "Einstein", "label": "person"}]},
                ],
            })

            with open(json_path) as f:
                data = json.load(f)

            assert isinstance(data, list)
            assert len(data) == 1
            assert len(data[0]["entities"]) == 1
            assert data[0]["relations"] == []

    def test_json_output_with_entity_ids(self, tmp_path):
        json_path = str(tmp_path / "with_ids.json")
        with patch("grapsit.construct.graph_writer.resolve_from_pool_or_create") as mock_cs:
            mock_store = MagicMock()
            mock_cs.return_value = mock_store

            builder = IngestConfigBuilder()
            builder.graph_writer(setup_indexes=False, json_output=json_path)
            executor = builder.build()

            executor.execute({
                "data": [
                    {
                        "entities": [
                            {"text": "Einstein", "label": "person", "id": "Q937"},
                            {"text": "Albert Einstein", "label": "person", "id": "Q937"},
                        ],
                    },
                ],
            })

            with open(json_path) as f:
                data = json.load(f)

            # Deduped to 1 entity, should have the ID
            assert isinstance(data, list)
            assert len(data) == 1
            assert len(data[0]["entities"]) == 1
            assert data[0]["entities"][0]["id"] == "Q937"

    def test_json_output_skips_unresolved_relations(self, tmp_path):
        json_path = str(tmp_path / "unresolved.json")
        with patch("grapsit.construct.graph_writer.resolve_from_pool_or_create") as mock_cs:
            mock_store = MagicMock()
            mock_cs.return_value = mock_store

            builder = IngestConfigBuilder()
            builder.graph_writer(setup_indexes=False, json_output=json_path)
            executor = builder.build()

            executor.execute({
                "data": [
                    {
                        "entities": [{"text": "Einstein", "label": "person"}],
                        "relations": [
                            # "Ulm" doesn't exist as entity — should be skipped
                            {"head": "Einstein", "tail": "Ulm", "type": "born_in"},
                        ],
                    },
                ],
            })

            with open(json_path) as f:
                data = json.load(f)

            assert isinstance(data, list)
            assert len(data) == 1
            assert len(data[0]["entities"]) == 1
            assert len(data[0]["relations"]) == 0

    def test_json_output_no_file_when_not_configured(self, tmp_path):
        with patch("grapsit.construct.graph_writer.resolve_from_pool_or_create") as mock_cs:
            mock_store = MagicMock()
            mock_cs.return_value = mock_store

            builder = IngestConfigBuilder()
            builder.graph_writer(setup_indexes=False)
            executor = builder.build()

            executor.execute({
                "data": [
                    {"entities": [{"text": "Einstein", "label": "person"}]},
                ],
            })

            # No JSON file should exist
            assert not list(tmp_path.iterdir())

    def test_json_output_creates_parent_dirs(self, tmp_path):
        json_path = str(tmp_path / "sub" / "dir" / "output.json")
        with patch("grapsit.construct.graph_writer.resolve_from_pool_or_create") as mock_cs:
            mock_store = MagicMock()
            mock_cs.return_value = mock_store

            builder = IngestConfigBuilder()
            builder.graph_writer(setup_indexes=False, json_output=json_path)
            executor = builder.build()

            executor.execute({
                "data": [
                    {"entities": [{"text": "Einstein", "label": "person"}]},
                ],
            })

            with open(json_path) as f:
                data = json.load(f)
            assert isinstance(data, list)
            assert len(data) == 1
            assert len(data[0]["entities"]) == 1

    def test_json_is_ingest_compatible(self, tmp_path):
        """Verify the JSON output can be fed back into ingest_data()."""
        json_path = str(tmp_path / "roundtrip.json")
        with patch("grapsit.construct.graph_writer.resolve_from_pool_or_create") as mock_cs:
            mock_store = MagicMock()
            mock_cs.return_value = mock_store

            builder = IngestConfigBuilder()
            builder.graph_writer(setup_indexes=False, json_output=json_path)
            executor = builder.build()

            executor.execute({
                "data": [
                    {
                        "entities": [
                            {"text": "Einstein", "label": "person"},
                            {"text": "Ulm", "label": "location"},
                        ],
                        "relations": [
                            {"head": "Einstein", "tail": "Ulm", "type": "born_in"},
                        ],
                    },
                ],
            })

            with open(json_path) as f:
                exported = json.load(f)

            # Output is already in ingest-ready format — feed it directly
            assert isinstance(exported, list)
            proc = DataIngestProcessor({})
            result = proc(data=exported)
            assert len(result["entities"][0]) == 2
            assert len(result["relations"][0]) == 1

    def test_build_config_builder_json_output(self, tmp_path):
        """Test json_output works via BuildConfigBuilder too."""
        json_path = str(tmp_path / "build_output.json")
        from grapsit.core.builders import BuildConfigBuilder
        builder = BuildConfigBuilder()
        builder.graph_writer(setup_indexes=False, json_output=json_path)
        builder.ner_gliner(labels=["person"])
        config = builder.get_config()
        writer_config = config["nodes"][-1]["config"]
        assert writer_config["json_output"] == json_path