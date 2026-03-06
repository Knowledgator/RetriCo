"""Tests for GraphWriterProcessor — relational store integration."""

import pytest
from unittest.mock import MagicMock, patch, call

from retrico.models.document import Chunk, Document
from retrico.models.entity import EntityMention
from retrico.models.relation import Relation


@pytest.fixture
def mock_graph_store():
    store = MagicMock()
    store.setup_indexes = MagicMock()
    return store


@pytest.fixture
def mock_relational_store():
    store = MagicMock()
    store.write_records = MagicMock()
    return store


class TestGraphWriterWithRelationalStore:
    """Graph writer writes chunks/documents to relational store when configured."""

    @patch("retrico.construct.graph_writer.resolve_from_pool_or_create")
    def test_writes_chunks_and_documents(self, mock_resolve, mock_graph_store, mock_relational_store):
        """When relational store is configured, chunks and documents are written to it."""
        def _resolve(config, category):
            if category == "graph":
                return mock_graph_store
            elif category == "relational":
                return mock_relational_store
            raise ValueError(f"Unknown: {category}")

        mock_resolve.side_effect = _resolve

        from retrico.construct.graph_writer import GraphWriterProcessor

        config = {
            "store_type": "neo4j",
            "relational_store_type": "sqlite",
            "sqlite_path": ":memory:",
            "setup_indexes": False,
        }
        writer = GraphWriterProcessor(config)
        assert writer.relational_store is mock_relational_store

        docs = [Document(id="d1", text="Hello world", source="test.txt")]
        chunks = [
            Chunk(id="c1", document_id="d1", text="Hello world", index=0, start_char=0, end_char=11),
        ]
        entities = [
            [EntityMention(text="World", label="location", start=6, end=11, score=0.9, chunk_id="c1")],
        ]

        writer(chunks=chunks, documents=docs, entities=entities, relations=[])

        # Check relational store received write_records calls
        assert mock_relational_store.write_records.call_count == 2

        # Documents call
        doc_call = mock_relational_store.write_records.call_args_list[0]
        assert doc_call[0][0] == "documents"
        assert len(doc_call[0][1]) == 1
        assert doc_call[0][1][0]["id"] == "d1"
        assert doc_call[0][1][0]["source"] == "test.txt"

        # Chunks call
        chunk_call = mock_relational_store.write_records.call_args_list[1]
        assert chunk_call[0][0] == "chunks"
        assert len(chunk_call[0][1]) == 1
        assert chunk_call[0][1][0]["id"] == "c1"
        assert chunk_call[0][1][0]["text"] == "Hello world"

    @patch("retrico.construct.graph_writer.resolve_from_pool_or_create")
    def test_custom_table_names(self, mock_resolve, mock_graph_store, mock_relational_store):
        """Custom chunk_table and document_table names are used."""
        def _resolve(config, category):
            if category == "graph":
                return mock_graph_store
            elif category == "relational":
                return mock_relational_store
            raise ValueError(f"Unknown: {category}")

        mock_resolve.side_effect = _resolve

        from retrico.construct.graph_writer import GraphWriterProcessor

        config = {
            "store_type": "neo4j",
            "relational_store_type": "sqlite",
            "setup_indexes": False,
            "chunk_table": "my_chunks",
            "document_table": "my_docs",
        }
        writer = GraphWriterProcessor(config)
        assert writer.chunk_table == "my_chunks"
        assert writer.document_table == "my_docs"

        docs = [Document(id="d1", text="Test", source="test")]
        chunks = [Chunk(id="c1", document_id="d1", text="Test", index=0)]

        writer(chunks=chunks, documents=docs, entities=[], relations=[])

        doc_call = mock_relational_store.write_records.call_args_list[0]
        assert doc_call[0][0] == "my_docs"

        chunk_call = mock_relational_store.write_records.call_args_list[1]
        assert chunk_call[0][0] == "my_chunks"

    @patch("retrico.construct.graph_writer.resolve_from_pool_or_create")
    def test_no_relational_store_backward_compat(self, mock_resolve, mock_graph_store):
        """Without relational store config, writer works unchanged."""
        mock_resolve.side_effect = lambda config, category: (
            mock_graph_store if category == "graph" else (_ for _ in ()).throw(ValueError("no config"))
        )

        from retrico.construct.graph_writer import GraphWriterProcessor

        config = {"store_type": "neo4j", "setup_indexes": False}
        writer = GraphWriterProcessor(config)
        assert writer.relational_store is None

        chunks = [Chunk(id="c1", document_id="d1", text="Test", index=0)]
        entities = [
            [EntityMention(text="Test", label="thing", start=0, end=4, score=0.9, chunk_id="c1")],
        ]

        result = writer(chunks=chunks, documents=[], entities=entities, relations=[])
        assert result["entity_count"] == 1
        assert result["chunk_count"] == 1

    @patch("retrico.construct.graph_writer.resolve_from_pool_or_create")
    def test_empty_chunks_and_docs_skip_writes(self, mock_resolve, mock_graph_store, mock_relational_store):
        """Empty chunks/documents lists skip relational store writes."""
        def _resolve(config, category):
            if category == "graph":
                return mock_graph_store
            elif category == "relational":
                return mock_relational_store
            raise ValueError(f"Unknown: {category}")

        mock_resolve.side_effect = _resolve

        from retrico.construct.graph_writer import GraphWriterProcessor

        config = {
            "store_type": "neo4j",
            "relational_store_type": "sqlite",
            "setup_indexes": False,
        }
        writer = GraphWriterProcessor(config)

        writer(chunks=[], documents=[], entities=[], relations=[])

        # write_records should not be called with empty lists
        mock_relational_store.write_records.assert_not_called()

    @patch("retrico.construct.graph_writer.resolve_from_pool_or_create")
    def test_relational_store_via_pool(self, mock_resolve, mock_graph_store, mock_relational_store):
        """Relational store resolved from pool when __store_pool__ is set."""
        from retrico.store.pool import StorePool

        pool = MagicMock(spec=StorePool)
        pool.has_relational.return_value = True

        def _resolve(config, category):
            if category == "graph":
                return mock_graph_store
            elif category == "relational":
                return mock_relational_store
            raise ValueError(f"Unknown: {category}")

        mock_resolve.side_effect = _resolve

        from retrico.construct.graph_writer import GraphWriterProcessor

        config = {
            "store_type": "neo4j",
            "setup_indexes": False,
            "__store_pool__": pool,
        }
        writer = GraphWriterProcessor(config)
        assert writer.relational_store is mock_relational_store


class TestBuilderRelationalStoreWiring:
    """Builder correctly wires relational stores into pool and writer config."""

    def test_chunk_store_registers_in_pool(self):
        from retrico.core.builders import RetriCoBuilder

        builder = RetriCoBuilder(name="test")
        builder.chunk_store(type="sqlite", sqlite_path="/tmp/test.db")

        assert "default" in builder._relational_stores
        assert builder._relational_stores["default"]["relational_store_type"] == "sqlite"
        assert builder._relational_stores["default"]["sqlite_path"] == "/tmp/test.db"

    def test_chunk_store_named(self):
        from retrico.core.builders import RetriCoBuilder

        builder = RetriCoBuilder(name="test")
        builder.chunk_store(type="sqlite", name="chunks_db", sqlite_path=":memory:")

        assert "chunks_db" in builder._relational_stores

    def test_chunk_store_config_object(self):
        from retrico.core.builders import RetriCoBuilder
        from retrico.store.config import SqliteRelationalConfig

        builder = RetriCoBuilder(name="test")
        builder.chunk_store(config=SqliteRelationalConfig(path="/tmp/test.db", name="my_store"))

        assert "my_store" in builder._relational_stores

    def test_stores_section_includes_relational(self):
        from retrico.core.builders import RetriCoBuilder

        builder = RetriCoBuilder(name="test")
        builder.graph_store()
        builder.chunk_store(type="sqlite")
        builder.ner_gliner(labels=["person"])
        builder.graph_writer()

        config = builder.get_config()
        assert "stores" in config
        assert "relational" in config["stores"]
        assert "default" in config["stores"]["relational"]

    def test_writer_config_includes_relational_keys(self):
        from retrico.core.builders import RetriCoBuilder

        builder = RetriCoBuilder(name="test")
        builder.chunk_store(type="sqlite", sqlite_path="/tmp/chunks.db")
        builder.ner_gliner(labels=["person"])
        builder.graph_writer()

        config = builder.get_config()
        writer_node = next(n for n in config["nodes"] if n["id"] == "graph_writer")
        assert writer_node["config"]["relational_store_type"] == "sqlite"
        assert writer_node["config"]["sqlite_path"] == "/tmp/chunks.db"

    def test_writer_config_custom_table_names(self):
        from retrico.core.builders import RetriCoBuilder

        builder = RetriCoBuilder(name="test")
        builder.chunk_store(type="sqlite")
        builder.ner_gliner(labels=["person"])
        builder.graph_writer(chunk_table="text_chunks", document_table="text_docs")

        config = builder.get_config()
        writer_node = next(n for n in config["nodes"] if n["id"] == "graph_writer")
        assert writer_node["config"]["chunk_table"] == "text_chunks"
        assert writer_node["config"]["document_table"] == "text_docs"

    def test_no_chunk_store_no_relational_keys(self):
        from retrico.core.builders import RetriCoBuilder

        builder = RetriCoBuilder(name="test")
        builder.ner_gliner(labels=["person"])
        builder.graph_writer()

        config = builder.get_config()
        writer_node = next(n for n in config["nodes"] if n["id"] == "graph_writer")
        assert "relational_store_type" not in writer_node["config"]

    def test_build_pool_includes_relational(self):
        from retrico.core.builders import RetriCoBuilder

        builder = RetriCoBuilder(name="test")
        builder.graph_store()
        builder.chunk_store(type="sqlite", sqlite_path=":memory:")
        builder.ner_gliner(labels=["person"])
        builder.graph_writer()

        pool = builder._build_pool()
        assert pool is not None
        assert pool.has_relational("default")

    def test_ingest_builder_with_chunk_store(self):
        from retrico.core.builders import RetriCoIngest

        builder = RetriCoIngest(name="test")
        builder.chunk_store(type="sqlite", sqlite_path=":memory:")
        builder.graph_writer()

        config = builder.get_config()
        writer_node = next(n for n in config["nodes"] if n["id"] == "graph_writer")
        assert writer_node["config"]["relational_store_type"] == "sqlite"
