"""Tests for StoreReaderProcessor and builder integration."""

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# StoreReaderProcessor unit tests
# ---------------------------------------------------------------------------


class TestStoreReaderProcessor:
    def _make_processor(self, config=None):
        from grapsit.construct.store_reader import StoreReaderProcessor
        base = {
            "relational_store_type": "sqlite",
            "table": "documents",
            "text_field": "text",
            "id_field": "id",
        }
        if config:
            base.update(config)
        return StoreReaderProcessor(base)

    def _mock_store(self, records):
        store = MagicMock()
        store.get_all_records.return_value = records
        return store

    def test_basic_read(self):
        proc = self._make_processor()
        store = self._mock_store([
            {"id": "d1", "text": "Hello world"},
            {"id": "d2", "text": "Goodbye world"},
        ])
        proc._store = store

        result = proc()
        assert len(result["texts"]) == 2
        assert result["texts"] == ["Hello world", "Goodbye world"]
        assert len(result["documents"]) == 2
        assert result["documents"][0].text == "Hello world"
        assert result["documents"][0].source == "d1"
        assert len(result["source_records"]) == 2

    def test_custom_text_field(self):
        proc = self._make_processor({"text_field": "content"})
        store = self._mock_store([
            {"id": "d1", "content": "Custom field text"},
        ])
        proc._store = store

        result = proc()
        assert result["texts"] == ["Custom field text"]

    def test_custom_id_field(self):
        proc = self._make_processor({"id_field": "doc_id"})
        store = self._mock_store([
            {"doc_id": "my-doc", "text": "Some text"},
        ])
        proc._store = store

        result = proc()
        assert result["documents"][0].source == "my-doc"

    def test_metadata_fields(self):
        proc = self._make_processor({"metadata_fields": ["author", "date"]})
        store = self._mock_store([
            {"id": "d1", "text": "Hello", "author": "Alice", "date": "2024-01-01", "extra": "ignored"},
        ])
        proc._store = store

        result = proc()
        doc = result["documents"][0]
        assert doc.metadata == {"author": "Alice", "date": "2024-01-01"}

    def test_filter_empty_skips_blank(self):
        proc = self._make_processor({"filter_empty": True})
        store = self._mock_store([
            {"id": "d1", "text": "Good text"},
            {"id": "d2", "text": ""},
            {"id": "d3", "text": "   "},
        ])
        proc._store = store

        result = proc()
        assert len(result["texts"]) == 1
        assert result["texts"][0] == "Good text"

    def test_filter_empty_skips_missing(self):
        proc = self._make_processor({"filter_empty": True})
        store = self._mock_store([
            {"id": "d1"},
            {"id": "d2", "text": "Has text"},
        ])
        proc._store = store

        result = proc()
        assert len(result["texts"]) == 1
        assert result["texts"][0] == "Has text"

    def test_filter_empty_disabled_keeps_blank(self):
        proc = self._make_processor({"filter_empty": False})
        store = self._mock_store([
            {"id": "d1", "text": ""},
            {"id": "d2"},
        ])
        proc._store = store

        result = proc()
        assert len(result["texts"]) == 2
        assert result["texts"][0] == ""
        assert result["texts"][1] == ""

    def test_limit_and_offset_passed_to_store(self):
        proc = self._make_processor({"limit": 10, "offset": 5})
        store = self._mock_store([])
        proc._store = store

        proc()
        store.get_all_records.assert_called_once_with("documents", limit=10, offset=5)

    def test_empty_table_returns_empty(self):
        proc = self._make_processor()
        store = self._mock_store([])
        proc._store = store

        result = proc()
        assert result["texts"] == []
        assert result["documents"] == []
        assert result["source_records"] == []

    def test_lazy_store_resolution(self):
        """Store is resolved lazily on first call."""
        proc = self._make_processor()
        assert proc._store is None

        with patch(
            "grapsit.construct.store_reader.resolve_from_pool_or_create"
        ) as mock_resolve:
            mock_store = self._mock_store([{"id": "d1", "text": "hi"}])
            mock_resolve.return_value = mock_store
            proc()
            mock_resolve.assert_called_once()

    def test_registered_in_registry(self):
        from grapsit.core.registry import processor_registry
        assert "store_reader" in processor_registry._factories


# ---------------------------------------------------------------------------
# Builder integration tests
# ---------------------------------------------------------------------------


class TestBuildConfigBuilderStoreReader:
    def test_store_reader_dag_wiring(self):
        """store_reader node is prepended, chunker reads from it."""
        from grapsit.core.builders import BuildConfigBuilder

        builder = BuildConfigBuilder(name="test")
        builder.chunk_store(type="sqlite", sqlite_path=":memory:")
        builder.store_reader(table="articles", text_field="body")
        builder.chunker(method="sentence")
        builder.ner_gliner(labels=["person"])
        builder.graph_writer()

        config = builder.get_config()
        nodes = config["nodes"]
        node_ids = [n["id"] for n in nodes]

        # store_reader is first
        assert node_ids[0] == "store_reader"
        # chunker is second
        assert node_ids[1] == "chunker"

        # Check store_reader config
        sr_node = nodes[0]
        assert sr_node["config"]["table"] == "articles"
        assert sr_node["config"]["text_field"] == "body"

        # Check chunker reads from store_reader
        chunker_node = nodes[1]
        assert chunker_node["inputs"]["texts"]["source"] == "store_reader_result"
        assert chunker_node["inputs"]["documents"]["source"] == "store_reader_result"
        assert chunker_node["requires"] == ["store_reader"]

    def test_no_store_reader_backward_compat(self):
        """Without store_reader, chunker reads from $input as before."""
        from grapsit.core.builders import BuildConfigBuilder

        builder = BuildConfigBuilder(name="test")
        builder.chunker(method="sentence")
        builder.ner_gliner(labels=["person"])
        builder.graph_writer()

        config = builder.get_config()
        nodes = config["nodes"]
        node_ids = [n["id"] for n in nodes]

        assert "store_reader" not in node_ids
        chunker_node = nodes[0]
        assert chunker_node["inputs"]["texts"]["source"] == "$input"
        assert "requires" not in chunker_node

    def test_store_reader_inherits_chunk_store(self):
        """store_reader() picks up builder-level chunk_store config."""
        from grapsit.core.builders import BuildConfigBuilder

        builder = BuildConfigBuilder(name="test")
        builder.chunk_store(type="sqlite", sqlite_path="/data/docs.db")
        builder.store_reader(table="my_table")
        builder.ner_gliner(labels=["org"])
        builder.graph_writer()

        config = builder.get_config()
        sr_config = config["nodes"][0]["config"]
        assert sr_config["relational_store_type"] == "sqlite"
        assert sr_config["sqlite_path"] == "/data/docs.db"
        assert sr_config["table"] == "my_table"


# ---------------------------------------------------------------------------
# SQLite get_all_records integration (real SQLite, no mocking)
# ---------------------------------------------------------------------------


class TestSqliteGetAllRecords:
    def _make_store(self):
        from grapsit.store.relational.sqlite_store import SqliteRelationalStore
        return SqliteRelationalStore(path=":memory:")

    def test_get_all(self):
        store = self._make_store()
        store.write_records("docs", [
            {"id": "d1", "text": "First"},
            {"id": "d2", "text": "Second"},
            {"id": "d3", "text": "Third"},
        ])
        records = store.get_all_records("docs")
        assert len(records) == 3
        ids = {r["id"] for r in records}
        assert ids == {"d1", "d2", "d3"}
        store.close()

    def test_get_all_with_limit(self):
        store = self._make_store()
        store.write_records("docs", [
            {"id": f"d{i}", "text": f"Text {i}"} for i in range(10)
        ])
        records = store.get_all_records("docs", limit=3)
        assert len(records) == 3
        store.close()

    def test_get_all_with_offset(self):
        store = self._make_store()
        store.write_records("docs", [
            {"id": f"d{i}", "text": f"Text {i}"} for i in range(5)
        ])
        all_records = store.get_all_records("docs")
        offset_records = store.get_all_records("docs", offset=2)
        assert len(offset_records) == len(all_records) - 2
        store.close()

    def test_get_all_with_limit_and_offset(self):
        store = self._make_store()
        store.write_records("docs", [
            {"id": f"d{i}", "text": f"Text {i}"} for i in range(10)
        ])
        records = store.get_all_records("docs", limit=3, offset=2)
        assert len(records) == 3
        store.close()

    def test_get_all_empty_table(self):
        store = self._make_store()
        records = store.get_all_records("nonexistent")
        assert records == []
        store.close()

    def test_get_all_deserializes_json(self):
        store = self._make_store()
        store.write_records("meta", [
            {"id": "m1", "data": {"key": "value"}, "tags": ["a", "b"]},
        ])
        records = store.get_all_records("meta")
        assert len(records) == 1
        assert records[0]["data"] == {"key": "value"}
        assert records[0]["tags"] == ["a", "b"]
        store.close()
