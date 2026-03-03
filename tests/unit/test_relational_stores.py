"""Tests for relational store implementations."""

import pytest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# SQLite store — real tests (uses :memory:, no mocking needed)
# ---------------------------------------------------------------------------


class TestSqliteRelationalStore:
    def _make_store(self):
        from grapsit.store.relational.sqlite_store import SqliteRelationalStore
        return SqliteRelationalStore(path=":memory:")

    def test_write_and_get(self):
        store = self._make_store()
        store.write_records("docs", [
            {"id": "d1", "text": "Hello world", "source": "test.txt"},
            {"id": "d2", "text": "Goodbye world", "source": "test2.txt"},
        ])
        r = store.get_record("docs", "d1")
        assert r is not None
        assert r["id"] == "d1"
        assert r["text"] == "Hello world"
        assert r["source"] == "test.txt"
        store.close()

    def test_get_nonexistent(self):
        store = self._make_store()
        assert store.get_record("docs", "nope") is None
        store.close()

    def test_get_from_nonexistent_table(self):
        store = self._make_store()
        assert store.get_record("nonexistent", "id1") is None
        store.close()

    def test_upsert(self):
        store = self._make_store()
        store.write_records("docs", [{"id": "d1", "text": "original"}])
        store.write_records("docs", [{"id": "d1", "text": "updated"}])
        r = store.get_record("docs", "d1")
        assert r["text"] == "updated"
        store.close()

    def test_delete(self):
        store = self._make_store()
        store.write_records("docs", [
            {"id": "d1", "text": "keep"},
            {"id": "d2", "text": "delete"},
        ])
        store.delete_records("docs", ["d2"])
        assert store.get_record("docs", "d1") is not None
        assert store.get_record("docs", "d2") is None
        store.close()

    def test_delete_nonexistent_table(self):
        store = self._make_store()
        # Should not raise
        store.delete_records("nonexistent", ["id1"])
        store.close()

    def test_search_fts(self):
        store = self._make_store()
        store.write_records("chunks", [
            {"id": "c1", "text": "Albert Einstein was a physicist"},
            {"id": "c2", "text": "Marie Curie won the Nobel Prize"},
            {"id": "c3", "text": "Einstein published relativity theory"},
        ])
        results = store.search("chunks", "Einstein", top_k=10)
        assert len(results) >= 1
        ids = [r["id"] for r in results]
        assert "c1" in ids or "c3" in ids
        store.close()

    def test_search_empty_table(self):
        store = self._make_store()
        results = store.search("nonexistent", "query")
        assert results == []
        store.close()

    def test_write_empty_records(self):
        store = self._make_store()
        store.write_records("docs", [])  # Should not raise
        store.close()

    def test_write_requires_id(self):
        store = self._make_store()
        with pytest.raises(ValueError, match="id"):
            store.write_records("docs", [{"text": "no id"}])
        store.close()

    def test_serializes_dict_values(self):
        store = self._make_store()
        store.write_records("meta", [
            {"id": "m1", "data": {"key": "value"}, "tags": ["a", "b"]},
        ])
        r = store.get_record("meta", "m1")
        assert r["data"] == {"key": "value"}
        assert r["tags"] == ["a", "b"]
        store.close()

    def test_close_and_reopen(self):
        store = self._make_store()
        store.write_records("docs", [{"id": "d1", "text": "test"}])
        store.close()
        # After close, connection is gone; get_record re-opens
        # But :memory: DBs are ephemeral, so data is lost
        assert store.get_record("docs", "d1") is None
        store.close()


# ---------------------------------------------------------------------------
# SQLite factory test
# ---------------------------------------------------------------------------


    def test_get_all_records(self):
        store = self._make_store()
        store.write_records("docs", [
            {"id": "d1", "text": "First"},
            {"id": "d2", "text": "Second"},
        ])
        records = store.get_all_records("docs")
        assert len(records) == 2
        assert {r["id"] for r in records} == {"d1", "d2"}
        store.close()

    def test_get_all_records_with_limit(self):
        store = self._make_store()
        store.write_records("docs", [
            {"id": f"d{i}", "text": f"Text {i}"} for i in range(5)
        ])
        records = store.get_all_records("docs", limit=2)
        assert len(records) == 2
        store.close()

    def test_get_all_records_with_offset(self):
        store = self._make_store()
        store.write_records("docs", [
            {"id": f"d{i}", "text": f"Text {i}"} for i in range(5)
        ])
        records = store.get_all_records("docs", offset=3)
        assert len(records) == 2
        store.close()

    def test_get_all_records_nonexistent_table(self):
        store = self._make_store()
        records = store.get_all_records("nonexistent")
        assert records == []
        store.close()


class TestSqliteFactory:
    def test_create_via_factory(self):
        from grapsit.store.relational import create_relational_store
        store = create_relational_store({"relational_store_type": "sqlite"})
        from grapsit.store.relational.sqlite_store import SqliteRelationalStore
        assert isinstance(store, SqliteRelationalStore)
        assert store.path == ":memory:"
        store.close()

    def test_create_with_path(self):
        from grapsit.store.relational import create_relational_store
        store = create_relational_store({
            "relational_store_type": "sqlite",
            "sqlite_path": "/tmp/test_grapsit.db",
        })
        assert store.path == "/tmp/test_grapsit.db"
        store.close()

    def test_missing_type_raises(self):
        from grapsit.store.relational import create_relational_store
        with pytest.raises(ValueError, match="relational_store_type"):
            create_relational_store({})


# ---------------------------------------------------------------------------
# PostgreSQL store — mocked psycopg
# ---------------------------------------------------------------------------


class TestPostgresRelationalStore:
    def _make_store(self):
        from grapsit.store.relational.postgres_store import PostgresRelationalStore
        return PostgresRelationalStore(
            host="pghost", port=5433, user="pguser",
            password="pgpass", database="pgdb",
        )

    def test_init_params(self):
        store = self._make_store()
        assert store.host == "pghost"
        assert store.port == 5433
        assert store.user == "pguser"
        assert store.password == "pgpass"
        assert store.database == "pgdb"

    def test_lazy_connection(self):
        store = self._make_store()
        assert store._conn is None

    @patch("grapsit.store.relational.postgres_store.psycopg", create=True)
    def test_write_records(self, mock_psycopg_module):
        # We need to patch at import level
        mock_conn = MagicMock()
        import sys
        mock_psycopg = MagicMock()
        mock_psycopg.connect.return_value = mock_conn
        sys.modules["psycopg"] = mock_psycopg

        try:
            store = self._make_store()
            store.write_records("docs", [
                {"id": "d1", "text": "hello"},
            ])
            assert mock_conn.execute.called
        finally:
            del sys.modules["psycopg"]

    @patch("grapsit.store.relational.postgres_store.psycopg", create=True)
    def test_get_record(self, mock_psycopg_module):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ("d1", "hello")
        mock_cursor.description = [("id",), ("text",)]
        mock_conn.execute.return_value = mock_cursor

        import sys
        mock_psycopg = MagicMock()
        mock_psycopg.connect.return_value = mock_conn
        sys.modules["psycopg"] = mock_psycopg

        try:
            store = self._make_store()
            r = store.get_record("docs", "d1")
            assert r is not None
            assert r["id"] == "d1"
            assert r["text"] == "hello"
        finally:
            del sys.modules["psycopg"]

    def test_factory_creates_postgres(self):
        from grapsit.store.relational import create_relational_store
        store = create_relational_store({
            "relational_store_type": "postgres",
            "postgres_host": "myhost",
            "postgres_port": 5433,
        })
        from grapsit.store.relational.postgres_store import PostgresRelationalStore
        assert isinstance(store, PostgresRelationalStore)
        assert store.host == "myhost"
        assert store.port == 5433

    def test_close(self):
        import sys
        mock_psycopg = MagicMock()
        mock_conn = MagicMock()
        mock_psycopg.connect.return_value = mock_conn
        sys.modules["psycopg"] = mock_psycopg

        try:
            store = self._make_store()
            store._ensure_connection()
            store.close()
            assert store._conn is None
            mock_conn.close.assert_called_once()
        finally:
            del sys.modules["psycopg"]


# ---------------------------------------------------------------------------
# Elasticsearch store — mocked elasticsearch client
# ---------------------------------------------------------------------------


class TestElasticsearchRelationalStore:
    def _make_store(self):
        from grapsit.store.relational.elasticsearch_store import ElasticsearchRelationalStore
        return ElasticsearchRelationalStore(
            url="http://eshost:9200",
            api_key="test-key",
            index_prefix="test_",
        )

    def test_init_params(self):
        store = self._make_store()
        assert store.url == "http://eshost:9200"
        assert store.api_key == "test-key"
        assert store.index_prefix == "test_"

    def test_lazy_client(self):
        store = self._make_store()
        assert store._client is None

    def test_index_name(self):
        store = self._make_store()
        assert store._index_name("docs") == "test_docs"
        assert store._index_name("chunks") == "test_chunks"

    def test_write_records_calls_bulk(self):
        import sys
        mock_es_module = MagicMock()
        mock_client = MagicMock()
        mock_es_module.Elasticsearch.return_value = mock_client
        sys.modules["elasticsearch"] = mock_es_module

        try:
            store = self._make_store()
            store.write_records("docs", [
                {"id": "d1", "text": "hello"},
                {"id": "d2", "text": "world"},
            ])
            mock_client.bulk.assert_called_once()
            args = mock_client.bulk.call_args
            ops = args[1]["operations"]
            # Should have 4 items: 2 action + 2 body
            assert len(ops) == 4
            assert ops[0] == {"index": {"_index": "test_docs", "_id": "d1"}}
            assert ops[1] == {"id": "d1", "text": "hello"}
        finally:
            del sys.modules["elasticsearch"]

    def test_get_record(self):
        import sys
        mock_es_module = MagicMock()
        mock_client = MagicMock()
        mock_client.get.return_value = {"_source": {"id": "d1", "text": "hello"}}
        mock_es_module.Elasticsearch.return_value = mock_client
        sys.modules["elasticsearch"] = mock_es_module

        try:
            store = self._make_store()
            r = store.get_record("docs", "d1")
            assert r == {"id": "d1", "text": "hello"}
            mock_client.get.assert_called_once_with(index="test_docs", id="d1")
        finally:
            del sys.modules["elasticsearch"]

    def test_get_record_not_found(self):
        import sys
        mock_es_module = MagicMock()
        mock_client = MagicMock()
        mock_client.get.side_effect = Exception("NotFoundError")
        mock_es_module.Elasticsearch.return_value = mock_client
        sys.modules["elasticsearch"] = mock_es_module

        try:
            store = self._make_store()
            r = store.get_record("docs", "missing")
            assert r is None
        finally:
            del sys.modules["elasticsearch"]

    def test_search(self):
        import sys
        mock_es_module = MagicMock()
        mock_client = MagicMock()
        mock_client.search.return_value = {
            "hits": {"hits": [
                {"_source": {"id": "d1", "text": "Einstein"}},
            ]}
        }
        mock_es_module.Elasticsearch.return_value = mock_client
        sys.modules["elasticsearch"] = mock_es_module

        try:
            store = self._make_store()
            results = store.search("docs", "Einstein", top_k=5)
            assert len(results) == 1
            assert results[0]["id"] == "d1"
        finally:
            del sys.modules["elasticsearch"]

    def test_delete_records(self):
        import sys
        mock_es_module = MagicMock()
        mock_client = MagicMock()
        mock_es_module.Elasticsearch.return_value = mock_client
        sys.modules["elasticsearch"] = mock_es_module

        try:
            store = self._make_store()
            store.delete_records("docs", ["d1", "d2"])
            mock_client.bulk.assert_called_once()
            ops = mock_client.bulk.call_args[1]["operations"]
            assert len(ops) == 2
            assert ops[0] == {"delete": {"_index": "test_docs", "_id": "d1"}}
            assert ops[1] == {"delete": {"_index": "test_docs", "_id": "d2"}}
        finally:
            del sys.modules["elasticsearch"]

    def test_factory_creates_elasticsearch(self):
        from grapsit.store.relational import create_relational_store
        store = create_relational_store({
            "relational_store_type": "elasticsearch",
            "elasticsearch_url": "http://myhost:9200",
            "elasticsearch_api_key": "mykey",
            "elasticsearch_index_prefix": "myapp_",
        })
        from grapsit.store.relational.elasticsearch_store import ElasticsearchRelationalStore
        assert isinstance(store, ElasticsearchRelationalStore)
        assert store.url == "http://myhost:9200"
        assert store.api_key == "mykey"
        assert store.index_prefix == "myapp_"

    def test_write_requires_id(self):
        import sys
        mock_es_module = MagicMock()
        mock_client = MagicMock()
        mock_es_module.Elasticsearch.return_value = mock_client
        sys.modules["elasticsearch"] = mock_es_module

        try:
            store = self._make_store()
            with pytest.raises(ValueError, match="id"):
                store.write_records("docs", [{"text": "no id"}])
        finally:
            del sys.modules["elasticsearch"]

    def test_close(self):
        import sys
        mock_es_module = MagicMock()
        mock_client = MagicMock()
        mock_es_module.Elasticsearch.return_value = mock_client
        sys.modules["elasticsearch"] = mock_es_module

        try:
            store = self._make_store()
            store._ensure_client()
            store.close()
            assert store._client is None
            mock_client.close.assert_called_once()
        finally:
            del sys.modules["elasticsearch"]


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestRelationalStoreRegistry:
    def test_all_types_registered(self):
        from grapsit.store.relational import relational_store_registry
        types = relational_store_registry.list()
        assert "sqlite" in types
        assert "postgres" in types
        assert "elasticsearch" in types

    def test_base_is_abstract(self):
        from grapsit.store.relational.base import BaseRelationalStore
        with pytest.raises(TypeError):
            BaseRelationalStore()
