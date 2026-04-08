"""Unit tests for null byte sanitization in graph stores (Issue #7).

Null bytes (\\x00) in PDF text can break FalkorDB/Neo4j Cypher queries with:
'errMsg: Invalid input at end of input: expected '=' ... errCtx: CYPHER id'

This test suite verifies that all graph stores properly sanitize parameters
before executing queries.
"""

import pytest
from unittest.mock import MagicMock, patch

from retrico.models.document import Chunk, Document
from retrico.models.entity import Entity, EntityMention
from retrico.models.relation import Relation


class TestFalkorDBNullByteSanitization:
    """Test FalkorDB null byte handling."""

    @pytest.fixture
    def mock_falkordb_store(self):
        """Mocked FalkorDB store."""
        from retrico.store.graph.falkordb_store import FalkorDBGraphStore
        store = FalkorDBGraphStore(host="localhost", port=6379, graph="test")

        mock_graph = MagicMock()
        mock_result = MagicMock()
        mock_result.result_set = []
        mock_graph.query.return_value = mock_result

        store._db = MagicMock()
        store._graph = mock_graph

        yield store, mock_graph

    def test_sanitize_params_removes_null_bytes(self):
        """Test that _sanitize_params removes null bytes from strings."""
        from retrico.store.graph.falkordb_store import FalkorDBGraphStore

        params = {
            "text": "Hello\x00World",
            "name": "Test\x00Name",
            "number": 42,
            "list": ["a", "b"],
        }

        sanitized = FalkorDBGraphStore._sanitize_params(params)

        assert sanitized["text"] == "HelloWorld"
        assert sanitized["name"] == "TestName"
        assert sanitized["number"] == 42
        assert sanitized["list"] == ["a", "b"]

    def test_sanitize_params_preserves_non_string_types(self):
        """Test that _sanitize_params preserves non-string parameter types."""
        from retrico.store.graph.falkordb_store import FalkorDBGraphStore

        params = {
            "int_val": 123,
            "float_val": 45.67,
            "bool_val": True,
            "none_val": None,
            "list_val": [1, 2, 3],
            "dict_val": {"key": "value"},
        }

        sanitized = FalkorDBGraphStore._sanitize_params(params)

        assert sanitized == params

    def test_write_chunk_with_null_bytes(self, mock_falkordb_store):
        """Test writing a Chunk with null bytes in text."""
        store, mock_graph = mock_falkordb_store

        chunk = Chunk(
            id="chunk-1",
            document_id="doc-1",
            text="This text has\x00null\x00bytes",
            index=0,
            start_char=0,
            end_char=100,
        )

        store.write_chunk(chunk)

        # Verify query was called
        assert mock_graph.query.called
        call_args = mock_graph.query.call_args
        params = call_args[0][1]

        # Null bytes should be removed from text parameter
        assert "\x00" not in params["text"]
        assert params["text"] == "This text hasnullbytes"

    def test_write_entity_with_null_bytes(self, mock_falkordb_store):
        """Test writing an Entity with null bytes in label."""
        store, mock_graph = mock_falkordb_store

        entity = Entity(
            id="entity-1",
            label="Company\x00Name",
            entity_type="organization",
        )

        store.write_entity(entity)

        assert mock_graph.query.called
        call_args = mock_graph.query.call_args
        params = call_args[0][1]

        # Null bytes should be removed
        assert "\x00" not in params["label"]
        assert params["label"] == "CompanyName"

    def test_write_document_with_null_bytes(self, mock_falkordb_store):
        """Test writing a Document with null bytes."""
        store, mock_graph = mock_falkordb_store

        doc = Document(
            id="doc-1",
            source="file\x00name.pdf",
            text="Content\x00with\x00nulls",
        )

        store.write_document(doc)

        assert mock_graph.query.called
        call_args = mock_graph.query.call_args
        params = call_args[0][1]

        assert "\x00" not in params["source"]
        assert params["source"] == "filename.pdf"

    def test_write_mention_link_with_null_bytes(self, mock_falkordb_store):
        """Test writing mention link with null bytes in mention text."""
        store, mock_graph = mock_falkordb_store

        mention = EntityMention(
            text="Entity\x00Name",
            label="person",
            chunk_id="chunk-1",
            start=0,
            end=10,
        )

        store.write_mention_link("entity-1", "chunk-1", mention)

        assert mock_graph.query.called
        call_args = mock_graph.query.call_args
        params = call_args[0][1]

        assert "\x00" not in params["text"]
        assert params["text"] == "EntityName"

    def test_real_pdf_case_in_fighting(self, mock_falkordb_store):
        """Test real-world case: 'in-fighting' with null byte from PDF parser.

        This reproduces the exact scenario from Issue #7 where PDF text
        'in-fighting' was parsed as 'in-\\x00ghting', causing Cypher errors.
        """
        store, mock_graph = mock_falkordb_store

        # Real text from Family Tree.pdf that caused the bug
        chunk = Chunk(
            id="real-chunk",
            document_id="real-doc",
            text="the in-\x00ghting could result in a shift of power in the Genting kingdom",
            index=0,
        )

        # Should not raise exception
        store.write_chunk(chunk)

        assert mock_graph.query.called
        call_args = mock_graph.query.call_args
        params = call_args[0][1]

        # Null byte removed, text readable
        assert "\x00" not in params["text"]
        assert "in-ghting" in params["text"]  # Hyphen preserved, null byte removed


class TestFalkorDBLiteNullByteSanitization:
    """Test FalkorDB Lite null byte handling (uses same logic as FalkorDB)."""

    @pytest.fixture
    def real_lite_store(self):
        """Real FalkorDBLite instance for integration testing."""
        from retrico.store.graph.falkordb_lite_store import FalkorDBLiteGraphStore
        store = FalkorDBLiteGraphStore(db_path=":memory:", graph="test_null_bytes")
        yield store
        try:
            store.close()
        except:
            pass

    def test_chunk_with_null_bytes_integration(self, real_lite_store):
        """Integration test: write and read chunk with null bytes."""
        chunk = Chunk(
            id="test-chunk-null",
            document_id="test-doc",
            text="Text with\x00multiple\x00null\x00bytes",
            index=0,
        )

        # Should not raise
        real_lite_store.write_chunk(chunk)

        # Verify it was written (null bytes removed)
        result = real_lite_store._run(
            "MATCH (c:Chunk {id: $id}) RETURN c.text AS text",
            {"id": "test-chunk-null"}
        )

        assert len(result) == 1
        # FalkorDB returns scalars directly, not as properties
        saved_text = result[0][0] if isinstance(result[0][0], str) else result[0][0].properties['text']
        assert "\x00" not in saved_text

    def test_entity_with_null_bytes_integration(self, real_lite_store):
        """Integration test: write entity with null bytes in label."""
        entity = Entity(
            id="test-entity-null",
            label="Label\x00With\x00Nulls",
            entity_type="test",
        )

        # Should not raise
        real_lite_store.write_entity(entity)

        # Verify it exists
        result = real_lite_store.get_entity_by_id("test-entity-null")
        assert result is not None
        assert "\x00" not in result["label"]


class TestNeo4jNullByteSanitization:
    """Test Neo4j null byte handling."""

    @pytest.fixture
    def mock_neo4j_store(self):
        """Mocked Neo4j store."""
        from retrico.store.graph.neo4j_store import Neo4jGraphStore

        with patch('retrico.store.graph.neo4j_store.GraphDatabase'):
            store = Neo4jGraphStore(
                uri="bolt://localhost:7687",
                user="neo4j",
                password="password",
            )

            mock_session = MagicMock()
            mock_result = MagicMock()
            mock_result.__iter__ = MagicMock(return_value=iter([]))
            mock_session.run.return_value = mock_result
            mock_session.__enter__ = MagicMock(return_value=mock_session)
            mock_session.__exit__ = MagicMock(return_value=False)

            store._driver = MagicMock()
            store._driver.session.return_value = mock_session

            yield store, mock_session

    def test_neo4j_sanitize_params(self):
        """Test Neo4j _sanitize_params method."""
        from retrico.store.graph.neo4j_store import Neo4jGraphStore

        params = {
            "text": "Hello\x00Neo4j",
            "number": 42,
        }

        sanitized = Neo4jGraphStore._sanitize_params(params)

        assert sanitized["text"] == "HelloNeo4j"
        assert sanitized["number"] == 42

    def test_neo4j_write_chunk_with_null_bytes(self, mock_neo4j_store):
        """Test Neo4j writing chunk with null bytes."""
        store, mock_session = mock_neo4j_store

        chunk = Chunk(
            id="chunk-1",
            document_id="doc-1",
            text="Neo4j\x00text",
            index=0,
        )

        store.write_chunk(chunk)

        assert mock_session.run.called
        call_args = mock_session.run.call_args
        params = call_args[0][1]

        assert "\x00" not in params["text"]
        assert params["text"] == "Neo4jtext"


class TestMemgraphNullByteSanitization:
    """Test Memgraph null byte handling (inherits from Neo4j)."""

    def test_memgraph_inherits_sanitization(self):
        """Verify Memgraph inherits _sanitize_params from Neo4j."""
        from retrico.store.graph.memgraph_store import MemgraphGraphStore
        from retrico.store.graph.neo4j_store import Neo4jGraphStore

        # Memgraph should have the same _sanitize_params method
        assert hasattr(MemgraphGraphStore, '_sanitize_params')
        assert MemgraphGraphStore._sanitize_params == Neo4jGraphStore._sanitize_params

    @pytest.fixture
    def mock_memgraph_store(self):
        """Mocked Memgraph store."""
        from retrico.store.graph.memgraph_store import MemgraphGraphStore

        with patch('retrico.store.graph.neo4j_store.GraphDatabase'):
            store = MemgraphGraphStore(
                uri="bolt://localhost:7687",
                user="",
                password="",
            )

            mock_session = MagicMock()
            mock_result = MagicMock()
            mock_result.__iter__ = MagicMock(return_value=iter([]))
            mock_session.run.return_value = mock_result
            mock_session.__enter__ = MagicMock(return_value=mock_session)
            mock_session.__exit__ = MagicMock(return_value=False)

            store._driver = MagicMock()
            store._driver.session.return_value = mock_session

            yield store, mock_session

    def test_memgraph_write_entity_with_null_bytes(self, mock_memgraph_store):
        """Test Memgraph writing entity with null bytes."""
        store, mock_session = mock_memgraph_store

        entity = Entity(
            id="entity-1",
            label="Memgraph\x00Entity",
            entity_type="test",
        )

        store.write_entity(entity)

        assert mock_session.run.called
        call_args = mock_session.run.call_args
        params = call_args[0][1]

        assert "\x00" not in params["label"]
        assert params["label"] == "MemgraphEntity"


class TestEdgeCases:
    """Edge cases for null byte sanitization."""

    def test_multiple_consecutive_null_bytes(self):
        """Test handling of multiple consecutive null bytes."""
        from retrico.store.graph.falkordb_store import FalkorDBGraphStore

        params = {"text": "a\x00\x00\x00b"}
        sanitized = FalkorDBGraphStore._sanitize_params(params)

        assert sanitized["text"] == "ab"

    def test_null_byte_at_start(self):
        """Test null byte at the start of string."""
        from retrico.store.graph.falkordb_store import FalkorDBGraphStore

        params = {"text": "\x00start"}
        sanitized = FalkorDBGraphStore._sanitize_params(params)

        assert sanitized["text"] == "start"

    def test_null_byte_at_end(self):
        """Test null byte at the end of string."""
        from retrico.store.graph.falkordb_store import FalkorDBGraphStore

        params = {"text": "end\x00"}
        sanitized = FalkorDBGraphStore._sanitize_params(params)

        assert sanitized["text"] == "end"

    def test_only_null_bytes(self):
        """Test string consisting only of null bytes."""
        from retrico.store.graph.falkordb_store import FalkorDBGraphStore

        params = {"text": "\x00\x00\x00"}
        sanitized = FalkorDBGraphStore._sanitize_params(params)

        assert sanitized["text"] == ""

    def test_empty_string_unchanged(self):
        """Test that empty strings are preserved."""
        from retrico.store.graph.falkordb_store import FalkorDBGraphStore

        params = {"text": ""}
        sanitized = FalkorDBGraphStore._sanitize_params(params)

        assert sanitized["text"] == ""

    def test_unicode_with_null_bytes(self):
        """Test Unicode strings with null bytes."""
        from retrico.store.graph.falkordb_store import FalkorDBGraphStore

        params = {"text": "Hello\x00世界\x00мир"}
        sanitized = FalkorDBGraphStore._sanitize_params(params)

        assert sanitized["text"] == "Hello世界мир"
        assert "\x00" not in sanitized["text"]
