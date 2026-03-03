"""Unit tests for Qdrant vector store (uses qdrant-client in-memory mode)."""

import pytest

qdrant_client = pytest.importorskip("qdrant_client", reason="qdrant-client not installed")

from grapsit.store.vector.qdrant import QdrantVectorStore
from grapsit.store.vector import create_vector_store


class TestQdrantVectorStore:
    """Tests for QdrantVectorStore using in-memory Qdrant."""

    @pytest.fixture
    def store(self):
        # url=None, path=None → in-memory mode
        s = QdrantVectorStore()
        yield s
        s.close()

    @pytest.fixture
    def store_with_index(self, store):
        store.create_index("test", dimension=4)
        return store

    def test_create_index(self, store):
        store.create_index("my_index", dimension=128)
        store.get_embedding("my_index", "nonexistent")

    def test_create_duplicate_index_raises(self, store):
        store.create_index("idx", dimension=64)
        with pytest.raises(ValueError, match="already exists"):
            store.create_index("idx", dimension=64)

    def test_store_and_retrieve_embedding(self, store_with_index):
        vec = [1.0, 0.0, 0.0, 0.0]
        store_with_index.store_embeddings("test", [("item1", vec)])

        result = store_with_index.get_embedding("test", "item1")
        assert result is not None
        assert len(result) == 4
        assert pytest.approx(result[0], abs=1e-5) == 1.0

    def test_get_nonexistent_embedding(self, store_with_index):
        assert store_with_index.get_embedding("test", "missing") is None

    def test_wrong_dimension_raises(self, store_with_index):
        with pytest.raises(ValueError, match="dimension"):
            store_with_index.store_embeddings("test", [("x", [1.0, 2.0])])

    def test_store_multiple_embeddings(self, store_with_index):
        items = [
            ("a", [1.0, 0.0, 0.0, 0.0]),
            ("b", [0.0, 1.0, 0.0, 0.0]),
            ("c", [0.0, 0.0, 1.0, 0.0]),
        ]
        store_with_index.store_embeddings("test", items)

        for item_id, _ in items:
            result = store_with_index.get_embedding("test", item_id)
            assert result is not None
            assert len(result) == 4

    def test_search_similar_basic(self, store_with_index):
        """Cosine similarity search returns correct ordering."""
        store_with_index.store_embeddings(
            "test",
            [
                ("a", [1.0, 0.0, 0.0, 0.0]),
                ("b", [0.9, 0.1, 0.0, 0.0]),
                ("c", [0.0, 0.0, 0.0, 1.0]),
            ],
        )

        results = store_with_index.search_similar("test", [1.0, 0.0, 0.0, 0.0], top_k=3)

        assert len(results) == 3
        # "a" should be most similar (exact match)
        assert results[0][0] == "a"
        assert results[0][1] == pytest.approx(1.0, abs=1e-4)
        # "b" should be second
        assert results[1][0] == "b"
        # "c" should be least similar (orthogonal)
        assert results[2][0] == "c"
        assert results[2][1] == pytest.approx(0.0, abs=1e-4)

    def test_search_similar_top_k(self, store_with_index):
        """top_k limits number of results."""
        store_with_index.store_embeddings(
            "test",
            [
                ("a", [1.0, 0.0, 0.0, 0.0]),
                ("b", [0.0, 1.0, 0.0, 0.0]),
                ("c", [0.0, 0.0, 1.0, 0.0]),
            ],
        )

        results = store_with_index.search_similar("test", [1.0, 0.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0][0] == "a"

    def test_search_empty_index(self, store_with_index):
        results = store_with_index.search_similar("test", [1.0, 0.0, 0.0, 0.0])
        assert results == []

    def test_update_existing_embedding(self, store_with_index):
        """Storing with same ID updates the embedding."""
        store_with_index.store_embeddings("test", [("a", [1.0, 0.0, 0.0, 0.0])])
        store_with_index.store_embeddings("test", [("a", [0.0, 1.0, 0.0, 0.0])])

        # Search should now find "a" close to [0,1,0,0]
        results = store_with_index.search_similar("test", [0.0, 1.0, 0.0, 0.0], top_k=1)
        assert results[0][0] == "a"
        assert results[0][1] == pytest.approx(1.0, abs=1e-4)

    def test_delete_index(self, store_with_index):
        store_with_index.store_embeddings("test", [("a", [1.0, 0.0, 0.0, 0.0])])
        store_with_index.delete_index("test")

        with pytest.raises(KeyError, match="does not exist"):
            store_with_index.get_embedding("test", "a")

    def test_delete_nonexistent_index_raises(self, store):
        with pytest.raises(KeyError, match="does not exist"):
            store.delete_index("nope")

    def test_nonexistent_index_raises(self, store):
        with pytest.raises(KeyError, match="does not exist"):
            store.store_embeddings("nope", [("a", [1.0])])

    def test_close_clears_state(self, store_with_index):
        store_with_index.store_embeddings("test", [("a", [1.0, 0.0, 0.0, 0.0])])
        store_with_index.close()

        with pytest.raises(KeyError):
            store_with_index.get_embedding("test", "a")

    def test_cosine_similarity_normalized(self, store_with_index):
        """Cosine similarity is magnitude-invariant."""
        store_with_index.store_embeddings(
            "test",
            [
                ("unit", [1.0, 0.0, 0.0, 0.0]),
                ("scaled", [10.0, 0.0, 0.0, 0.0]),
            ],
        )

        results = store_with_index.search_similar("test", [5.0, 0.0, 0.0, 0.0], top_k=2)
        assert results[0][1] == pytest.approx(1.0, abs=1e-4)
        assert results[1][1] == pytest.approx(1.0, abs=1e-4)

    def test_store_empty_items(self, store_with_index):
        """Storing empty list is a no-op."""
        store_with_index.store_embeddings("test", [])
        results = store_with_index.search_similar("test", [1.0, 0.0, 0.0, 0.0])
        assert results == []

    def test_lazy_loading(self):
        """Client is not created until first use."""
        store = QdrantVectorStore()
        assert store._client is None
        store.create_index("test", dimension=4)
        assert store._client is not None
        store.close()


class TestCreateVectorStoreQdrant:
    """Tests for the create_vector_store factory with qdrant."""

    def test_qdrant_type_in_memory(self):
        store = create_vector_store({"vector_store_type": "qdrant"})
        assert isinstance(store, QdrantVectorStore)
        store.close()

    def test_qdrant_with_url(self):
        # Just verify construction, don't actually connect
        store = create_vector_store({
            "vector_store_type": "qdrant",
            "qdrant_url": "http://localhost:6333",
        })
        assert isinstance(store, QdrantVectorStore)
        assert store._url == "http://localhost:6333"
        # Don't call close() — no real connection was made