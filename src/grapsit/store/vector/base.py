"""Abstract base class for vector stores."""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


class BaseVectorStore(ABC):
    """Abstract interface for vector embedding storage and similarity search."""

    @abstractmethod
    def create_index(self, name: str, dimension: int):
        """Create a new vector index.

        Args:
            name: Index name.
            dimension: Embedding dimension.
        """
        ...

    @abstractmethod
    def store_embeddings(self, index_name: str, items: List[Tuple[str, List[float]]]):
        """Store embedding vectors.

        Args:
            index_name: Target index.
            items: List of (item_id, embedding_vector) tuples.
        """
        ...

    @abstractmethod
    def search_similar(
        self, index_name: str, query_vector: List[float], top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Find most similar items by cosine similarity.

        Args:
            index_name: Index to search.
            query_vector: Query embedding vector.
            top_k: Number of results to return.

        Returns:
            List of (item_id, similarity_score) tuples, highest first.
        """
        ...

    @abstractmethod
    def get_embedding(self, index_name: str, item_id: str) -> Optional[List[float]]:
        """Retrieve a stored embedding by item ID.

        Args:
            index_name: Index to look up.
            item_id: The item identifier.

        Returns:
            The embedding vector, or None if not found.
        """
        ...

    @abstractmethod
    def delete_index(self, name: str):
        """Delete an index and all its embeddings.

        Args:
            name: Index to delete.
        """
        ...

    @abstractmethod
    def close(self):
        """Release resources."""
        ...
