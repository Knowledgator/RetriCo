"""Abstract base class for relational (tabular/document) stores."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseRelationalStore(ABC):
    """Abstract interface for tabular/document storage.

    Designed for chunk storage, document metadata, and any tabular data
    that does not belong in the graph. Implementations may be backed by
    SQLite, PostgreSQL, Elasticsearch, or any other tabular store.
    """

    @abstractmethod
    def write_records(self, table: str, records: List[Dict[str, Any]]):
        """Write one or more records to a table (upsert semantics).

        Args:
            table: Table/collection name.
            records: List of dicts, each representing a row. Each dict must
                contain an ``"id"`` key used for upsert.
        """
        ...

    @abstractmethod
    def get_record(self, table: str, record_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single record by ID.

        Args:
            table: Table/collection name.
            record_id: Primary key value.

        Returns:
            The record dict, or None if not found.
        """
        ...

    @abstractmethod
    def search(
        self, table: str, query: str, top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Full-text search within a table.

        Args:
            table: Table/collection name.
            query: Search query string.
            top_k: Maximum results to return.

        Returns:
            List of matching records, ordered by relevance.
        """
        ...

    @abstractmethod
    def delete_records(self, table: str, record_ids: List[str]):
        """Delete records by ID.

        Args:
            table: Table/collection name.
            record_ids: List of primary key values to delete.
        """
        ...

    @abstractmethod
    def close(self):
        """Release resources."""
        ...
