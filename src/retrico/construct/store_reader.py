"""Store reader processor — pulls texts from a relational store."""

from typing import Any, Dict, List
import logging

from ..core.base import BaseProcessor
from ..core.registry import construct_registry
from ..models.document import Document
from ..store.pool import resolve_from_pool_or_create

logger = logging.getLogger(__name__)


class StoreReaderProcessor(BaseProcessor):
    """Read records from a relational store and produce texts + documents.

    Config keys:
        table: Table/collection to read from (default: ``"documents"``).
        text_field: Column containing the text (default: ``"text"``).
        id_field: Column used as document source/ID (default: ``"id"``).
        metadata_fields: List of extra columns to include in Document metadata.
        limit: Max records to fetch (0 = all, default: 0).
        offset: Number of records to skip (default: 0).
        filter_empty: Skip records with empty/missing text (default: True).

    Plus any relational store config keys (``relational_store_type``, etc.)
    or ``__store_pool__`` for pool-based resolution.

    Output:
        ``{"texts": List[str], "documents": List[Document], "source_records": List[Dict]}``
    """

    default_inputs = {}
    default_output = "store_reader_result"

    def __init__(self, config_dict: Dict[str, Any], pipeline: Any = None):
        super().__init__(config_dict, pipeline)
        self.table: str = config_dict.get("table", "documents")
        self.text_field: str = config_dict.get("text_field", "text")
        self.id_field: str = config_dict.get("id_field", "id")
        self.metadata_fields: List[str] = config_dict.get("metadata_fields", [])
        self.limit: int = config_dict.get("limit", 0)
        self.offset: int = config_dict.get("offset", 0)
        self.filter_empty: bool = config_dict.get("filter_empty", True)
        self._store = None

    def _ensure_store(self):
        if self._store is None:
            self._store = resolve_from_pool_or_create(
                self.config_dict, category="relational",
            )

    def __call__(self, **kwargs) -> Dict[str, Any]:
        self._ensure_store()
        records = self._store.get_all_records(
            self.table, limit=self.limit, offset=self.offset,
        )

        texts: List[str] = []
        documents: List[Document] = []
        source_records: List[Dict] = []

        for record in records:
            text = record.get(self.text_field)
            if text is None:
                if self.filter_empty:
                    logger.debug(
                        "Skipping record %s: missing field %r",
                        record.get(self.id_field, "?"), self.text_field,
                    )
                    continue
                text = ""

            text = str(text)
            if self.filter_empty and not text.strip():
                logger.debug(
                    "Skipping record %s: empty text",
                    record.get(self.id_field, "?"),
                )
                continue

            source = str(record.get(self.id_field, ""))
            metadata = {}
            for field in self.metadata_fields:
                if field in record:
                    metadata[field] = record[field]

            documents.append(Document(
                text=text,
                source=source,
                metadata=metadata,
            ))
            texts.append(text)
            source_records.append(record)

        logger.info(
            "StoreReader: read %d records from table %r (%d after filtering)",
            len(records), self.table, len(texts),
        )

        return {
            "texts": texts,
            "documents": documents,
            "source_records": source_records,
        }


@construct_registry.register("store_reader")
def create_store_reader(config_dict: dict, pipeline=None):
    return StoreReaderProcessor(config_dict, pipeline)
