"""Elasticsearch-backed relational store."""

from typing import Any, Dict, List, Optional

from .base import BaseRelationalStore


class ElasticsearchRelationalStore(BaseRelationalStore):
    """Relational store backed by Elasticsearch.

    Each logical "table" maps to an Elasticsearch index (prefixed with
    ``index_prefix``).  Full-text search uses Elasticsearch's built-in
    ``match`` query.

    Requires the ``elasticsearch`` package (``pip install elasticsearch``).
    """

    def __init__(
        self,
        url: str = "http://localhost:9200",
        api_key: Optional[str] = None,
        index_prefix: str = "grapsit_",
    ):
        self.url = url
        self.api_key = api_key
        self.index_prefix = index_prefix
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        from elasticsearch import Elasticsearch

        kwargs: dict = {"hosts": [self.url]}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        self._client = Elasticsearch(**kwargs)

    def _index_name(self, table: str) -> str:
        return f"{self.index_prefix}{table}"

    def write_records(self, table: str, records: List[Dict[str, Any]]):
        if not records:
            return
        self._ensure_client()
        index = self._index_name(table)
        actions = []
        for r in records:
            record_id = r.get("id")
            if record_id is None:
                raise ValueError("Each record must contain an 'id' key")
            actions.append({"index": {"_index": index, "_id": record_id}})
            actions.append(r)
        self._client.bulk(operations=actions, refresh="wait_for")

    def get_record(self, table: str, record_id: str) -> Optional[Dict[str, Any]]:
        self._ensure_client()
        index = self._index_name(table)
        try:
            resp = self._client.get(index=index, id=record_id)
            return resp["_source"]
        except Exception:
            return None

    def search(
        self, table: str, query: str, top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        self._ensure_client()
        index = self._index_name(table)
        try:
            resp = self._client.search(
                index=index,
                query={"match": {"_all": query}},
                size=top_k,
            )
        except Exception:
            # Fallback: multi_match on all fields
            try:
                resp = self._client.search(
                    index=index,
                    query={"multi_match": {"query": query, "type": "best_fields", "fields": ["*"]}},
                    size=top_k,
                )
            except Exception:
                return []
        results = []
        for hit in resp.get("hits", {}).get("hits", []):
            results.append(hit["_source"])
        return results

    def get_all_records(
        self, table: str, limit: int = 0, offset: int = 0,
    ) -> List[Dict[str, Any]]:
        self._ensure_client()
        index = self._index_name(table)
        size = limit if limit > 0 else 10000
        try:
            resp = self._client.search(
                index=index,
                query={"match_all": {}},
                size=size,
                from_=offset,
            )
        except Exception:
            return []
        results = []
        for hit in resp.get("hits", {}).get("hits", []):
            results.append(hit["_source"])
        return results

    def delete_records(self, table: str, record_ids: List[str]):
        if not record_ids:
            return
        self._ensure_client()
        index = self._index_name(table)
        actions = []
        for record_id in record_ids:
            actions.append({"delete": {"_index": index, "_id": record_id}})
        try:
            self._client.bulk(operations=actions, refresh="wait_for")
        except Exception:
            pass

    def close(self):
        if self._client is not None:
            self._client.close()
            self._client = None
