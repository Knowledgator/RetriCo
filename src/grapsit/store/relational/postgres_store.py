"""PostgreSQL-backed relational store with tsvector full-text search."""

import json
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseRelationalStore


class PostgresRelationalStore(BaseRelationalStore):
    """Relational store backed by PostgreSQL.

    Auto-creates tables with a ``tsv`` tsvector column for full-text search.
    Uses ``INSERT ... ON CONFLICT (id) DO UPDATE`` for upsert semantics.

    Requires the ``psycopg`` package (``pip install psycopg[binary]``).
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        user: str = "postgres",
        password: str = "",
        database: str = "grapsit",
    ):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self._conn = None
        self._known_tables: set = set()

    def _ensure_connection(self):
        if self._conn is not None:
            return
        import psycopg

        self._conn = psycopg.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            dbname=self.database,
            autocommit=True,
        )

    def _ensure_table(self, table: str, columns: List[str]):
        if table in self._known_tables:
            return
        self._ensure_connection()

        non_id = [c for c in columns if c != "id"]
        col_defs = ", ".join(f'"{c}" TEXT' for c in non_id)
        # Create table with tsvector column for FTS
        self._conn.execute(
            f'CREATE TABLE IF NOT EXISTS "{table}" ('
            f'"id" TEXT PRIMARY KEY, {col_defs}, '
            f'"tsv" tsvector GENERATED ALWAYS AS ('
            + " || ' ' || ".join(
                f"coalesce(\"{c}\", '')" for c in non_id
            )
            + ") STORED)"
        )
        # GIN index on tsvector column
        idx_name = f"{table}_tsv_idx"
        self._conn.execute(
            f'CREATE INDEX IF NOT EXISTS "{idx_name}" ON "{table}" USING gin("tsv")'
        )
        self._known_tables.add(table)

    @staticmethod
    def _serialize_value(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, (dict, list)):
            return json.dumps(v)
        return str(v)

    @staticmethod
    def _deserialize_row(row: dict) -> Dict[str, Any]:
        d = dict(row)
        d.pop("tsv", None)
        for k, v in d.items():
            if isinstance(v, str):
                try:
                    parsed = json.loads(v)
                    if isinstance(parsed, (dict, list)):
                        d[k] = parsed
                except (json.JSONDecodeError, ValueError):
                    pass
        return d

    def write_records(self, table: str, records: List[Dict[str, Any]]):
        if not records:
            return

        all_cols = []
        seen = set()
        for r in records:
            for k in r:
                if k not in seen:
                    all_cols.append(k)
                    seen.add(k)
        if "id" not in seen:
            raise ValueError("Each record must contain an 'id' key")

        self._ensure_table(table, all_cols)
        non_id = [c for c in all_cols if c != "id"]
        cols_str = ", ".join(f'"{c}"' for c in all_cols)
        placeholders = ", ".join(f"%({c})s" for c in all_cols)
        update_set = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in non_id)

        sql = (
            f'INSERT INTO "{table}" ({cols_str}) VALUES ({placeholders}) '
            f"ON CONFLICT (id) DO UPDATE SET {update_set}"
        )

        for r in records:
            params = {c: self._serialize_value(r.get(c)) for c in all_cols}
            self._conn.execute(sql, params)

    def get_record(self, table: str, record_id: str) -> Optional[Dict[str, Any]]:
        self._ensure_connection()
        try:
            cur = self._conn.execute(
                f'SELECT * FROM "{table}" WHERE id = %s', (record_id,)
            )
        except Exception:
            return None
        row = cur.fetchone()
        if row is None:
            return None
        # psycopg returns tuples by default; use description to build dict
        cols = [desc[0] for desc in cur.description]
        return self._deserialize_row(dict(zip(cols, row)))

    def search(
        self, table: str, query: str, top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        self._ensure_connection()
        # Convert query words to tsquery with | (OR) between words
        # so rows matching *any* term are returned, ranked by relevance
        words = query.strip().split()
        if not words:
            return []
        ts_query = " | ".join(words)

        try:
            cur = self._conn.execute(
                f'SELECT *, ts_rank("tsv", to_tsquery(%s)) AS rank '
                f'FROM "{table}" '
                f'WHERE "tsv" @@ to_tsquery(%s) '
                f"ORDER BY rank DESC LIMIT %s",
                (ts_query, ts_query, top_k),
            )
        except Exception:
            return []
        cols = [desc[0] for desc in cur.description]
        results = []
        for row in cur:
            d = self._deserialize_row(dict(zip(cols, row)))
            d.pop("rank", None)
            results.append(d)
        return results

    def get_all_records(
        self, table: str, limit: int = 0, offset: int = 0,
    ) -> List[Dict[str, Any]]:
        self._ensure_connection()
        sql = f'SELECT * FROM "{table}"'
        params: list = []
        if limit > 0:
            sql += " LIMIT %s OFFSET %s"
            params = [limit, offset]
        elif offset > 0:
            sql += " OFFSET %s"
            params = [offset]
        try:
            cur = self._conn.execute(sql, params)
        except Exception:
            return []
        cols = [desc[0] for desc in cur.description]
        return [self._deserialize_row(dict(zip(cols, row))) for row in cur]

    @staticmethod
    def _build_filter_clause(
        filters: List[Dict[str, Any]],
    ) -> Tuple[str, List[Any]]:
        """Build a WHERE clause from filter dicts (PostgreSQL %s params)."""
        if not filters:
            return "", []
        clauses: List[str] = []
        params: List[Any] = []
        op_map = {
            "eq": "=", "neq": "!=", "gt": ">", "gte": ">=",
            "lt": "<", "lte": "<=",
        }
        for f in filters:
            field = f["field"]
            operator = f["operator"]
            value = f["value"]
            if operator in op_map:
                clauses.append(f'"{field}" {op_map[operator]} %s')
                params.append(value)
            elif operator == "contains":
                clauses.append(f'"{field}" ILIKE \'%%\' || %s || \'%%\'')
                params.append(value)
            elif operator == "starts_with":
                clauses.append(f'"{field}" ILIKE %s || \'%%\'')
                params.append(value)
        if not clauses:
            return "", []
        return " WHERE " + " AND ".join(clauses), params

    def query_records(
        self,
        table: str,
        filters: Optional[List[Dict[str, Any]]] = None,
        sort_by: Optional[str] = None,
        sort_order: str = "asc",
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        self._ensure_connection()
        where_sql, params = self._build_filter_clause(filters or [])
        sql = f'SELECT * FROM "{table}"{where_sql}'
        if sort_by:
            direction = "DESC" if sort_order.lower() == "desc" else "ASC"
            sql += f' ORDER BY "{sort_by}" {direction}'
        sql += " LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        try:
            cur = self._conn.execute(sql, params)
        except Exception:
            return []
        cols = [desc[0] for desc in cur.description]
        return [self._deserialize_row(dict(zip(cols, row))) for row in cur]

    def delete_records(self, table: str, record_ids: List[str]):
        if not record_ids:
            return
        self._ensure_connection()
        placeholders = ", ".join(["%s"] * len(record_ids))
        try:
            self._conn.execute(
                f'DELETE FROM "{table}" WHERE id IN ({placeholders})',
                record_ids,
            )
        except Exception:
            pass

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            self._known_tables.clear()
