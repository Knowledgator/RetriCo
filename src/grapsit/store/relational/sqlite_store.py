"""SQLite-backed relational store with FTS5 full-text search."""

import json
import sqlite3
from typing import Any, Dict, List, Optional

from .base import BaseRelationalStore


class SqliteRelationalStore(BaseRelationalStore):
    """Relational store backed by SQLite.

    Auto-creates tables on first write. Full-text search via FTS5 virtual tables.
    Uses ``INSERT OR REPLACE`` for upsert semantics keyed on ``id``.
    """

    def __init__(self, path: str = ":memory:"):
        self.path = path
        self._conn: Optional[sqlite3.Connection] = None
        self._known_tables: set = set()

    def _ensure_connection(self):
        if self._conn is None:
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")

    def _ensure_table(self, table: str, columns: List[str]):
        """Create main table and FTS5 shadow table if they don't exist."""
        if table in self._known_tables:
            return
        self._ensure_connection()
        col_defs = ", ".join(f'"{c}" TEXT' for c in columns if c != "id")
        self._conn.execute(
            f'CREATE TABLE IF NOT EXISTS "{table}" ('
            f'"id" TEXT PRIMARY KEY, {col_defs})'
        )
        # FTS5 virtual table for full-text search on all non-id columns
        fts_cols = ", ".join(f'"{c}"' for c in columns if c != "id")
        fts_name = f"{table}_fts"
        self._conn.execute(
            f'CREATE VIRTUAL TABLE IF NOT EXISTS "{fts_name}" '
            f'USING fts5({fts_cols}, content="{table}", content_rowid="rowid")'
        )
        # Triggers to keep FTS index in sync
        non_id = [c for c in columns if c != "id"]
        new_vals = ", ".join(f'new."{c}"' for c in non_id)
        old_vals = ", ".join(f'old."{c}"' for c in non_id)
        fts_col_list = ", ".join(f'"{c}"' for c in non_id)

        self._conn.execute(
            f'CREATE TRIGGER IF NOT EXISTS "{table}_ai" AFTER INSERT ON "{table}" BEGIN '
            f'INSERT INTO "{fts_name}"(rowid, {fts_col_list}) '
            f"VALUES (new.rowid, {new_vals}); END"
        )
        self._conn.execute(
            f'CREATE TRIGGER IF NOT EXISTS "{table}_ad" AFTER DELETE ON "{table}" BEGIN '
            f'INSERT INTO "{fts_name}"("{fts_name}", rowid, {fts_col_list}) '
            f"VALUES ('delete', old.rowid, {old_vals}); END"
        )
        self._conn.execute(
            f'CREATE TRIGGER IF NOT EXISTS "{table}_au" AFTER UPDATE ON "{table}" BEGIN '
            f'INSERT INTO "{fts_name}"("{fts_name}", rowid, {fts_col_list}) '
            f"VALUES ('delete', old.rowid, {old_vals}); "
            f'INSERT INTO "{fts_name}"(rowid, {fts_col_list}) '
            f"VALUES (new.rowid, {new_vals}); END"
        )
        self._conn.commit()
        self._known_tables.add(table)

    @staticmethod
    def _serialize_value(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, (dict, list)):
            return json.dumps(v)
        return str(v)

    @staticmethod
    def _deserialize_row(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
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
        # Collect all columns across records
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
        cols_str = ", ".join(f'"{c}"' for c in all_cols)
        placeholders = ", ".join("?" for _ in all_cols)
        sql = f'INSERT OR REPLACE INTO "{table}" ({cols_str}) VALUES ({placeholders})'

        rows = []
        for r in records:
            rows.append(tuple(self._serialize_value(r.get(c)) for c in all_cols))
        self._conn.executemany(sql, rows)
        self._conn.commit()

    def get_record(self, table: str, record_id: str) -> Optional[Dict[str, Any]]:
        self._ensure_connection()
        try:
            cur = self._conn.execute(
                f'SELECT * FROM "{table}" WHERE id = ?', (record_id,)
            )
        except sqlite3.OperationalError:
            return None
        row = cur.fetchone()
        return self._deserialize_row(row) if row else None

    def search(
        self, table: str, query: str, top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        self._ensure_connection()
        fts_name = f"{table}_fts"
        try:
            cur = self._conn.execute(
                f'SELECT t.*, "{fts_name}".rank '
                f'FROM "{fts_name}" '
                f'JOIN "{table}" t ON t.rowid = "{fts_name}".rowid '
                f'WHERE "{fts_name}" MATCH ? '
                f"ORDER BY rank LIMIT ?",
                (query, top_k),
            )
        except sqlite3.OperationalError:
            return []
        results = []
        for row in cur:
            d = self._deserialize_row(row)
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
            sql += " LIMIT ? OFFSET ?"
            params = [limit, offset]
        elif offset > 0:
            sql += " LIMIT -1 OFFSET ?"
            params = [offset]
        try:
            cur = self._conn.execute(sql, params)
        except sqlite3.OperationalError:
            return []
        return [self._deserialize_row(row) for row in cur]

    def delete_records(self, table: str, record_ids: List[str]):
        if not record_ids:
            return
        self._ensure_connection()
        placeholders = ", ".join("?" for _ in record_ids)
        try:
            self._conn.execute(
                f'DELETE FROM "{table}" WHERE id IN ({placeholders})', record_ids,
            )
            self._conn.commit()
        except sqlite3.OperationalError:
            pass

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            self._known_tables.clear()
