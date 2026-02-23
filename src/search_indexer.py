#!/usr/bin/env python3
"""BlackRoad OS - OpenSearch compatible search indexer with SQLite FTS5 backend."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

DB_PATH = Path.home() / ".blackroad" / "opensearch.db"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Index:
    name: str
    mappings: dict[str, Any] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    doc_count: int = 0
    created_at: float = field(default_factory=time.time)


@dataclass
class Document:
    index: str
    id: str
    source: dict[str, Any]
    version: int = 1
    found: bool = True


@dataclass
class SearchHit:
    index: str
    id: str
    score: float
    source: dict[str, Any]


@dataclass
class SearchResponse:
    total: int
    hits: list[SearchHit]
    took_ms: float
    aggregations: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SearchIndexer
# ---------------------------------------------------------------------------

class SearchIndexer:
    """SQLite FTS5-backed search indexer with OpenSearch-compatible API."""

    FIELD_WEIGHTS = {"title": 10.0, "name": 8.0, "description": 5.0, "content": 3.0, "body": 3.0}

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_meta()

    # ------------------------------------------------------------------
    def _init_meta(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS _indices (
                name TEXT PRIMARY KEY,
                mappings TEXT NOT NULL DEFAULT '{}',
                settings TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL
            );
        """)
        self._conn.commit()

    def _doc_table(self, index: str) -> str:
        return f"docs_{index.replace('-', '_').replace('.', '_')}"

    def _fts_table(self, index: str) -> str:
        return f"fts_{index.replace('-', '_').replace('.', '_')}"

    def _ensure_index_tables(self, index: str) -> None:
        dt = self._doc_table(index)
        ft = self._fts_table(index)
        self._conn.executescript(f"""
            CREATE TABLE IF NOT EXISTS "{dt}" (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                version INTEGER NOT NULL DEFAULT 1,
                indexed_at REAL NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS "{ft}"
                USING fts5(id UNINDEXED, body, content="{dt}", content_rowid='rowid');
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def create_index(self, name: str, mappings: dict | None = None, settings: dict | None = None) -> Index:
        mappings = mappings or {}
        settings = settings or {}
        now = time.time()
        self._conn.execute(
            "INSERT OR IGNORE INTO _indices (name, mappings, settings, created_at) VALUES (?,?,?,?)",
            (name, json.dumps(mappings), json.dumps(settings), now),
        )
        self._conn.commit()
        self._ensure_index_tables(name)
        return Index(name=name, mappings=mappings, settings=settings, created_at=now)

    def delete_index(self, name: str) -> bool:
        dt = self._doc_table(name)
        ft = self._fts_table(name)
        self._conn.executescript(f'DROP TABLE IF EXISTS "{ft}"; DROP TABLE IF EXISTS "{dt}";')
        self._conn.execute("DELETE FROM _indices WHERE name=?", (name,))
        self._conn.commit()
        return True

    def get_mapping(self, index: str) -> dict:
        row = self._conn.execute("SELECT mappings FROM _indices WHERE name=?", (index,)).fetchone()
        return json.loads(row["mappings"]) if row else {}

    def put_mapping(self, index: str, mappings: dict) -> bool:
        self._conn.execute("UPDATE _indices SET mappings=? WHERE name=?", (json.dumps(mappings), index))
        self._conn.commit()
        return True

    def index_stats(self, index: str) -> dict:
        dt = self._doc_table(index)
        count = self._conn.execute(f'SELECT COUNT(*) FROM "{dt}"').fetchone()[0]
        return {"index": index, "doc_count": count, "store_size_bytes": self.db_path.stat().st_size}

    # ------------------------------------------------------------------
    # Document CRUD
    # ------------------------------------------------------------------

    def index_document(self, index: str, doc_id: str, source: dict) -> Document:
        self.create_index(index)
        dt = self._doc_table(index)
        ft = self._fts_table(index)
        body = " ".join(str(v) for v in source.values())
        now = time.time()
        self._conn.execute(
            f'INSERT OR REPLACE INTO "{dt}" (id, source, version, indexed_at) VALUES (?,?,1,?)',
            (doc_id, json.dumps(source), now),
        )
        # rebuild FTS shadow
        self._conn.execute(f'INSERT OR REPLACE INTO "{ft}"(id, body) VALUES (?,?)', (doc_id, body))
        self._conn.commit()
        return Document(index=index, id=doc_id, source=source)

    def bulk_index(self, index: str, documents: list[tuple[str, dict]]) -> dict:
        self.create_index(index)
        ok, errors = 0, []
        for doc_id, source in documents:
            try:
                self.index_document(index, doc_id, source)
                ok += 1
            except Exception as exc:
                errors.append({"id": doc_id, "error": str(exc)})
        return {"indexed": ok, "errors": errors}

    def get_document(self, index: str, doc_id: str) -> Document:
        dt = self._doc_table(index)
        row = self._conn.execute(f'SELECT * FROM "{dt}" WHERE id=?', (doc_id,)).fetchone()
        if not row:
            return Document(index=index, id=doc_id, source={}, found=False)
        return Document(index=index, id=doc_id, source=json.loads(row["source"]), version=row["version"])

    def delete_document(self, index: str, doc_id: str) -> bool:
        dt = self._doc_table(index)
        ft = self._fts_table(index)
        self._conn.execute(f'DELETE FROM "{ft}" WHERE id=?', (doc_id,))
        self._conn.execute(f'DELETE FROM "{dt}" WHERE id=?', (doc_id,))
        self._conn.commit()
        return True

    def update_document(self, index: str, doc_id: str, partial: dict) -> Document:
        existing = self.get_document(index, doc_id)
        merged = {**existing.source, **partial}
        return self.index_document(index, doc_id, merged)

    def count(self, index: str, query: str = "") -> int:
        dt = self._doc_table(index)
        if not query:
            return self._conn.execute(f'SELECT COUNT(*) FROM "{dt}"').fetchone()[0]
        ft = self._fts_table(index)
        return self._conn.execute(f'SELECT COUNT(*) FROM "{ft}" WHERE body MATCH ?', (query,)).fetchone()[0]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, index: str, query: str = "", size: int = 10, from_: int = 0,
               filters: dict | None = None) -> SearchResponse:
        t0 = time.time()
        self.create_index(index)
        ft = self._fts_table(index)
        dt = self._doc_table(index)
        hits: list[SearchHit] = []

        if query:
            rows = self._conn.execute(
                f'SELECT f.id, bm25("{ft}") as score, d.source '
                f'FROM "{ft}" f JOIN "{dt}" d ON f.id=d.id '
                f'WHERE f.body MATCH ? ORDER BY score LIMIT ? OFFSET ?',
                (query, size, from_),
            ).fetchall()
        else:
            rows = self._conn.execute(
                f'SELECT id, 1.0 as score, source FROM "{dt}" LIMIT ? OFFSET ?',
                (size, from_),
            ).fetchall()

        for r in rows:
            src = json.loads(r["source"])
            hits.append(SearchHit(index=index, id=r["id"], score=r["score"] or 1.0, source=src))

        total = self.count(index, query)
        return SearchResponse(total=total, hits=hits, took_ms=(time.time() - t0) * 1000)

    # ------------------------------------------------------------------
    # Aggregations
    # ------------------------------------------------------------------

    def aggregate(self, index: str, field_name: str, agg_type: str = "terms", size: int = 10) -> dict:
        dt = self._doc_table(index)
        rows = self._conn.execute(f'SELECT source FROM "{dt}"').fetchall()
        counts: dict[str, int] = {}
        for r in rows:
            src = json.loads(r["source"])
            val = str(src.get(field_name, "__missing__"))
            counts[val] = counts.get(val, 0) + 1
        buckets = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:size]
        return {"buckets": [{"key": k, "doc_count": v} for k, v in buckets]}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="BlackRoad OS Search Indexer")
    sub = parser.add_subparsers(dest="cmd")

    p_indices = sub.add_parser("indices", help="List all indices")

    p_index = sub.add_parser("index", help="Index a document")
    p_index.add_argument("index_name")
    p_index.add_argument("doc_id")
    p_index.add_argument("json_source", help='JSON string, e.g. \'{"title":"hello"}\'')

    p_search = sub.add_parser("search", help="Search documents")
    p_search.add_argument("index_name")
    p_search.add_argument("query", nargs="?", default="")
    p_search.add_argument("--size", type=int, default=10)

    args = parser.parse_args()
    si = SearchIndexer()

    if args.cmd == "indices":
        rows = si._conn.execute("SELECT name, created_at FROM _indices").fetchall()
        for r in rows:
            print(f"  {r['name']}  (created {r['created_at']:.0f})")
        if not rows:
            print("No indices found.")
    elif args.cmd == "index":
        source = json.loads(args.json_source)
        doc = si.index_document(args.index_name, args.doc_id, source)
        print(f"Indexed {doc.id} → {doc.index}")
    elif args.cmd == "search":
        resp = si.search(args.index_name, args.query, size=args.size)
        print(f"Total: {resp.total}  took {resp.took_ms:.1f}ms")
        for h in resp.hits:
            print(f"  [{h.score:.3f}] {h.id}: {h.source}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
