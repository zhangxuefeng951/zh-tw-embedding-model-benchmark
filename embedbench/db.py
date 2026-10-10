"""SQLite 資料庫：設定、自訂預設檔、任務、評測結果（摘要）。

向量與逐題排名等大型資料不放在資料庫，而是存成 data/ 底下的 .npy / .npz 檔。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from embedbench import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS user_presets (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,              -- embedding | reranker
    data TEXT NOT NULL,              -- JSON
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,              -- embedding | bm25 | rerank | prepare_dataset | legacy_import
    status TEXT NOT NULL,            -- queued | running | done | failed | cancelled | interrupted
    title TEXT NOT NULL,
    config TEXT NOT NULL,            -- JSON
    progress TEXT,                   -- JSON
    message TEXT,
    error TEXT,
    result_ids TEXT,                 -- JSON list
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS results (
    id TEXT PRIMARY KEY,
    job_id TEXT,
    kind TEXT NOT NULL,              -- embedding | bm25 | rerank | legacy
    dataset_key TEXT NOT NULL,
    label TEXT NOT NULL,
    model TEXT,
    provider TEXT,
    pair_label TEXT,
    query_prefix TEXT,
    document_prefix TEXT,
    dims INTEGER,
    parent_result_id TEXT,
    num_queries INTEGER NOT NULL,
    num_documents INTEGER NOT NULL,
    metrics TEXT NOT NULL,           -- JSON（預設 K 值的指標）
    config TEXT NOT NULL,            -- JSON（完整設定快照）
    health TEXT,                     -- JSON（向量健康檢查）
    timing TEXT,                     -- JSON
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_results_dataset ON results(dataset_key);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
"""

JSON_COLUMNS = {
    "jobs": ("config", "progress", "result_ids"),
    "results": ("metrics", "config", "health", "timing"),
    "user_presets": ("data",),
}


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class Database:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else config.data_dir() / "bench.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # ---- 基本操作 -------------------------------------------------------

    def execute(self, sql: str, params: tuple | list = ()) -> None:
        with self._lock, self.connect() as conn:
            conn.execute(sql, params)

    def query(self, sql: str, params: tuple | list = (), table: str | None = None) -> list[dict]:
        with self.connect() as conn:
            rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
        if table:
            rows = [self._decode(table, r) for r in rows]
        return rows

    def query_one(self, sql: str, params: tuple | list = (), table: str | None = None) -> dict | None:
        rows = self.query(sql, params, table)
        return rows[0] if rows else None

    def insert(self, table: str, row: dict) -> None:
        row = self._encode(table, row)
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        self.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", list(row.values()))

    def update(self, table: str, key: str, values: dict, key_col: str = "id") -> None:
        values = self._encode(table, values)
        sets = ", ".join(f"{c} = ?" for c in values)
        self.execute(f"UPDATE {table} SET {sets} WHERE {key_col} = ?", [*values.values(), key])

    @staticmethod
    def _encode(table: str, row: dict) -> dict:
        out = dict(row)
        for col in JSON_COLUMNS.get(table, ()):
            if col in out and out[col] is not None and not isinstance(out[col], str):
                out[col] = json.dumps(out[col], ensure_ascii=False)
        return out

    @staticmethod
    def _decode(table: str, row: dict) -> dict:
        for col in JSON_COLUMNS.get(table, ()):
            if row.get(col):
                row[col] = json.loads(row[col])
        return row

    # ---- 設定 -----------------------------------------------------------

    def get_setting(self, key: str, default: Any = None) -> Any:
        row = self.query_one("SELECT value FROM settings WHERE key = ?", (key,))
        return json.loads(row["value"]) if row else default

    def set_setting(self, key: str, value: Any) -> None:
        self.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, json.dumps(value, ensure_ascii=False)),
        )


_db: Database | None = None
_db_lock = threading.Lock()


def get_db() -> Database:
    global _db
    with _db_lock:
        if _db is None or _db.path != config.data_dir() / "bench.db":
            _db = Database()
        return _db


DEFAULT_SETTINGS = {
    "max_concurrent_jobs": 1,
    "default_ks": [1, 3, 5, 10],
}


def get_settings(db: Database | None = None) -> dict:
    db = db or get_db()
    return {k: db.get_setting(k, v) for k, v in DEFAULT_SETTINGS.items()}
