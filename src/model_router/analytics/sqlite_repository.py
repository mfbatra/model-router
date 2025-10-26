"""SQLite-backed analytics repository."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from .interfaces import IAnalyticsRepository, RequestRecord

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS requests (
    id TEXT PRIMARY KEY,
    timestamp INTEGER NOT NULL,
    model TEXT NOT NULL,
    prompt_hash TEXT,
    cost REAL NOT NULL,
    latency_ms INTEGER NOT NULL,
    tokens_input INTEGER NOT NULL,
    tokens_output INTEGER NOT NULL,
    success INTEGER NOT NULL
);
"""

_INSERT_SQL = """
INSERT INTO requests (id, timestamp, model, prompt_hash, cost, latency_ms, tokens_input, tokens_output, success)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(id) DO UPDATE SET
    timestamp=excluded.timestamp,
    model=excluded.model,
    prompt_hash=excluded.prompt_hash,
    cost=excluded.cost,
    latency_ms=excluded.latency_ms,
    tokens_input=excluded.tokens_input,
    tokens_output=excluded.tokens_output,
    success=excluded.success;
"""

_SELECT_BY_DATE_SQL = """
SELECT id, timestamp, model, cost, latency_ms, success
FROM requests
WHERE timestamp BETWEEN ? AND ?
ORDER BY timestamp ASC;
"""

_SELECT_BY_MODEL_SQL = """
SELECT id, timestamp, model, cost, latency_ms, success
FROM requests
WHERE model = ?
ORDER BY timestamp ASC;
"""


class SQLiteRepository(IAnalyticsRepository):
    """Lightweight repository focused on persistence only."""

    def __init__(self, db_path: str | Path):
        self._db_path = str(db_path)
        self._ensure_schema()

    def save(self, record: RequestRecord) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                _INSERT_SQL,
                (
                    record.id,
                    int(record.timestamp.timestamp()),
                    record.model,
                    "",  # prompt hashes handled upstream for privacy
                    record.cost,
                    int(record.latency * 1000),
                    0,
                    0,
                    1 if record.success else 0,
                ),
            )
            conn.commit()

    def find_by_date(self, start: datetime, end: datetime) -> List[RequestRecord]:
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                _SELECT_BY_DATE_SQL,
                (int(start.timestamp()), int(end.timestamp())),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def find_by_model(self, model: str) -> List[RequestRecord]:
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(_SELECT_BY_MODEL_SQL, (model,)).fetchall()
        return [self._row_to_record(row) for row in rows]

    def _ensure_schema(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(_CREATE_TABLE_SQL)
            conn.commit()

    @staticmethod
    @staticmethod
    def _row_to_record(row: Tuple[str, int, str, float, int, int]) -> RequestRecord:
        id_, timestamp, model, cost, latency_ms, success = row
        return RequestRecord(
            id=id_,
            timestamp=datetime.fromtimestamp(float(timestamp)),
            model=model,
            cost=cost,
            latency=latency_ms / 1000,
            success=bool(success),
        )
