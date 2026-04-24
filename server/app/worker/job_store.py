"""SQLite-backed job persistence. Each job tracks status and output paths."""
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    input_path TEXT NOT NULL,
    input_filename TEXT,
    output_video_path TEXT,
    output_report_path TEXT,
    error_message TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def create(self, job_id: str, input_path: Path, input_filename: str) -> dict:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO jobs (id, status, created_at, input_path, input_filename) VALUES (?, ?, ?, ?, ?)",
                (job_id, "queued", _now(), str(input_path), input_filename),
            )
        got = self.get(job_id)
        assert got is not None
        return got

    def get(self, job_id: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            return dict(row) if row else None

    def list_all(self, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def update(self, job_id: str, **fields) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k} = ?" for k in fields)
        vals = list(fields.values()) + [job_id]
        with self._lock, self._connect() as conn:
            conn.execute(f"UPDATE jobs SET {cols} WHERE id = ?", vals)

    def mark_processing(self, job_id: str) -> None:
        self.update(job_id, status="processing", started_at=_now())

    def mark_completed(self, job_id: str, video_path: Path, report_path: Path) -> None:
        self.update(
            job_id,
            status="completed",
            completed_at=_now(),
            output_video_path=str(video_path),
            output_report_path=str(report_path),
        )

    def mark_failed(self, job_id: str, error: str) -> None:
        self.update(job_id, status="failed", completed_at=_now(), error_message=error)
