"""SQLite is the only persistence dependency. Credentials never enter this module."""

import json
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, session TEXT NOT NULL, created TEXT NOT NULL,
                    protocol TEXT NOT NULL, model TEXT NOT NULL, endpoint TEXT NOT NULL,
                    success INTEGER NOT NULL, seconds REAL NOT NULL, payload TEXT NOT NULL,
                    image BLOB NOT NULL
                );
            """)

    @contextmanager
    def connect(self):
        with closing(sqlite3.connect(self.path, timeout=10)) as db, db:
            db.row_factory = sqlite3.Row
            yield db

    def save(self, run: dict, session: str, image: bytes):
        with self.connect() as db:
            db.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run["id"],
                    session,
                    run["created"],
                    run["protocol"],
                    run["model"],
                    run["endpoint"],
                    int(run["success"]),
                    run["seconds"],
                    json.dumps(run, ensure_ascii=False),
                    image,
                ),
            )

    def get(self, run_id: str, session: str) -> tuple[dict, bytes]:
        with self.connect() as db:
            row = db.execute(
                "SELECT payload, image FROM runs WHERE id=? AND session=?", (run_id, session)
            ).fetchone()
        if not row:
            raise ValueError("Run not found.")
        return json.loads(row["payload"]), row["image"]

    def history(self, session: str) -> list[sqlite3.Row]:
        with self.connect() as db:
            return db.execute(
                "SELECT id, created, model, success FROM runs "
                "WHERE session=? ORDER BY created DESC LIMIT 100",
                (session,),
            ).fetchall()
