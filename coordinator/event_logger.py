from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from core.messages import utc_now


class EventLogger:
    def __init__(self, database_path: Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    client_id INTEGER,
                    round INTEGER,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    client_id INTEGER NOT NULL,
                    round INTEGER NOT NULL,
                    epoch INTEGER NOT NULL,
                    batch INTEGER NOT NULL,
                    global_step INTEGER NOT NULL,
                    loss REAL,
                    accuracy REAL,
                    status TEXT NOT NULL,
                    pid INTEGER
                );
                CREATE TABLE IF NOT EXISTS workers (
                    client_id INTEGER PRIMARY KEY,
                    updated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS rounds (
                    round INTEGER PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    loss REAL NOT NULL,
                    accuracy REAL NOT NULL,
                    duration REAL NOT NULL,
                    active_clients INTEGER NOT NULL,
                    successful_updates INTEGER NOT NULL,
                    selected_clients_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS state (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS hpo_trials (
                    trial_id INTEGER PRIMARY KEY,
                    learning_rate REAL NOT NULL,
                    batch_size INTEGER NOT NULL,
                    local_epochs INTEGER NOT NULL,
                    accuracy REAL,
                    duration REAL,
                    status TEXT NOT NULL,
                    result_dir TEXT NOT NULL
                );
                """
            )

    def log_event(
        self,
        event_type: str,
        source: str = "coordinator",
        client_id: int | None = None,
        round_number: int | None = None,
        timestamp: str | None = None,
        **payload: Any,
    ) -> int:
        stamp = timestamp or utc_now()
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO events(timestamp,event_type,source,client_id,round,payload_json) VALUES(?,?,?,?,?,?)",
                (stamp, event_type, source, client_id, round_number, json.dumps(payload, default=str)),
            )
            return int(cursor.lastrowid)

    def update_event_payload(self, event_id: int, payload: dict) -> None:
        """Enrich an existing lifecycle event without changing timeline order."""
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE events SET payload_json=? WHERE id=?",
                (json.dumps(payload, default=str), event_id),
            )

    def record_telemetry(self, payload: dict) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO telemetry(timestamp,client_id,round,epoch,batch,global_step,loss,accuracy,status,pid)
                VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    payload["timestamp"], payload["client_id"], payload["round"], payload["epoch"],
                    payload["batch"], payload["global_step"], payload.get("loss"), payload.get("accuracy"),
                    payload["status"], payload.get("pid"),
                ),
            )

    def upsert_worker(self, client_id: int, payload: dict) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO workers(client_id,updated_at,payload_json) VALUES(?,?,?)
                ON CONFLICT(client_id) DO UPDATE SET updated_at=excluded.updated_at,payload_json=excluded.payload_json""",
                (client_id, utc_now(), json.dumps(payload, default=str)),
            )

    def record_round(self, round_number: int, metrics: dict) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO rounds(round,timestamp,loss,accuracy,duration,active_clients,successful_updates,selected_clients_json)
                VALUES(?,?,?,?,?,?,?,?)""",
                (
                    round_number, utc_now(), metrics["loss"], metrics["accuracy"], metrics["duration"],
                    metrics["active_clients"], metrics["successful_updates"], json.dumps(metrics["selected_clients"]),
                ),
            )

    def set_state(self, key: str, value: Any) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO state(key,value_json,updated_at) VALUES(?,?,?)
                ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json,updated_at=excluded.updated_at""",
                (key, json.dumps(value, default=str), utc_now()),
            )

    def record_hpo_trial(self, trial: dict) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO hpo_trials(trial_id,learning_rate,batch_size,local_epochs,accuracy,duration,status,result_dir)
                VALUES(?,?,?,?,?,?,?,?)""",
                (
                    trial["trial_id"], trial["learning_rate"], trial["batch_size"], trial["local_epochs"],
                    trial.get("accuracy"), trial.get("duration"), trial["status"], trial["result_dir"],
                ),
            )

    def fetch_all(self, query: str, parameters: tuple = ()) -> list[dict]:
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(query, parameters).fetchall()]
