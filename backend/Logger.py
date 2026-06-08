import sqlite3
import time
import uuid
from datetime import datetime, timezone
from typing import Optional
from contextlib import contextmanager
import os

DB_PATH = 'db/rag_logs.db'
def _get_conn() -> sqlite3.Connection:
        os.makedirs("db", exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    
    
def init_db():
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_logs (
                id                      TEXT PRIMARY KEY,
                timestamp               TEXT NOT NULL,
                query                   TEXT NOT NULL,
                domain_detection_ms     REAL,
                section_detection_ms    REAL,
                retrieval_ms            REAL,
                llm_generation_ms       REAL,
                status                  TEXT NOT NULL,  -- 'success' | 'error'
                error_message           TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON query_logs(timestamp DESC)
        """)
        conn.commit()

class Logging:


    STAGES = ("domain_detection", "section_detection", "retrieval", "llm_generation")

    def __init__(self, query: str):
        self.logid = str(uuid.uuid4())
        self.query = query
        self._status: Optional[str] = None
        self._error: Optional[str] = None
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.latency: dict[str, Optional[float]] = {stage: None for stage in self.STAGES}
        self._saved = False 

    def setStatus(self, status: str, error_message: Optional[str] = None):
        if status not in ("success", "error"):
            raise ValueError("Status must be 'success' or 'error'")
        self._status = status
        self._error = error_message   

    @contextmanager
    def setLatency(self, stage: str):
        start_time = time.perf_counter();
        try:
            yield
        finally:
            self.latency[stage] = round((time.perf_counter() - start_time) * 1000, 2)

    
    def log(self):
        if self._saved:        # ← add this guard
            return
        if not self._status:
            raise ValueError("Call .success() or .failure() before saving.")
 
        with _get_conn() as conn:
            conn.execute(
                """
                INSERT INTO query_logs (
                    id, timestamp, query,
                    domain_detection_ms, section_detection_ms,
                    retrieval_ms, llm_generation_ms,
                    status, error_message
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.logid,
                    self.timestamp,
                    self.query,
                    self.latency["domain_detection"],
                    self.latency["section_detection"],
                    self.latency["retrieval"],
                    self.latency["llm_generation"],
                    self._status,
                    self._error,
                ),
            )
            conn.commit()
        self._saved = True  
        #   self._print_log()


init_db();