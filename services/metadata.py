"""SQLite metadata store — source of truth for documents and ingest jobs."""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class MetadataStore:
    """SQLite-backed document and job metadata."""

    def __init__(self):
        self.db_path = Path(settings.sqlite_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Create tables if they don't exist. Enable WAL mode for multi-process safety."""
        with sqlite3.connect(self.db_path) as conn:
            # Enable WAL mode for concurrent reads/writes (FastAPI + Celery worker)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")

            # Documents table — source of truth for document lifecycle
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    status TEXT NOT NULL,
                    chunks_indexed INTEGER DEFAULT 0,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)")

            # Ingest jobs table — links Celery task IDs to documents
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ingest_jobs (
                    job_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    finished_at TEXT,
                    FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ingest_jobs_document_id ON ingest_jobs(document_id)")

            # Chunks table — temporary staging between parse and embed tasks
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    page INTEGER,
                    PRIMARY KEY (document_id, chunk_index),
                    FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
                )
            """)

            conn.commit()
            logger.info("SQLite metadata DB initialized at %s", self.db_path)

    def _now(self) -> str:
        return datetime.utcnow().isoformat()

    # ------------------------------------------------------------------ #
    # Documents                                                          #
    # ------------------------------------------------------------------ #

    def create_document(self, document_id: str, user_id: str, filename: str) -> None:
        """Insert a new document with status=uploaded."""
        with sqlite3.connect(self.db_path) as conn:
            now = self._now()
            conn.execute(
                """
                INSERT INTO documents (document_id, user_id, filename, status, created_at, updated_at)
                VALUES (?, ?, ?, 'uploaded', ?, ?)
                """,
                (document_id, user_id, filename, now, now),
            )
            conn.commit()
        logger.info("Created document record: document_id=%s user_id=%s", document_id, user_id)

    def update_document_status(
        self,
        document_id: str,
        status: str,
        chunks_indexed: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        """Update document status, optionally set chunks_indexed and error."""
        with sqlite3.connect(self.db_path) as conn:
            updates = ["status = ?", "updated_at = ?"]
            params = [status, self._now()]

            if chunks_indexed is not None:
                updates.append("chunks_indexed = ?")
                params.append(chunks_indexed)
            if error is not None:
                updates.append("error = ?")
                params.append(error)

            params.append(document_id)
            conn.execute(
                f"UPDATE documents SET {', '.join(updates)} WHERE document_id = ?",
                params,
            )
            conn.commit()
        logger.info("Updated document status: document_id=%s status=%s", document_id, status)

    def get_document(self, document_id: str, user_id: str) -> Optional[dict]:
        """Get a document by ID, scoped to user. Returns None if not found or wrong owner."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM documents WHERE document_id = ? AND user_id = ?",
                (document_id, user_id),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_document_by_id(self, document_id: str) -> Optional[dict]:
        """Get a document by ID (for job status polling)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM documents WHERE document_id = ?",
                (document_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def list_documents(self, user_id: str, status: str = "indexed") -> list[dict]:
        """List all documents for a user, filtered by status (default: indexed only)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT document_id, filename, chunks_indexed, created_at
                FROM documents
                WHERE user_id = ? AND status = ?
                ORDER BY created_at DESC
                """,
                (user_id, status),
            )
            return [dict(row) for row in cursor.fetchall()]

    def delete_document(self, document_id: str, user_id: str) -> bool:
        """Soft-delete a document (set status=deleted). Returns True if found."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE documents SET status = 'deleted', updated_at = ? WHERE document_id = ? AND user_id = ?",
                (self._now(), document_id, user_id),
            )
            conn.commit()
            deleted = cursor.rowcount > 0
            if deleted:
                logger.info("Soft-deleted document: document_id=%s user_id=%s", document_id, user_id)
            return deleted

    def get_indexed_document_ids(self, user_id: str) -> set[str]:
        """Return set of all indexed document IDs for a user."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT document_id FROM documents WHERE user_id = ? AND status = 'indexed'",
                (user_id,),
            )
            return {row[0] for row in cursor.fetchall()}

    def has_indexed_documents(self, user_id: str) -> bool:
        """Check if user has any indexed documents."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT 1 FROM documents WHERE user_id = ? AND status = 'indexed' LIMIT 1",
                (user_id,),
            )
            return cursor.fetchone() is not None

    # ------------------------------------------------------------------ #
    # Ingest jobs                                                        #
    # ------------------------------------------------------------------ #

    def create_job(self, job_id: str, document_id: str) -> None:
        """Record a new ingest job with status=queued."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO ingest_jobs (job_id, document_id, status, created_at) VALUES (?, ?, 'queued', ?)",
                (job_id, document_id, self._now()),
            )
            conn.commit()
        logger.info("Created ingest job: job_id=%s document_id=%s", job_id, document_id)

    def update_job_status(self, job_id: str, status: str) -> None:
        """Update job status. Set finished_at if status is done or failed."""
        with sqlite3.connect(self.db_path) as conn:
            if status in {"done", "failed"}:
                conn.execute(
                    "UPDATE ingest_jobs SET status = ?, finished_at = ? WHERE job_id = ?",
                    (status, self._now(), job_id),
                )
            else:
                conn.execute(
                    "UPDATE ingest_jobs SET status = ? WHERE job_id = ?",
                    (status, job_id),
                )
            conn.commit()

    def get_job(self, job_id: str) -> Optional[dict]:
        """Get job record by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM ingest_jobs WHERE job_id = ?", (job_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    # ------------------------------------------------------------------ #
    # Chunks (temporary staging for ingest)                            #
    # ------------------------------------------------------------------ #

    def save_chunks(self, document_id: str, chunks: list[dict]) -> int:
        """Save parsed chunks to staging table. Returns count inserted."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                "INSERT INTO chunks (document_id, chunk_index, text, page) VALUES (?, ?, ?, ?)",
                [(document_id, c["chunk_index"], c["text"], c.get("page")) for c in chunks],
            )
            conn.commit()
        logger.info("Saved %d chunks for document_id=%s", len(chunks), document_id)
        return len(chunks)

    def get_chunks(self, document_id: str) -> list[dict]:
        """Retrieve all chunks for a document, ordered by chunk_index."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT chunk_index, text, page FROM chunks WHERE document_id = ? ORDER BY chunk_index",
                (document_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def delete_chunks(self, document_id: str) -> int:
        """Delete all chunks for a document. Returns count deleted."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            conn.commit()
            deleted = cursor.rowcount
            if deleted:
                logger.info("Deleted %d chunks for document_id=%s", deleted, document_id)
            return deleted

    # ------------------------------------------------------------------ #
    # Health                                                             #
    # ------------------------------------------------------------------ #

    def health_check(self) -> bool:
        """Basic SQLite health check."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("SELECT 1").fetchone()
            return True
        except Exception as exc:
            logger.error("SQLite health check failed: %s", exc)
            return False


_metadata_store: Optional[MetadataStore] = None


def get_metadata_store() -> MetadataStore:
    global _metadata_store
    if _metadata_store is None:
        _metadata_store = MetadataStore()
    return _metadata_store
