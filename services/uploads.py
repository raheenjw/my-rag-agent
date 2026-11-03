"""Local staging for uploaded documents (Celery workers read from disk)."""

import uuid
from pathlib import Path

from config.settings import PROJECT_ROOT, get_settings

settings = get_settings()


def get_upload_dir() -> Path:
    """Return upload directory, creating it if needed."""
    path = (PROJECT_ROOT / settings.data_dir / "uploads").resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def stage_upload(filename: str, content: bytes, document_id: str | None = None) -> tuple[str, Path]:
    """Save upload to disk. Returns (document_id, file_path)."""
    ext = Path(filename).suffix.lower()
    document_id = document_id or str(uuid.uuid4())
    file_path = get_upload_dir() / f"{document_id}{ext}"
    file_path.write_bytes(content)
    return document_id, file_path


def upload_path(document_id: str, filename: str) -> Path:
    """Resolve staged file path for a document."""
    ext = Path(filename).suffix.lower()
    return get_upload_dir() / f"{document_id}{ext}"


def delete_upload(document_id: str) -> None:
    """Remove staged file(s) for a document."""
    upload_dir = get_upload_dir()
    for path in upload_dir.glob(f"{document_id}.*"):
        if path.is_file():
            path.unlink()
