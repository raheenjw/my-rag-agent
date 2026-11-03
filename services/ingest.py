"""Document ingestion with LangChain loaders and text splitters."""

import logging
import time
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_settings
from services.store import get_store
from services.metadata import get_metadata_store

logger = logging.getLogger(__name__)
settings = get_settings()


class IngestService:
    SUPPORTED_TYPES = {".pdf", ".txt", ".md"}

    def __init__(self):
        self.store = get_store()
        self.metadata = get_metadata_store()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )

    def _load_documents(self, file_path: Path, filename: str) -> list[Document]:
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif ext in {".txt", ".md"}:
            loader = TextLoader(str(file_path), encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        docs = loader.load()
        for doc in docs:
            doc.metadata["filename"] = filename
        return docs

    def parse_and_chunk(self, file_path: Path, filename: str, document_id: str) -> dict:
        """Task 1: Load file, split into chunks, save to SQLite staging. Returns metadata for task 2."""
        start_time = time.time()

        ext = file_path.suffix.lower()
        if ext not in self.SUPPORTED_TYPES:
            raise ValueError(f"Unsupported file type: {ext}. Supported: {self.SUPPORTED_TYPES}")

        if not file_path.is_file():
            raise FileNotFoundError(f"Upload not found: {file_path}")

        raw_docs = self._load_documents(file_path, filename)

        if not raw_docs:
            raise ValueError("No extractable text in document")

        chunks = self.splitter.split_documents(raw_docs)
        if not chunks:
            raise ValueError("No chunks generated from document")

        # Prepare chunk records for SQLite
        chunk_records = []
        for i, chunk in enumerate(chunks):
            chunk_records.append({
                "chunk_index": i,
                "text": chunk.page_content,
                "page": chunk.metadata.get("page"),
            })

        # Save to staging table
        self.metadata.save_chunks(document_id, chunk_records)
        
        elapsed = int((time.time() - start_time) * 1000)
        logger.info("Parsed %d chunks from %s in %dms", len(chunk_records), filename, elapsed)

        return {
            "document_id": document_id,
            "filename": filename,
            "chunk_count": len(chunk_records),
        }

    def embed_and_index(self, document_id: str, filename: str, user_id: str) -> dict:
        """Task 2: Read chunks from SQLite, embed, index to OpenSearch, cleanup. Returns final result."""
        start_time = time.time()

        # Read chunks from staging
        chunk_records = self.metadata.get_chunks(document_id)
        if not chunk_records:
            raise ValueError(f"No chunks found in staging for document_id={document_id}")

        # Reconstruct LangChain Document objects with metadata
        docs = []
        for record in chunk_records:
            metadata = {
                "document_id": document_id,
                "filename": filename,
                "chunk_index": record["chunk_index"],
                "page": record["page"],
            }
            docs.append(Document(page_content=record["text"], metadata=metadata))

        # Embed and index to OpenSearch
        chunks_indexed = self.store.add_documents(docs, user_id)

        # Delete chunks from staging on success
        self.metadata.delete_chunks(document_id)

        elapsed = int((time.time() - start_time) * 1000)
        logger.info("Indexed %d chunks from %s in %dms", chunks_indexed, filename, elapsed)

        return {
            "document_id": document_id,
            "filename": filename,
            "chunks_indexed": chunks_indexed,
        }


_ingest_service: Optional[IngestService] = None


def get_ingest_service() -> IngestService:
    global _ingest_service
    if _ingest_service is None:
        _ingest_service = IngestService()
    return _ingest_service
