"""Pydantic schemas for API request/response models."""

from pydantic import BaseModel, Field
from typing import Optional


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    chunks_indexed: int


class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    chunks_indexed: int


class DocumentsResponse(BaseModel):
    documents: list[DocumentSummary]


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of chunks to use")
    document_ids: list[str] = Field(
        default_factory=list,
        description="Limit search to these document IDs. Empty = search all.",
    )


class Source(BaseModel):
    text: str
    source: str
    page: Optional[int] = None


class RAGOutput(BaseModel):
    """Internal RAG pipeline output (before HTTP response)."""
    answer: str
    sources: list[dict]


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    latency_ms: int


class IngestJobResponse(BaseModel):
    job_id: str
    status: str = "queued"


class IngestStatusResponse(BaseModel):
    job_id: str
    status: str                        # queued | started | done | failed
    document_id: Optional[str] = None  # document ID from metadata DB
    result: Optional[IngestResponse] = None
    error: Optional[str] = None


class HealthCheck(BaseModel):
    status: str
    opensearch: str
    ollama: str
    sqlite: str
