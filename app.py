"""FastAPI RAG application."""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from celery_app import celery
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from agent import build_graph
from config import get_settings, setup_logging
from schemas import (
    IngestResponse, IngestJobResponse, IngestStatusResponse,
    ChatRequest, ChatResponse, Source,
    HealthCheck, DocumentSummary, DocumentsResponse,
)
from services.llm import get_llm, get_embeddings, llm_health_check
from services.reranker import get_reranker
from services.semantic_cache import get_semantic_cache
from services.store import get_store
from services.uploads import delete_upload, get_upload_dir, stage_upload
from services.metadata import get_metadata_store
from services.tracing import agent_trace, flush_tracing, get_langfuse_client
from tasks import parse_and_chunk_task, embed_and_index_task

LOGS_DIR = setup_logging()
logger = logging.getLogger(__name__)
settings = get_settings()

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=BASE_DIR / "frontend" / "templates")

_llm = None
_embeddings = None
_cache = None
_agent_graph = None


def _format_ingest_error(error: str | None) -> str:
    """Turn raw exception strings into user-friendly messages."""
    if not error:
        return "Ingestion failed"
    # ValueError('message') or ValueError("message")
    if error.startswith("ValueError(") and error.endswith(")"):
        inner = error[len("ValueError("):-1].strip().strip("'").strip('"')
        return inner or error
    return error


_DOCUMENT_STATUS_MAP = {
    "uploaded": "queued",
    "processing": "started",
    "indexed": "done",
    "failed": "failed",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _llm, _embeddings, _cache, _agent_graph
    logger.info("Starting RAG application...")
    get_upload_dir()
    get_metadata_store()       # init SQLite metadata DB
    get_store()
    _embeddings = get_embeddings()
    _llm = get_llm()           # initialise once, reuse on every request
    _cache = get_semantic_cache()
    _agent_graph = build_graph()
    get_reranker().warmup()
    if get_langfuse_client():
        logger.info("Langfuse tracing active for agent queries")
    logger.info("Agent graph compiled")
    yield
    flush_tracing()
    logger.info("Shutting down RAG application...")


app = FastAPI(
    title="RAG API",
    description="RAG with hybrid search, reranking, and small models",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=BASE_DIR / "frontend" / "static"), name="static")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/v1/health", response_model=HealthCheck, tags=["Health"])
def health_check():
    store = get_store()
    metadata = get_metadata_store()
    opensearch_ok = store.health_check()
    ollama_ok = llm_health_check()
    sqlite_ok = metadata.health_check()
    return HealthCheck(
        status="healthy" if (opensearch_ok and ollama_ok and sqlite_ok) else "degraded",
        opensearch="ok" if opensearch_ok else "error",
        ollama="ok" if ollama_ok else "error",
        sqlite="ok" if sqlite_ok else "error",
    )


@app.post("/v1/ingest", response_model=IngestJobResponse, status_code=202, tags=["Ingest"])
async def ingest_document(
    file: UploadFile = File(...),
    x_user_id: str = Header(..., alias="X-User-Id")
):
    """Upload a document. Returns immediately — poll /v1/ingest/{job_id} for status."""
    contents = await file.read()
    if len(contents) / (1024 * 1024) > settings.max_upload_mb:
        raise HTTPException(status_code=400, detail=f"File too large (max {settings.max_upload_mb}MB)")

    # Stage file and create metadata record
    document_id, _ = stage_upload(file.filename, contents)
    metadata = get_metadata_store()
    metadata.create_document(document_id, x_user_id, file.filename)
    
    # Enqueue task 1; task 2 runs via link on success. Poll task 1's ID.
    result = parse_and_chunk_task.apply_async(
        args=[document_id, file.filename, x_user_id],
        link=embed_and_index_task.s(),
    )
    metadata.create_job(result.id, document_id)
    
    return IngestJobResponse(job_id=result.id)


@app.get("/v1/ingest/{job_id}", response_model=IngestStatusResponse, tags=["Ingest"])
def ingest_status(job_id: str):
    """Poll ingestion job status. SQLite document status is the source of truth."""
    metadata = get_metadata_store()
    
    job = metadata.get_job(job_id)
    document_id = job["document_id"] if job else None
    doc = metadata.get_document_by_id(document_id) if document_id else None

    if doc:
        status = _DOCUMENT_STATUS_MAP.get(doc["status"], doc["status"])

        if status in {"done", "failed"} and job:
            metadata.update_job_status(job_id, status)

        if status == "done":
            return IngestStatusResponse(
                job_id=job_id,
                status="done",
                document_id=document_id,
                result=IngestResponse(
                    document_id=document_id,
                    filename=doc["filename"],
                    chunks_indexed=doc["chunks_indexed"],
                ),
            )

        if status == "failed":
            return IngestStatusResponse(
                job_id=job_id,
                status="failed",
                document_id=document_id,
                error=_format_ingest_error(doc.get("error")),
            )

        return IngestStatusResponse(
            job_id=job_id,
            status=status,
            document_id=document_id,
        )

    # Fallback if job/document not in SQLite yet
    result = celery.AsyncResult(job_id)
    state_map = {"PENDING": "queued", "STARTED": "started", "SUCCESS": "done", "FAILURE": "failed"}
    status = state_map.get(result.state, result.state.lower())

    return IngestStatusResponse(
        job_id=job_id,
        status=status,
        document_id=document_id,
        result=IngestResponse(**result.result) if result.state == "SUCCESS" else None,
        error=_format_ingest_error(str(result.result)) if result.state == "FAILURE" else None,
    )


@app.get("/v1/documents", response_model=DocumentsResponse, tags=["Documents"])
def list_documents(x_user_id: str = Header(..., alias="X-User-Id")):
    """List indexed documents for a user (reads from SQLite)."""
    metadata = get_metadata_store()
    return DocumentsResponse(documents=[DocumentSummary(**doc) for doc in metadata.list_documents(x_user_id)])


@app.delete("/v1/documents/{document_id}", status_code=204, tags=["Documents"])
def delete_document(
    document_id: str,
    x_user_id: str = Header(..., alias="X-User-Id")
):
    """Remove a document and all its indexed chunks from OpenSearch."""
    metadata = get_metadata_store()
    
    # Check ownership in SQLite first (source of truth)
    if not metadata.delete_document(document_id, x_user_id):
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Delete chunks from OpenSearch, staging table, and staged file
    get_store().delete_document(document_id, x_user_id)
    metadata.delete_chunks(document_id)  # cleanup any leftover staging chunks
    delete_upload(document_id)


@app.post("/v1/query", response_model=ChatResponse, tags=["Query"])
def query(
    request: ChatRequest,
    x_user_id: str = Header(..., alias="X-User-Id")
):
    """Ask a question — routed through agent (greeting → guardrail → RAG)."""
    start = time.time()
    metadata = get_metadata_store()

    # Check if user has any indexed documents
    if not metadata.has_indexed_documents(x_user_id):
        raise HTTPException(status_code=404, detail="No documents indexed. Upload via POST /v1/ingest first.")

    # Resolve document scope (validates requested IDs against user's actual indexed docs)
    store = get_store()
    scoped_ids = store.resolve_scoped_document_ids(x_user_id, request.document_ids or None)

    # Early exit if user has no matching documents in the requested scope
    if scoped_ids == []:
        raise HTTPException(status_code=404, detail="No matching documents found for the given scope.")

    agent_input = {
        "query": request.query,
        "user_id": x_user_id,
        "document_ids": scoped_ids or [],
        "top_k": request.top_k,
        "embeddings": _embeddings,
        "llm": _llm,
    }

    try:
        with agent_trace(
            user_id=x_user_id,
            query=request.query,
            document_ids=scoped_ids or [],
            top_k=request.top_k,
        ) as (trace_config, observation):
            result = _agent_graph.invoke(agent_input, config=trace_config)
            if observation is not None:
                observation.update(
                    output={
                        "route": result.get("route"),
                        "answer": result.get("answer"),
                        "sources_count": len(result.get("sources") or []),
                    }
                )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    latency_ms = int((time.time() - start) * 1000)
    logger.info(
        "[query] route=%s user=%s latency_ms=%d query=%r",
        result.get("route", "unknown"), x_user_id, latency_ms, request.query,
    )

    return ChatResponse(
        answer=result["answer"],
        sources=[Source(**s) for s in result.get("sources", [])],
        latency_ms=latency_ms,
    )
