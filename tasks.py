"""Celery tasks for background ingestion."""

import logging

from celery_app import celery
from services.ingest import get_ingest_service
from services.metadata import get_metadata_store
from services.store import get_store
from services.uploads import upload_path

logger = logging.getLogger(__name__)


@celery.task(bind=True)
def parse_and_chunk_task(self, document_id: str, filename: str, user_id: str) -> dict:
    """Task 1: Parse file and chunk. Returns metadata for task 2."""
    metadata = get_metadata_store()
    
    try:
        # Update status to processing
        metadata.update_document_status(document_id, "processing")
        metadata.update_job_status(self.request.id, "started")
        
        # Parse and chunk
        file_path = upload_path(document_id, filename)
        result = get_ingest_service().parse_and_chunk(file_path, filename, document_id)
        
        # Pass data to task 2 (including job_id for status updates)
        return {
            "document_id": document_id,
            "filename": filename,
            "user_id": user_id,
            "job_id": self.request.id,
            "chunk_count": result["chunk_count"],
        }
        
    except Exception as exc:
        error_msg = str(exc)
        logger.error("Parse failed for document_id=%s: %s", document_id, error_msg)
        
        # Clean up staging chunks on failure
        metadata.delete_chunks(document_id)
        
        metadata.update_document_status(document_id, "failed", error=error_msg)
        metadata.update_job_status(self.request.id, "failed")
        raise


@celery.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_backoff_max=600, max_retries=3)
def embed_and_index_task(self, task1_result: dict) -> dict:
    """Task 2: Embed and index chunks to OpenSearch. Auto-retries on failure."""
    metadata = get_metadata_store()
    store = get_store()
    
    document_id = task1_result["document_id"]
    filename = task1_result["filename"]
    user_id = task1_result["user_id"]
    job_id = task1_result["job_id"]  # passed from task 1
    
    try:
        # Embed and index
        result = get_ingest_service().embed_and_index(document_id, filename, user_id)
        
        # Mark as indexed on success
        metadata.update_document_status(
            document_id,
            "indexed",
            chunks_indexed=result["chunks_indexed"],
        )
        metadata.update_job_status(job_id, "done")
        
        return result
        
    except Exception as exc:
        # On final failure after all retries, roll back and mark failed
        if self.request.retries >= self.max_retries:
            error_msg = str(exc)
            logger.error("Embed/index failed for document_id=%s after %d retries: %s", 
                        document_id, self.max_retries, error_msg)
            
            # Roll back any partial chunks from OpenSearch
            store.delete_document(document_id, user_id)
            
            # Mark as failed (leave chunks in SQLite for debugging)
            metadata.update_document_status(document_id, "failed", error=error_msg)
            metadata.update_job_status(job_id, "failed")
        else:
            logger.warning("Embed/index retry %d/%d for document_id=%s: %s",
                          self.request.retries + 1, self.max_retries, document_id, exc)
        
        raise
