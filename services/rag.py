"""RAG query service — core retrieval and generation pipeline."""

import logging

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

from prompts import RAG_PROMPT, format_context
from schemas import RAGOutput
from services.retrieval import get_retrieval_service
from services.semantic_cache import get_semantic_cache
from services.metadata import get_metadata_store

logger = logging.getLogger(__name__)


def get_page(doc: Document) -> int | None:
    """Return 1-based page number for display (PyPDFLoader uses 0-based ints)."""
    page = doc.metadata.get("page")
    if isinstance(page, int):
        return page + 1
    return page


def run_rag_query(
    query: str,
    user_id: str,
    document_ids: list[str] | None,
    top_k: int | None,
    *,
    embeddings,
    llm,
    run_config: RunnableConfig | None = None,
) -> RAGOutput:
    """
    Execute the RAG pipeline: cache check → retrieve → generate.
    
    Returns:
        RAGOutput: Answer and source dicts.
    
    Raises:
        ValueError: If no relevant documents found.
        RuntimeError: If LLM generation fails.
    """
    cache = get_semantic_cache()
    metadata = get_metadata_store()
    
    # Compute scope hash for cache isolation
    if document_ids:
        scope_hash = cache.compute_scope_hash(document_ids)
    else:
        scope_hash = cache.compute_scope_hash(metadata.get_indexed_document_ids(user_id))
    
    # Embed query once (reused by cache lookup + retrieval)
    query_vector = embeddings.embed_query(query)
    
    # Check semantic cache
    cached = cache.lookup(query, query_vector, scope_hash)
    if cached:
        logger.info("[rag] cache=hit user=%s scope=%s query=%r", user_id, scope_hash, query)
        return RAGOutput(
            answer=cached["answer"],
            sources=cached["sources"],
        )
    
    logger.info("[rag] cache=miss user=%s scope=%s query=%r", user_id, scope_hash, query)
    
    # Retrieve documents
    docs = get_retrieval_service().retrieve(
        query,
        user_id,
        rerank_top_k=top_k,
        document_ids=document_ids or None,
        query_vector=query_vector,
    )["documents"]
    
    if not docs:
        raise ValueError("No relevant documents found")
    
    # Format context for prompt
    prompt_chunks = [
        {
            "text": doc.page_content,
            "source": doc.metadata.get("filename", "unknown"),
            "page": get_page(doc),
        }
        for doc in docs
    ]
    
    # Generate answer
    try:
        messages = RAG_PROMPT.format_messages(
            context=format_context(prompt_chunks),
            query=query,
        )
        answer = llm.invoke(messages, config=run_config).content
    except Exception as exc:
        logger.error("[rag] LLM generation failed: %s", exc)
        raise RuntimeError(f"LLM service error: {exc}") from exc
    
    # Build sources
    sources = [
        {
            "text": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
            "source": doc.metadata.get("filename", "unknown"),
            "page": get_page(doc),
        }
        for doc in docs
    ]
    
    # Store in cache
    cache.store(
        query=query,
        query_vector=query_vector,
        scope_hash=scope_hash,
        answer=answer,
        sources=sources,
    )
    
    return RAGOutput(answer=answer, sources=sources)
