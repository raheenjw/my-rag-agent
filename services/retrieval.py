"""Hybrid retrieval with LangChain retrievers + RRF fusion + FlashRank reranking."""

import logging
import time
from typing import Optional

from langchain_core.documents import Document

from config import get_settings
from services.store import get_store
from services.reranker import get_reranker

logger = logging.getLogger(__name__)
settings = get_settings()


def _doc_key(doc: Document) -> str:
    """Stable key for deduplication during fusion."""
    doc_id = doc.metadata.get("document_id", "")
    chunk_idx = doc.metadata.get("chunk_index", "")
    if doc_id != "" and chunk_idx != "":
        return f"{doc_id}:{chunk_idx}"
    return doc.page_content[:200]


def _preview(text: str, max_len: int = 120) -> str:
    text = " ".join(text.split())
    return text[:max_len] + ("…" if len(text) > max_len else "")



def _doc_label(doc: Document) -> str:
    source = doc.metadata.get("filename", doc.metadata.get("source", "unknown"))
    chunk_idx = doc.metadata.get("chunk_index", "?")
    page = doc.metadata.get("page")
    page_str = f" p{page + 1}" if isinstance(page, int) else (f" p{page}" if page else "")
    return f"{source} [chunk {chunk_idx}{page_str}]"


def _log_docs(stage: str, query: str, docs: list[Document], scores: Optional[dict[str, float]] = None) -> None:
    """Log retrieved chunks for a pipeline stage."""
    logger.info("── %s ── query=%r | %d results", stage, query, len(docs))
    for rank, doc in enumerate(docs, start=1):
        key = _doc_key(doc)
        score_part = f" | score={scores[key]:.4f}" if scores and key in scores else ""
        logger.info("  #%d %s%s | %s", rank, _doc_label(doc), score_part, _preview(doc.page_content))


def reciprocal_rank_fusion(
    result_lists: list[list[Document]],
    k: int = 60,
) -> tuple[list[Document], dict[str, float]]:
    """Merge ranked document lists using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    docs_by_key: dict[str, Document] = {}

    for results in result_lists:
        for rank, doc in enumerate(results, start=1):
            key = _doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in docs_by_key:
                docs_by_key[key] = doc

    sorted_docs = sorted(docs_by_key.values(), key=lambda d: scores[_doc_key(d)], reverse=True)
    return sorted_docs, scores


class RetrievalService:
    """Hybrid retrieval: BM25 + semantic search, RRF, then FlashRank."""

    def __init__(self):
        self.store = get_store()
        self.reranker = get_reranker()

    def retrieve(
        self,
        query: str,
        user_id: str,
        rerank_top_k: Optional[int] = None,
        document_ids: Optional[list[str]] = None,
        query_vector: Optional[list[float]] = None,
    ) -> dict:
        rerank_top_k = rerank_top_k or settings.rerank_top_k
        query = query.strip()
        latency: dict[str, int] = {}

        scoped_ids = self.store.resolve_scoped_document_ids(user_id, document_ids)
        if scoped_ids == []:
            logger.info("══ Retrieval skipped ══ no matching documents for scope=%r", document_ids)
            return {"documents": [], "latency": {"retrieve": 0, "rerank": 0}}

        scope_label = scoped_ids if scoped_ids is not None else "all"
        logger.info("══ Retrieval start ══ query=%r | scope=%s", query, scope_label)

        start = time.time()
        semantic_docs = self.store.semantic_search(query, settings.semantic_top_k, user_id, scoped_ids, query_vector=query_vector)
        _log_docs("SEMANTIC", query, semantic_docs)

        bm25_docs = self.store.keyword_search(query, settings.bm25_top_k, user_id, scoped_ids)
        _log_docs("BM25", query, bm25_docs)

        fused_docs, rrf_scores = reciprocal_rank_fusion(
            [bm25_docs, semantic_docs],
            k=settings.rrf_k,
        )
        docs = fused_docs[: settings.fusion_top_k]
        fused_scores = {_doc_key(d): rrf_scores[_doc_key(d)] for d in docs}
        _log_docs("FUSION (RRF)", query, docs, fused_scores)

        latency["retrieve"] = int((time.time() - start) * 1000)

        start = time.time()
        reranked = self.reranker.rerank(query, docs, top_k=rerank_top_k)
        latency["rerank"] = int((time.time() - start) * 1000)
        _log_docs("RERANK (final)", query, reranked)

        logger.info(
            "══ Retrieval done ══ retrieve=%dms rerank=%dms | final_chunks=%d",
            latency["retrieve"],
            latency["rerank"],
            len(reranked),
        )

        return {"documents": reranked, "latency": latency}


_retrieval_service: Optional[RetrievalService] = None


def get_retrieval_service() -> RetrievalService:
    global _retrieval_service
    if _retrieval_service is None:
        _retrieval_service = RetrievalService()
    return _retrieval_service
