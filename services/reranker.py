"""FlashRank cross-encoder reranker for LangChain Documents."""

import logging
from pathlib import Path
from typing import Optional

from flashrank import Ranker, RerankRequest
from langchain_core.documents import Document

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Absolute path so cache works regardless of process cwd
ROOT_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = (ROOT_DIR / settings.flashrank_cache_dir).resolve()


class RerankerService:
    def __init__(self):
        self._ranker: Optional[Ranker] = None
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _get_ranker(self) -> Ranker:
        if self._ranker is None:
            model_dir = CACHE_DIR / settings.rerank_model
            if model_dir.exists():
                logger.info("Loading cached FlashRank model from %s", model_dir)
            else:
                logger.info("Downloading FlashRank model %s to %s", settings.rerank_model, CACHE_DIR)

            self._ranker = Ranker(
                model_name=settings.rerank_model,
                cache_dir=str(CACHE_DIR),
            )
        return self._ranker

    def warmup(self) -> None:
        """Load model once at startup (download only if not cached)."""
        self._get_ranker()

    def rerank(self, query: str, docs: list[Document], top_k: int = 5) -> list[Document]:
        if not docs:
            return []

        if len(docs) <= top_k:
            return docs

        try:
            passages = [
                {"id": str(i), "text": doc.page_content}
                for i, doc in enumerate(docs)
            ]
            results = self._get_ranker().rerank(RerankRequest(query=query, passages=passages))
            ranked_ids = [int(item["id"]) for item in results[:top_k]]
            return [docs[i] for i in ranked_ids]
        except Exception as exc:
            logger.error("Reranking failed: %s", exc)
            return docs[:top_k]


_reranker: Optional[RerankerService] = None


def get_reranker() -> RerankerService:
    global _reranker
    if _reranker is None:
        _reranker = RerankerService()
    return _reranker
