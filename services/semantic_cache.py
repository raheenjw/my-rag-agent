"""Semantic query cache backed by a dedicated OpenSearch index.

Caches full query results (answer + sources) keyed by query embedding similarity.
Paraphrased queries ("refund policy" vs "how do I get a refund") hit the cache
when cosine similarity exceeds the configured threshold.
"""

import hashlib
import json
import logging
import time
from typing import Optional

from opensearchpy import OpenSearch, NotFoundError
from langchain_core.embeddings import Embeddings

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

CACHE_INDEX = "rag_query_cache"

CACHE_MAPPING = {
    "settings": {
        "index": {
            "knn": True,
            "knn.algo_param.ef_search": 64,
        }
    },
    "mappings": {
        "properties": {
            "query_text": {"type": "text"},
            "query_embedding": {
                "type": "knn_vector",
                "dimension": settings.embed_dim,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "lucene",
                    "parameters": {"ef_construction": 64, "m": 8},
                },
            },
            "scope_hash": {"type": "keyword"},
            "answer": {"type": "text", "index": False},
            "sources_json": {"type": "text", "index": False},
            "created_at": {"type": "long"},
        }
    },
}


class SemanticCache:
    """Semantic query cache using OpenSearch k-NN for similarity matching."""

    def __init__(self, embeddings: Embeddings):
        self.client = OpenSearch(settings.opensearch_url)
        self.embeddings = embeddings
        self.similarity_threshold = settings.cache_similarity_threshold
        self.ttl_seconds = settings.cache_ttl_hours * 3600
        self._ensure_index()

    def _ensure_index(self) -> None:
        if not self.client.indices.exists(index=CACHE_INDEX):
            self.client.indices.create(index=CACHE_INDEX, body=CACHE_MAPPING)
            logger.info("Created semantic cache index '%s'", CACHE_INDEX)

    def compute_scope_hash(self, document_ids) -> str:
        """Deterministic hash of a user's indexed document set."""
        key = ",".join(sorted(document_ids))
        return hashlib.sha256(key.encode()).hexdigest()[:12]

    def lookup(
        self,
        query: str,
        query_vector: list[float],
        scope_hash: str,
    ) -> Optional[dict]:
        """Search cache for a semantically similar query.

        Returns dict with 'answer' and 'sources' on hit, None on miss.
        """
        body = {
            "size": 1,
            "query": {
                "bool": {
                    "must": [
                        {"knn": {"query_embedding": {"vector": query_vector, "k": 1}}}
                    ],
                    "filter": [
                        {"term": {"scope_hash": scope_hash}},
                    ],
                }
            },
            "_source": ["query_text", "answer", "sources_json", "created_at"],
        }

        try:
            resp = self.client.search(index=CACHE_INDEX, body=body)
        except Exception as exc:
            logger.warning("[query-cache] lookup failed (degrading to miss): %s", exc)
            return None

        hits = resp["hits"]["hits"]
        if not hits:
            logger.info("[query-cache] MISS scope=%s query=%r (no entries)", scope_hash, query)
            return None

        hit = hits[0]
        score = hit["_score"]
        # Lucene knn (cosinesimil): score is cosine similarity in [0, 1]
        cosine_sim = score
        if cosine_sim < self.similarity_threshold:
            logger.info(
                "[query-cache] MISS scope=%s query=%r similarity=%.4f (below threshold %.4f, matched %r)",
                scope_hash, query, cosine_sim, self.similarity_threshold,
                hit["_source"]["query_text"],
            )
            return None

        # Check TTL expiry
        created_at = hit["_source"].get("created_at", 0)
        age_seconds = time.time() - (created_at / 1000.0)
        if age_seconds > self.ttl_seconds:
            logger.info(
                "[query-cache] MISS scope=%s query=%r (expired, age=%.0fs > ttl=%ds)",
                scope_hash, query, age_seconds, self.ttl_seconds,
            )
            return None

        logger.info(
            "[query-cache] HIT scope=%s query=%r matched=%r similarity=%.4f",
            scope_hash, query, hit["_source"]["query_text"], cosine_sim,
        )

        src = hit["_source"]
        return {
            "answer": src["answer"],
            "sources": json.loads(src["sources_json"]),
        }

    def store(
        self,
        query: str,
        query_vector: list[float],
        scope_hash: str,
        answer: str,
        sources: list[dict],
    ) -> None:
        """Store a query result in the cache for future semantic matching."""
        doc = {
            "query_text": query,
            "query_embedding": query_vector,
            "scope_hash": scope_hash,
            "answer": answer,
            "sources_json": json.dumps(sources),
            "created_at": int(time.time() * 1000),
        }
        try:
            self.client.index(index=CACHE_INDEX, body=doc, refresh=False)
            logger.info("[query-cache] STORE scope=%s query=%r sources=%d", scope_hash, query, len(sources))
        except Exception as exc:
            logger.warning("[query-cache] store failed (non-fatal): %s", exc)

    def invalidate_scope(self, scope_hash: str) -> None:
        """Clear all cached entries for a given scope (e.g., after document deletion)."""
        try:
            self.client.delete_by_query(
                index=CACHE_INDEX,
                body={"query": {"term": {"scope_hash": scope_hash}}},
                refresh=False,
            )
            logger.info("[query-cache] INVALIDATE scope=%s", scope_hash)
        except NotFoundError:
            pass
        except Exception as exc:
            logger.warning("[query-cache] invalidation failed (non-fatal): %s", exc)

    def health_check(self) -> bool:
        try:
            return self.client.indices.exists(index=CACHE_INDEX)
        except Exception:
            return False


_semantic_cache: Optional[SemanticCache] = None


def get_semantic_cache() -> SemanticCache:
    global _semantic_cache
    if _semantic_cache is None:
        from services.llm import get_embeddings
        _semantic_cache = SemanticCache(get_embeddings())
    return _semantic_cache
