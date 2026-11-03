"""OpenSearch document store — vector (k-NN) + BM25 in a single index."""

import logging
from typing import Optional

from opensearchpy import OpenSearch, NotFoundError
from langchain_core.documents import Document

from config import get_settings
from services.llm import get_embeddings
from services.metadata import get_metadata_store

logger = logging.getLogger(__name__)
settings = get_settings()

# Index mapping: text field for BM25, knn_vector field for semantic search.
# All metadata fields are keyword/integer so they can be used in filters and aggregations.
INDEX_MAPPING = {
    "settings": {
        "index": {
            "knn": True,
            "knn.algo_param.ef_search": 100,
        }
    },
    "mappings": {
        "properties": {
            "text": {"type": "text", "analyzer": "english"},
            "embedding": {
                "type": "knn_vector",
                "dimension": settings.embed_dim,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "lucene",
                    "parameters": {"ef_construction": 128, "m": 16},
                },
            },
            "user_id": {"type": "keyword"},
            "document_id": {"type": "keyword"},
            "filename": {"type": "keyword"},
            "chunk_index": {"type": "integer"},
            "page": {"type": "integer"},
        }
    },
}


class DocumentStore:
    """OpenSearch-backed store: BM25 and k-NN vector search in one index."""

    def __init__(self):
        self.client = OpenSearch(settings.opensearch_url)
        self.embeddings = get_embeddings()
        self._ensure_index()

    # ------------------------------------------------------------------ #
    # Index lifecycle                                                       #
    # ------------------------------------------------------------------ #

    def _ensure_index(self) -> None:
        """Create the index if it doesn't exist yet."""
        if not self.client.indices.exists(index=settings.opensearch_index):
            self.client.indices.create(index=settings.opensearch_index, body=INDEX_MAPPING)
            logger.info("Created OpenSearch index '%s'", settings.opensearch_index)
        else:
            logger.info("OpenSearch index '%s' already exists", settings.opensearch_index)

    # ------------------------------------------------------------------ #
    # Write                                                                #
    # ------------------------------------------------------------------ #

    def _document_filter(self, document_ids: list[str]) -> dict:
        return {"terms": {"document_id": document_ids}}

    def _user_filter(self, user_id: str) -> dict:
        return {"term": {"user_id": user_id}}

    def resolve_scoped_document_ids(self, user_id: str, document_ids: list[str] | None) -> list[str] | None:
        """Intersect requested IDs with user's indexed docs (reads from SQLite). None = search all user's docs."""
        metadata = get_metadata_store()
        indexed_ids = metadata.get_indexed_document_ids(user_id)
        
        if not indexed_ids:
            return []

        if document_ids:
            return [doc_id for doc_id in document_ids if doc_id in indexed_ids]

        return None

    def add_documents(self, docs: list[Document], user_id: str) -> int:
        """Embed and index a list of LangChain Documents (batched to prevent Ollama crashes)."""
        if not docs:
            return 0

        texts = [doc.page_content for doc in docs]
        
        # Batch embeddings to avoid overwhelming Ollama on large documents
        batch_size = settings.embed_batch_size
        all_embeddings = []
        
        logger.info("Embedding %d chunks in batches of %d", len(texts), batch_size)
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embeddings.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
            logger.debug("Embedded batch %d/%d (%d chunks)", 
                        i // batch_size + 1, 
                        (len(texts) + batch_size - 1) // batch_size,
                        len(batch))

        # Bulk index using the bulk API
        bulk_body = []
        for doc, vector in zip(docs, all_embeddings):
            meta = doc.metadata
            bulk_body.append({"index": {"_index": settings.opensearch_index}})
            bulk_body.append({
                "text": doc.page_content,
                "embedding": vector,
                "user_id": user_id,
                "document_id": meta.get("document_id", ""),
                "filename": meta.get("filename", meta.get("source", "unknown")),
                "chunk_index": meta.get("chunk_index", 0),
                "page": meta.get("page"),
            })

        response = self.client.bulk(body=bulk_body, refresh=True)
        if response.get("errors"):
            logger.error("Bulk index had errors: %s", response)

        logger.info("Indexed %d chunks into OpenSearch", len(docs))
        return len(docs)

    # ------------------------------------------------------------------ #
    # Read — semantic                                                       #
    # ------------------------------------------------------------------ #

    def semantic_search(
        self,
        query: str,
        top_k: int,
        user_id: str,
        document_ids: list[str] | None = None,
        query_vector: list[float] | None = None,
    ) -> list[Document]:
        """k-NN vector search, optionally scoped to specific documents (Lucene engine)."""
        vector = query_vector or self.embeddings.embed_query(query)

        knn_clause: dict = {"vector": vector, "k": top_k}
        
        # Always filter by user_id
        filters = [self._user_filter(user_id)]
        if document_ids:
            filters.append(self._document_filter(document_ids))
        
        if len(filters) == 1:
            knn_clause["filter"] = filters[0]
        else:
            knn_clause["filter"] = {"bool": {"must": filters}}

        body = {
            "size": top_k,
            "query": {"knn": {"embedding": knn_clause}},
            "_source": {"excludes": ["embedding"]},
        }

        response = self.client.search(index=settings.opensearch_index, body=body)
        return self._hits_to_docs(response)

    # ------------------------------------------------------------------ #
    # Read — keyword (BM25)                                                #
    # ------------------------------------------------------------------ #

    def keyword_search(
        self,
        query: str,
        top_k: int,
        user_id: str,
        document_ids: list[str] | None = None,
    ) -> list[Document]:
        """BM25 full-text search, optionally scoped to specific documents."""
        must_clauses = [{"match": {"text": query}}]
        filter_clauses = [self._user_filter(user_id)]
        
        if document_ids:
            filter_clauses.append(self._document_filter(document_ids))

        query_clause = {
            "bool": {
                "must": must_clauses,
                "filter": filter_clauses,
            }
        }

        body = {
            "size": top_k,
            "query": query_clause,
            "_source": {"excludes": ["embedding"]},
        }

        response = self.client.search(index=settings.opensearch_index, body=body)
        return self._hits_to_docs(response)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _hits_to_docs(self, response: dict) -> list[Document]:
        docs = []
        for hit in response["hits"]["hits"]:
            src = hit["_source"]
            metadata = {
                "document_id": src.get("document_id", ""),
                "filename": src.get("filename", "unknown"),
                "chunk_index": src.get("chunk_index", 0),
                "page": src.get("page"),
            }
            docs.append(Document(page_content=src["text"], metadata=metadata))
        return docs

    # ------------------------------------------------------------------ #
    # Utility                                                              #
    # ------------------------------------------------------------------ #

    def delete_document(self, document_id: str, user_id: str) -> bool:
        """Delete all chunks for a document from OpenSearch, scoped to user."""
        body = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"document_id": document_id}},
                        {"term": {"user_id": user_id}},
                    ]
                }
            }
        }
        try:
            resp = self.client.delete_by_query(
                index=settings.opensearch_index,
                body=body,
                refresh=True,
            )
        except NotFoundError:
            return False

        deleted = resp.get("deleted", 0)
        if deleted:
            logger.info("Deleted %d chunks for document_id=%s user_id=%s", deleted, document_id, user_id)
        return deleted > 0

    def health_check(self) -> bool:
        try:
            return self.client.ping()
        except Exception:
            return False


_store: Optional[DocumentStore] = None


def get_store() -> DocumentStore:
    global _store
    if _store is None:
        _store = DocumentStore()
    return _store
