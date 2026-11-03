"""Configuration management via environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # LLM — OpenRouter
    llm_api_key: str = ""
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "liquid/lfm-1b"

    # Embeddings — local Ollama
    ollama_base_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"

    # Chunking (smaller chunks = better precision for small LLMs)
    chunk_size: int = 500
    chunk_overlap: int = 100

    # Retrieval
    semantic_top_k: int = 20
    bm25_top_k: int = 20
    fusion_top_k: int = 15
    rerank_top_k: int = 3
    rrf_k: int = 60  # RRF constant

    # Reranker
    rerank_model: str = "ms-marco-MiniLM-L-12-v2"
    flashrank_cache_dir: str = "./models/flashrank"

    # Redis (Celery broker + result backend)
    redis_url: str = "redis://localhost:6379/0"

    # OpenSearch
    opensearch_url: str = "http://localhost:9200"
    opensearch_index: str = "rag_documents"
    embed_dim: int = 768  # nomic-embed-text; change if you swap embedding models

    # Local file storage (staged uploads for Celery ingest)
    data_dir: str = "./data"

    # SQLite metadata store (source of truth for documents and jobs)
    sqlite_path: str = "./data/rag.db"

    # Embedding batch size (prevent Ollama crashes on large documents)
    embed_batch_size: int = 50

    # Semantic cache
    cache_similarity_threshold: float = 0.95
    cache_ttl_hours: int = 24

    # Limits
    max_upload_mb: int = 10
    llm_timeout_sec: int = 60
    max_tokens: int = 1024
    temperature: float = 0.1
    # llm_context_window: int = 4096

    # Langfuse (optional — also read directly by langfuse SDK via .env)
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = ""

    class Config:
        env_file = str(PROJECT_ROOT / ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
