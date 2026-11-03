"""LLM via OpenRouter + embeddings via local Ollama."""

import logging
from typing import List

import httpx
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class NomicOllamaEmbeddings(Embeddings):
    """Ollama embeddings with nomic-embed-text asymmetric prefixes."""

    def __init__(self, model: str, base_url: str):
        self._client = OllamaEmbeddings(model=model, base_url=base_url)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        prefixed = [f"search_document: {t}" for t in texts]
        return self._client.embed_documents(prefixed)

    def embed_query(self, text: str) -> List[float]:
        return self._client.embed_query(f"search_query: {text}")


def get_embeddings() -> Embeddings:
    """Embedding model via local Ollama."""
    model = settings.embed_model
    base_url = settings.ollama_base_url

    if "nomic" in model.lower():
        return NomicOllamaEmbeddings(model=model, base_url=base_url)

    return OllamaEmbeddings(model=model, base_url=base_url)


def get_llm() -> BaseChatModel:
    """LFM via OpenRouter (OpenAI-compatible API)."""
    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )


def llm_health_check() -> bool:
    """Check if OpenRouter is reachable."""
    try:
        response = httpx.get("https://openrouter.ai/api/v1/models", timeout=5)
        return response.status_code == 200
    except Exception as exc:
        logger.error("LLM health check failed: %s", exc)
        return False
