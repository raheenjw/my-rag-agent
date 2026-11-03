"""Langfuse tracing for the agent graph."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger(__name__)

_client: Any | None = None
_init_failed = False


def is_tracing_enabled() -> bool:
    """True when Langfuse API keys are present in the environment."""
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def get_langfuse_client() -> Any | None:
    """Return a Langfuse client, or None if tracing is disabled or init fails."""
    global _client, _init_failed
    if _init_failed:
        return None
    if _client is not None:
        return _client
    if not is_tracing_enabled():
        return None
    try:
        from langfuse import get_client

        _client = get_client()
        logger.info(
            "Langfuse tracing enabled (host=%s)",
            os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL") or "default",
        )
        return _client
    except Exception as exc:
        logger.warning("Langfuse init failed, tracing disabled: %s", exc)
        _init_failed = True
        return None


def build_graph_config(*, user_id: str, observation: Any | None = None) -> dict:
    """Build LangGraph invoke config with Langfuse callback handler."""
    from langfuse.langchain import CallbackHandler

    metadata: dict[str, str] = {
        "langfuse_user_id": user_id,
        "langfuse_session_id": user_id,
    }
    if observation is not None:
        metadata["langfuse_trace_id"] = observation.trace_id
        metadata["langfuse_parent_observation_id"] = observation.id

    return {
        "callbacks": [CallbackHandler()],
        "metadata": metadata,
        "tags": ["rag-agent"],
    }


@contextmanager
def agent_trace(
    *,
    user_id: str,
    query: str,
    document_ids: list[str] | None = None,
    top_k: int | None = None,
) -> Iterator[tuple[dict, Any | None]]:
    """
    Context manager for one agent query.

    Yields (langgraph_config, observation). Call observation.update(output=...)
    after invoke; span is closed and events flushed on exit.
    """
    client = get_langfuse_client()
    if client is None:
        yield {}, None
        return

    observation = client.start_observation(
        as_type="span",
        name="rag-agent-query",
        input={
            "query": query,
            "user_id": user_id,
            "document_ids": document_ids or [],
            "top_k": top_k,
        },
    )
    config = build_graph_config(user_id=user_id, observation=observation)
    try:
        yield config, observation
    except Exception as exc:
        observation.update(level="ERROR", status_message=str(exc))
        raise
    finally:
        observation.end()
        client.flush()


def flush_tracing() -> None:
    """Flush pending Langfuse events (e.g. on shutdown)."""
    client = get_langfuse_client()
    if client is not None:
        client.flush()
