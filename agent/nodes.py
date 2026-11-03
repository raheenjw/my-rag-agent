"""Agent nodes — greeting, guardrail, and RAG."""

import logging

from langchain_core.runnables import RunnableConfig

from agent.state import AgentState
from prompts import (
    GREETING_PROMPT,
    GREETING_RESPONSE,
    GUARDRAIL_PROMPT,
    GUARDRAIL_REFUSAL,
    parse_json_response,
)
from services.rag import run_rag_query

logger = logging.getLogger(__name__)


def greeting_node(state: AgentState, config: RunnableConfig) -> dict:
    """Detect greetings and respond without hitting RAG."""
    query = state["query"]
    llm = state["llm"]

    messages = GREETING_PROMPT.format_messages(query=query)
    response = llm.invoke(messages, config=config).content

    result = parse_json_response(
        response,
        defaults={"is_greeting": False},
    )

    if result.get("is_greeting"):
        logger.info("[agent] route=greeting query=%r", query)
        return {
            "answer": GREETING_RESPONSE,
            "sources": [],
            "route": "greeting",
            "done": True,
        }

    return {"done": False}


def guardrail_node(state: AgentState, config: RunnableConfig) -> dict:
    """Block questions about identity, internals, or jailbreaks."""
    if state.get("done"):
        return {}

    query = state["query"]
    llm = state["llm"]

    messages = GUARDRAIL_PROMPT.format_messages(query=query)
    response = llm.invoke(messages, config=config).content

    result = parse_json_response(
        response,
        defaults={"allowed": True},
    )

    if not result.get("allowed", True):
        logger.info("[agent] route=guardrail query=%r", query)
        return {
            "answer": GUARDRAIL_REFUSAL,
            "sources": [],
            "route": "guardrail",
            "done": True,
        }

    return {"done": False}


def rag_node(state: AgentState, config: RunnableConfig) -> dict:
    """Execute RAG pipeline for document questions."""
    if state.get("done"):
        return {}

    query = state["query"]
    logger.info("[agent] route=rag query=%r", query)

    result = run_rag_query(
        query=query,
        user_id=state["user_id"],
        document_ids=state.get("document_ids") or None,
        top_k=state.get("top_k"),
        embeddings=state["embeddings"],
        llm=state["llm"],
        run_config=config,
    )

    return {
        "answer": result.answer,
        "sources": result.sources,
        "route": "rag",
        "done": True,
    }
