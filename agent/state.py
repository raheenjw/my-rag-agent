"""Agent state definition."""

from typing import Literal, TypedDict


class AgentState(TypedDict, total=False):
    """State passed through the agent graph."""

    # Input
    query: str
    user_id: str
    document_ids: list[str]
    top_k: int | None

    # Injected dependencies
    embeddings: object
    llm: object

    # Output
    answer: str
    sources: list[dict]
    route: Literal["greeting", "guardrail", "rag"]
    done: bool
