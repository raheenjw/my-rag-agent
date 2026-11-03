"""LangGraph workflow — sequential greeting → guardrail → RAG."""

from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes import greeting_node, guardrail_node, rag_node


def _should_continue(state: AgentState) -> str:
    """Route to END if done, otherwise continue to next node."""
    return "end" if state.get("done") else "continue"


def build_graph():
    """Build and compile the agent graph."""
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("greeting", greeting_node)
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("rag", rag_node)
    
    # Set entry point
    graph.set_entry_point("greeting")
    
    # Conditional edges: greeting -> (end | guardrail)
    graph.add_conditional_edges(
        "greeting",
        _should_continue,
        {"end": END, "continue": "guardrail"},
    )
    
    # Conditional edges: guardrail -> (end | rag)
    graph.add_conditional_edges(
        "guardrail",
        _should_continue,
        {"end": END, "continue": "rag"},
    )
    
    # RAG always ends
    graph.add_edge("rag", END)
    
    return graph.compile()
