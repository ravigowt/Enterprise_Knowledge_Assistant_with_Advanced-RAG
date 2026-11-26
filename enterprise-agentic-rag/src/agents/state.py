"""
agent/state.py

Shared state schema for the LangGraph pipeline.
Every node receives and returns a partial (or full) AgentState dict.
LangGraph merges the returned dict into the running state automatically.
"""
from typing import TypedDict, List, Dict, Optional, Any


class AgentState(TypedDict, total=False):
    # Original user question
    question: str
    # Current retrieval query
    query: str
    # Routing decision
    route: str  # simple | numeric | multihop
    # Tool bundle (AgentTools instance passed through the graph)
    tools: Any
    # Retrieval outputs
    chunks: List[Dict[str, str]]
    context: str
    sources: List[str]
    # Loop control
    iterations: int
    sufficient: bool
    # Final response
    answer: str
    # Grounding / hallucination check
    grounding: Dict[str, object]
    # Optional debugging
    retrieval_scores: List[float]
    reranker_scores: List[float]
    agent_trace: List[str]