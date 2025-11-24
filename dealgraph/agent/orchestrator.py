# ABOUTME: Agent orchestrator chaining search, ranking, and reasoning.
# ABOUTME: Provides run_agent() API returning structured logs.

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from pydantic import BaseModel

from . import tools
from ..data.schemas import RankedDeal, DealReasoningOutput


class AgentToolCall(BaseModel):
    tool: str
    inputs: Dict[str, str]
    outputs: Dict[str, str]


class AgentLog(BaseModel):
    query: str
    tool_calls: List[AgentToolCall]
    reasoning_output: DealReasoningOutput


class AgentOrchestratorError(Exception):
    """Raised when the orchestrator fails."""


def run_agent(
    query: str,
    *,
    max_results: int = 10,
    prompt_version: str = "latest",
    search_fn: Optional[Callable[..., List[Dict[str, object]]]] = None,
    rank_fn: Optional[Callable[..., List[RankedDeal]]] = None,
    reason_fn: Optional[Callable[..., DealReasoningOutput]] = None,
) -> AgentLog:
    search_fn = search_fn or tools.tool_graph_semantic_search
    rank_fn = rank_fn or (lambda q, candidates: tools.tool_rank_deals_ml(q, candidates, top_k=max_results))
    reason_fn = reason_fn or tools.tool_deal_reasoner
    tool_calls: List[AgentToolCall] = []

    try:
        # Retrieval
        candidates = search_fn(query=query, top_k=max_results, include_explanations=False)
        tool_calls.append(
            AgentToolCall(
                tool="tool_graph_semantic_search",
                inputs={"query": query, "top_k": str(max_results)},
                outputs={"candidates": str(len(candidates))},
            )
        )

        if not candidates:
            raise AgentOrchestratorError("No retrieval results returned.")

        # Ranking
        ranked_deals = rank_fn(query, candidates)
        tool_calls.append(
            AgentToolCall(
                tool="tool_rank_deals_ml",
                inputs={"candidates": str(len(candidates))},
                outputs={"ranked": str(len(ranked_deals))},
            )
        )

        # Reasoning
        reasoning_output = reason_fn(
            query,
            ranked_deals,
            prompt_version=prompt_version,
        )
        tool_calls.append(
            AgentToolCall(
                tool="tool_deal_reasoner",
                inputs={"ranked": str(len(ranked_deals))},
                outputs={"precedents": str(len(reasoning_output.precedents))},
            )
        )

        return AgentLog(
            query=query,
            tool_calls=tool_calls,
            reasoning_output=reasoning_output,
        )

    except Exception as exc:
        raise AgentOrchestratorError(f"Agent run failed: {exc}") from exc
