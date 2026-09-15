"""Placeholder LangGraph-style architecture.

The state and node boundaries are intentionally framework-neutral until the
retrieval, safety and observability contracts are reviewed.
"""
from dataclasses import dataclass, field


@dataclass
class ResearchState:
    query: str
    context: list[dict] = field(default_factory=list)
    answer: str = ""


class ResearchGraph:
    async def run(self, query: str) -> ResearchState:
        return ResearchState(query=query)
