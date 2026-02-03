"""Data models for query planning and retrieval results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import BaseModel


class QueryComplexity(StrEnum):
    """Classification of query complexity for retrieval strategy selection."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class AbstractionLevel(StrEnum):
    """Component abstraction level for RETR-06 awareness.

    Higher abstraction = broader scope, more autonomous (agents, MCPs).
    Lower abstraction = specific actions, direct control (commands, hooks).
    """

    HIGH = "high"  # agents, mcps - autonomous, broad scope
    MEDIUM = "medium"  # skills - reusable capabilities
    LOW = "low"  # commands, hooks, settings, sandbox - specific actions


@dataclass
class RetrievalPlan:
    """Execution plan for a retrieval query based on complexity analysis."""

    complexity: QueryComplexity
    use_ppr: bool
    use_flow_pruning: bool
    ppr_alpha: float
    max_results: int
    # RETR-06: Suggested component types based on abstraction level awareness
    abstraction_level: AbstractionLevel = AbstractionLevel.MEDIUM
    suggested_types: "list[str]" = field(default_factory=lambda: [])


class RankedComponent(BaseModel):
    """A component with retrieval score and ranking information."""

    component_id: str
    score: float
    rank: int
    source: str  # "vector", "graph", or "fused"
