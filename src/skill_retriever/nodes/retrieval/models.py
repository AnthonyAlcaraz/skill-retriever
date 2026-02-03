"""Data models for query planning and retrieval results."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel


class QueryComplexity(StrEnum):
    """Classification of query complexity for retrieval strategy selection."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class RetrievalPlan:
    """Execution plan for a retrieval query based on complexity analysis."""

    complexity: QueryComplexity
    use_ppr: bool
    use_flow_pruning: bool
    ppr_alpha: float
    max_results: int


class RankedComponent(BaseModel):
    """A component with retrieval score and ranking information."""

    component_id: str
    score: float
    rank: int
    source: str  # "vector", "graph", or "fused"
