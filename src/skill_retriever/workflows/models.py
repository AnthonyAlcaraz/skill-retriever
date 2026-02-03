"""Pipeline result models for retrieval orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skill_retriever.nodes.retrieval.context_assembler import RetrievalContext


@dataclass
class ConflictInfo:
    """Information about a conflict between two components."""

    component_a: str
    component_b: str
    reason: str


@dataclass
class PipelineResult:
    """Result of retrieval pipeline execution.

    Contains the assembled context, any detected conflicts,
    dependencies added during resolution, and performance metrics.
    """

    context: RetrievalContext
    conflicts: list[ConflictInfo]
    dependencies_added: list[str]
    latency_ms: float
    cache_hit: bool
    # RETR-06: Abstraction level awareness
    abstraction_level: str = "medium"
    suggested_types: list[str] | None = None
