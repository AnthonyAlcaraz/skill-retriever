"""Feedback engine for learning implicit graph relationships (LRNG-06).

Analyzes usage patterns and outcomes to:
1. Suggest new edges (BUNDLES_WITH, CONFLICTS_WITH)
2. Adjust edge weights based on real-world usage
3. Detect deprecated/superseded components
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SuggestionType(StrEnum):
    """Types of edge suggestions."""

    BUNDLES_WITH = "bundles_with"  # Frequently selected together
    CONFLICTS_WITH = "conflicts_with"  # Frequently fail together
    SUPERSEDES = "supersedes"  # One replaces another
    STRENGTHEN = "strengthen"  # Increase existing edge weight
    WEAKEN = "weaken"  # Decrease existing edge weight


class EdgeSuggestion(BaseModel):
    """A suggested edge modification."""

    suggestion_type: SuggestionType
    source_id: str
    target_id: str
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")
    evidence: str = ""  # Human-readable explanation
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    reviewed: bool = False
    accepted: bool = False


class FeedbackEngineState(BaseModel):
    """Persistent state for the feedback engine."""

    suggestions: list[EdgeSuggestion] = Field(default_factory=list)
    applied_suggestions: list[EdgeSuggestion] = Field(default_factory=list)
    last_analysis: datetime | None = None


@dataclass
class AnalysisConfig:
    """Configuration for feedback analysis."""

    min_co_selections: int = 3  # Min co-selections to suggest BUNDLES_WITH
    min_co_failures: int = 2  # Min co-failures to suggest CONFLICTS_WITH
    min_confidence: float = 0.6  # Minimum confidence for suggestions
    max_suggestions: int = 50  # Maximum pending suggestions


class FeedbackEngine:
    """Analyzes usage patterns to suggest graph improvements.

    Workflow:
    1. analyze() reads ComponentMemory and OutcomeTracker
    2. Generates EdgeSuggestions based on patterns
    3. Human reviews suggestions via MCP tool
    4. apply_suggestion() modifies graph if accepted
    """

    def __init__(
        self,
        state_path: Path | None = None,
        config: AnalysisConfig | None = None,
    ) -> None:
        """Initialize the feedback engine.

        Args:
            state_path: Path to persist state.
            config: Analysis configuration.
        """
        self.state_path = state_path or (
            Path.home() / ".skill-retriever" / "feedback-engine.json"
        )
        self.config = config or AnalysisConfig()
        self.state = FeedbackEngineState()
        self._load_state()

    def _load_state(self) -> None:
        """Load state from disk."""
        if self.state_path.exists():
            try:
                self.state = FeedbackEngineState.model_validate_json(
                    self.state_path.read_text(encoding="utf-8")
                )
                logger.info(
                    "Loaded feedback engine: %d pending suggestions",
                    len(self.state.suggestions)
                )
            except Exception as e:
                logger.warning("Failed to load feedback engine state: %s", e)

    def _save_state(self) -> None:
        """Save state to disk."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            self.state.model_dump_json(indent=2),
            encoding="utf-8"
        )

    def analyze(
        self,
        component_memory: "ComponentMemory",
        outcome_tracker: "OutcomeTracker",
    ) -> list[EdgeSuggestion]:
        """Analyze usage patterns and generate edge suggestions.

        Args:
            component_memory: Component memory with co-selection data.
            outcome_tracker: Outcome tracker with success/failure data.

        Returns:
            List of new suggestions generated.
        """
        from skill_retriever.memory.component_memory import ComponentMemory
        from skill_retriever.memory.outcome_tracker import OutcomeTracker

        new_suggestions: list[EdgeSuggestion] = []

        # Get existing suggestion pairs to avoid duplicates
        existing_pairs = {
            (s.source_id, s.target_id, s.suggestion_type)
            for s in self.state.suggestions
        }

        # 1. Analyze co-selections for BUNDLES_WITH suggestions
        for entry in component_memory.co_selections.values():
            if entry.count >= self.config.min_co_selections:
                pair = (entry.component_a, entry.component_b, SuggestionType.BUNDLES_WITH)
                if pair not in existing_pairs:
                    confidence = min(entry.count / 10, 1.0)  # Cap at 1.0
                    if confidence >= self.config.min_confidence:
                        suggestion = EdgeSuggestion(
                            suggestion_type=SuggestionType.BUNDLES_WITH,
                            source_id=entry.component_a,
                            target_id=entry.component_b,
                            confidence=confidence,
                            evidence=f"Co-selected {entry.count} times",
                        )
                        new_suggestions.append(suggestion)
                        existing_pairs.add(pair)

        # 2. Analyze co-failures for CONFLICTS_WITH suggestions
        co_failures = outcome_tracker.get_co_failure_pairs()
        for comp_a, comp_b, count in co_failures:
            if count >= self.config.min_co_failures:
                pair = (comp_a, comp_b, SuggestionType.CONFLICTS_WITH)
                if pair not in existing_pairs:
                    confidence = min(count / 5, 1.0)
                    if confidence >= self.config.min_confidence:
                        suggestion = EdgeSuggestion(
                            suggestion_type=SuggestionType.CONFLICTS_WITH,
                            source_id=comp_a,
                            target_id=comp_b,
                            confidence=confidence,
                            evidence=f"Failed together {count} times",
                        )
                        new_suggestions.append(suggestion)
                        existing_pairs.add(pair)

        # 3. Analyze selection rates for STRENGTHEN/WEAKEN suggestions
        for cid, stats in component_memory.usage_stats.items():
            if stats.recommendation_count >= 10:  # Need enough data
                rate = stats.selection_count / stats.recommendation_count
                if rate < 0.1:  # Rarely selected when recommended
                    # This could indicate the component is less relevant
                    # For now, just log - future: suggest weakening edges to it
                    logger.debug(
                        "Low selection rate for %s: %.2f (%d/%d)",
                        cid, rate, stats.selection_count, stats.recommendation_count
                    )

        # Add new suggestions to state
        self.state.suggestions.extend(new_suggestions)

        # Trim to max suggestions
        if len(self.state.suggestions) > self.config.max_suggestions:
            # Keep highest confidence suggestions
            self.state.suggestions.sort(key=lambda s: s.confidence, reverse=True)
            self.state.suggestions = self.state.suggestions[:self.config.max_suggestions]

        self.state.last_analysis = datetime.now(tz=UTC)
        self._save_state()

        logger.info("Generated %d new suggestions", len(new_suggestions))
        return new_suggestions

    def get_pending_suggestions(self) -> list[EdgeSuggestion]:
        """Get suggestions awaiting review."""
        return [s for s in self.state.suggestions if not s.reviewed]

    def review_suggestion(
        self,
        source_id: str,
        target_id: str,
        suggestion_type: SuggestionType,
        accept: bool,
    ) -> bool:
        """Review a pending suggestion.

        Args:
            source_id: Source component ID.
            target_id: Target component ID.
            suggestion_type: Type of suggestion.
            accept: Whether to accept the suggestion.

        Returns:
            True if suggestion was found and reviewed.
        """
        for suggestion in self.state.suggestions:
            if (
                suggestion.source_id == source_id
                and suggestion.target_id == target_id
                and suggestion.suggestion_type == suggestion_type
                and not suggestion.reviewed
            ):
                suggestion.reviewed = True
                suggestion.accepted = accept
                if accept:
                    self.state.applied_suggestions.append(suggestion)
                self._save_state()
                logger.info(
                    "Reviewed suggestion: %s %s->%s: %s",
                    suggestion_type, source_id, target_id,
                    "accepted" if accept else "rejected"
                )
                return True
        return False

    def apply_accepted_suggestions(
        self,
        graph_store: "GraphStore",
    ) -> int:
        """Apply accepted suggestions to the graph.

        Args:
            graph_store: Graph store to modify.

        Returns:
            Number of edges added/modified.
        """
        from skill_retriever.entities.graph import EdgeType, GraphEdge
        from skill_retriever.memory.graph_store import GraphStore

        applied_count = 0

        for suggestion in self.state.suggestions:
            if not suggestion.reviewed or not suggestion.accepted:
                continue

            # Skip if already applied
            if suggestion in self.state.applied_suggestions:
                continue

            # Map suggestion type to edge type
            edge_type_map = {
                SuggestionType.BUNDLES_WITH: EdgeType.BUNDLES_WITH,
                SuggestionType.CONFLICTS_WITH: EdgeType.CONFLICTS_WITH,
            }

            edge_type = edge_type_map.get(suggestion.suggestion_type)
            if edge_type is None:
                continue  # STRENGTHEN/WEAKEN not yet implemented

            # Check if edge already exists
            existing_edges = graph_store.get_edges(suggestion.source_id)
            edge_exists = any(
                e.target_id == suggestion.target_id and e.edge_type == edge_type
                for e in existing_edges
            )

            if not edge_exists:
                edge = GraphEdge(
                    source_id=suggestion.source_id,
                    target_id=suggestion.target_id,
                    edge_type=edge_type,
                    weight=suggestion.confidence,
                    metadata={
                        "source": "feedback_engine",
                        "evidence": suggestion.evidence,
                        "created_at": suggestion.created_at.isoformat(),
                    },
                )
                graph_store.add_edge(edge)
                applied_count += 1
                logger.info(
                    "Applied edge: %s %s->%s (confidence: %.2f)",
                    edge_type, suggestion.source_id, suggestion.target_id,
                    suggestion.confidence
                )

            self.state.applied_suggestions.append(suggestion)

        # Remove applied suggestions from pending
        self.state.suggestions = [
            s for s in self.state.suggestions
            if s not in self.state.applied_suggestions
        ]

        self._save_state()
        return applied_count

    def get_status(self) -> dict:
        """Get feedback engine status."""
        return {
            "pending_suggestions": len(self.get_pending_suggestions()),
            "total_suggestions": len(self.state.suggestions),
            "applied_count": len(self.state.applied_suggestions),
            "last_analysis": (
                self.state.last_analysis.isoformat()
                if self.state.last_analysis
                else None
            ),
        }


# Type hints for imports
if __name__ != "__main__":
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from skill_retriever.memory.component_memory import ComponentMemory
        from skill_retriever.memory.graph_store import GraphStore
        from skill_retriever.memory.outcome_tracker import OutcomeTracker
