"""Outcome tracking for execution feedback loops (LRNG-05).

Tracks installation outcomes, usage patterns, and negative feedback
to improve future recommendations.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class OutcomeType(StrEnum):
    """Types of tracked outcomes."""

    INSTALL_SUCCESS = "install_success"
    INSTALL_FAILURE = "install_failure"
    USED_IN_SESSION = "used_in_session"
    REMOVED_BY_USER = "removed_by_user"
    DEPRECATED = "deprecated"


class ComponentOutcome(BaseModel):
    """A single outcome event for a component."""

    component_id: str
    outcome_type: OutcomeType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    context: str = ""  # Optional context (error message, session ID, etc.)


class OutcomeStats(BaseModel):
    """Aggregated outcome statistics for a component."""

    component_id: str
    install_successes: int = 0
    install_failures: int = 0
    usage_count: int = 0
    removal_count: int = 0
    first_seen: datetime | None = None
    last_success: datetime | None = None
    last_failure: datetime | None = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate (successes / total attempts)."""
        total = self.install_successes + self.install_failures
        if total == 0:
            return 0.0
        return self.install_successes / total

    @property
    def is_problematic(self) -> bool:
        """Check if component has consistent failures."""
        return self.install_failures >= 3 and self.success_rate < 0.5


class OutcomeTracker(BaseModel):
    """Tracks execution outcomes for feedback-driven improvement.

    Records:
    - Installation successes/failures
    - Component usage in sessions
    - User removals (negative feedback)
    - Deprecated components

    This data feeds back into:
    - Score fusion (success_rate as multiplier)
    - Conflict detection (repeated failures suggest conflicts)
    - Deprecation warnings
    """

    outcomes: dict[str, OutcomeStats] = Field(default_factory=dict)
    recent_events: list[ComponentOutcome] = Field(default_factory=list)
    max_recent_events: int = Field(default=1000)

    def record_outcome(
        self,
        component_id: str,
        outcome_type: OutcomeType,
        context: str = "",
    ) -> None:
        """Record an outcome event for a component.

        Args:
            component_id: The component that had an outcome.
            outcome_type: Type of outcome (success, failure, etc.).
            context: Optional context information.
        """
        now = datetime.now(tz=UTC)

        # Update aggregated stats
        stats = self.outcomes.get(component_id)
        if stats is None:
            stats = OutcomeStats(component_id=component_id, first_seen=now)
            self.outcomes[component_id] = stats

        if outcome_type == OutcomeType.INSTALL_SUCCESS:
            stats.install_successes += 1
            stats.last_success = now
        elif outcome_type == OutcomeType.INSTALL_FAILURE:
            stats.install_failures += 1
            stats.last_failure = now
        elif outcome_type == OutcomeType.USED_IN_SESSION:
            stats.usage_count += 1
        elif outcome_type == OutcomeType.REMOVED_BY_USER:
            stats.removal_count += 1

        # Record event
        event = ComponentOutcome(
            component_id=component_id,
            outcome_type=outcome_type,
            timestamp=now,
            context=context,
        )
        self.recent_events.append(event)

        # Trim old events
        if len(self.recent_events) > self.max_recent_events:
            self.recent_events = self.recent_events[-self.max_recent_events:]

        logger.debug(
            "Recorded outcome: %s -> %s (context: %s)",
            component_id, outcome_type, context[:50] if context else "none"
        )

    def get_success_rate(self, component_id: str) -> float:
        """Get success rate for a component.

        Args:
            component_id: Component to check.

        Returns:
            Success rate (0.0-1.0), or 0.0 if no data.
        """
        stats = self.outcomes.get(component_id)
        if stats is None:
            return 0.0
        return stats.success_rate

    def get_problematic_components(self) -> list[str]:
        """Get components with consistent failures.

        Returns:
            List of component IDs that have low success rates.
        """
        return [
            cid for cid, stats in self.outcomes.items()
            if stats.is_problematic
        ]

    def get_frequently_removed(self, min_removals: int = 2) -> list[str]:
        """Get components frequently removed by users (negative feedback).

        Args:
            min_removals: Minimum removal count to include.

        Returns:
            List of component IDs with high removal counts.
        """
        return [
            cid for cid, stats in self.outcomes.items()
            if stats.removal_count >= min_removals
        ]

    def get_co_failure_pairs(self) -> list[tuple[str, str, int]]:
        """Detect component pairs that frequently fail together.

        This suggests potential conflicts that aren't in the graph.

        Returns:
            List of (component_a, component_b, co_failure_count) tuples.
        """
        # Group failures by timestamp (within 60 seconds = same session)
        failure_sessions: list[set[str]] = []
        current_session: set[str] = set()
        last_time: datetime | None = None

        for event in sorted(self.recent_events, key=lambda e: e.timestamp):
            if event.outcome_type != OutcomeType.INSTALL_FAILURE:
                continue

            if last_time and (event.timestamp - last_time).total_seconds() > 60:
                if len(current_session) >= 2:
                    failure_sessions.append(current_session)
                current_session = set()

            current_session.add(event.component_id)
            last_time = event.timestamp

        if len(current_session) >= 2:
            failure_sessions.append(current_session)

        # Count co-occurrences
        pair_counts: dict[tuple[str, str], int] = {}
        for session in failure_sessions:
            items = sorted(session)
            for i, a in enumerate(items):
                for b in items[i+1:]:
                    pair = (a, b)
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1

        # Return pairs with 2+ co-failures
        return [
            (a, b, count) for (a, b), count in pair_counts.items()
            if count >= 2
        ]

    def save(self, path: str) -> None:
        """Persist tracker to a JSON file."""
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Saved outcome tracker: %d components", len(self.outcomes))

    @classmethod
    def load(cls, path: str) -> OutcomeTracker:
        """Load tracker from a JSON file, or return empty if file missing."""
        p = Path(path)
        if not p.exists():
            return OutcomeTracker()
        try:
            return cls.model_validate_json(p.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Failed to load outcome tracker: %s", e)
            return OutcomeTracker()
