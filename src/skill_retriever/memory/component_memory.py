"""Component memory: usage tracking and co-selection patterns."""

from __future__ import annotations

import itertools
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field


class ComponentUsageStats(BaseModel):
    """Tracks recommendation and selection counts for a single component."""

    component_id: str
    recommendation_count: int = 0
    selection_count: int = 0
    last_recommended: datetime | None = None
    last_selected: datetime | None = None


class CoSelectionEntry(BaseModel):
    """Records how often two components are selected together.

    Invariant: component_a < component_b lexicographically.
    """

    component_a: str
    component_b: str
    count: int = 0


class ComponentMemory(BaseModel):
    """Persistent memory of component usage patterns.

    Tracks how often components are recommended vs selected and which
    components tend to be selected together (co-selection).
    """

    usage_stats: dict[str, ComponentUsageStats] = Field(default_factory=dict)
    co_selections: dict[str, CoSelectionEntry] = Field(default_factory=dict)

    def record_recommendation(self, component_id: str) -> None:
        """Record that a component was recommended to the user."""
        stats = self.usage_stats.get(component_id)
        if stats is None:
            stats = ComponentUsageStats(component_id=component_id)
            self.usage_stats[component_id] = stats
        stats.recommendation_count += 1
        stats.last_recommended = datetime.now(tz=UTC)

    def record_selection(self, component_ids: list[str]) -> None:
        """Record that the user selected these components.

        Updates individual selection counts and co-selection pairs.
        """
        now = datetime.now(tz=UTC)
        for cid in component_ids:
            stats = self.usage_stats.get(cid)
            if stats is None:
                stats = ComponentUsageStats(component_id=cid)
                self.usage_stats[cid] = stats
            stats.selection_count += 1
            stats.last_selected = now

        for a, b in itertools.combinations(sorted(component_ids), 2):
            key = f"{a}|{b}"
            entry = self.co_selections.get(key)
            if entry is None:
                entry = CoSelectionEntry(component_a=a, component_b=b)
                self.co_selections[key] = entry
            entry.count += 1

    def get_co_selected(
        self, component_id: str, top_k: int = 5
    ) -> list[tuple[str, int]]:
        """Return components most frequently co-selected with the given one.

        Returns a list of (other_component_id, count) sorted by count descending.
        """
        pairs: list[tuple[str, int]] = []
        for entry in self.co_selections.values():
            if entry.component_a == component_id:
                pairs.append((entry.component_b, entry.count))
            elif entry.component_b == component_id:
                pairs.append((entry.component_a, entry.count))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:top_k]

    def get_selection_rate(self, component_id: str) -> float:
        """Return selection_count / recommendation_count, or 0.0 if none."""
        stats = self.usage_stats.get(component_id)
        if stats is None or stats.recommendation_count == 0:
            return 0.0
        return stats.selection_count / stats.recommendation_count

    def save(self, path: str) -> None:
        """Persist memory to a JSON file."""
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> ComponentMemory:
        """Load memory from a JSON file, or return empty if file missing."""
        p = Path(path)
        if not p.exists():
            return ComponentMemory()
        return cls.model_validate_json(p.read_text(encoding="utf-8"))
