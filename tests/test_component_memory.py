"""Tests for ComponentMemory usage tracking and co-selection."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from skill_retriever.memory.component_memory import ComponentMemory


class TestRecordRecommendation:
    def test_record_recommendation(self) -> None:
        mem = ComponentMemory()
        mem.record_recommendation("comp-a")

        stats = mem.usage_stats["comp-a"]
        assert stats.recommendation_count == 1
        assert stats.last_recommended is not None
        assert stats.selection_count == 0

    def test_record_multiple_recommendations(self) -> None:
        mem = ComponentMemory()
        for _ in range(3):
            mem.record_recommendation("comp-a")

        assert mem.usage_stats["comp-a"].recommendation_count == 3


class TestRecordSelection:
    def test_record_selection(self) -> None:
        mem = ComponentMemory()
        mem.record_selection(["A", "B", "C"])

        for cid in ("A", "B", "C"):
            assert mem.usage_stats[cid].selection_count == 1

        assert len(mem.co_selections) == 3
        assert "A|B" in mem.co_selections
        assert "A|C" in mem.co_selections
        assert "B|C" in mem.co_selections

    def test_co_selection_ordering(self) -> None:
        mem = ComponentMemory()
        mem.record_selection(["B", "A"])

        assert "A|B" in mem.co_selections
        assert "B|A" not in mem.co_selections

    def test_selection_updates_last_selected(self) -> None:
        mem = ComponentMemory()
        before = datetime.now(tz=UTC)
        mem.record_selection(["X"])
        after = datetime.now(tz=UTC)

        last = mem.usage_stats["X"].last_selected
        assert last is not None
        assert before - timedelta(seconds=1) <= last <= after + timedelta(seconds=1)


class TestCoSelected:
    def test_get_co_selected(self) -> None:
        mem = ComponentMemory()
        mem.record_selection(["A", "B"])
        mem.record_selection(["A", "B"])
        mem.record_selection(["A", "C"])

        result = mem.get_co_selected("A")
        assert result[0] == ("B", 2)
        assert result[1] == ("C", 1)

    def test_get_co_selected_top_k(self) -> None:
        mem = ComponentMemory()
        partners = ["P1", "P2", "P3", "P4", "P5", "P6"]
        for p in partners:
            mem.record_selection(["X", p])

        result = mem.get_co_selected("X", top_k=3)
        assert len(result) == 3


class TestSelectionRate:
    def test_get_selection_rate(self) -> None:
        mem = ComponentMemory()
        for _ in range(10):
            mem.record_recommendation("comp-a")
        for _ in range(3):
            mem.record_selection(["comp-a"])

        assert mem.get_selection_rate("comp-a") == 0.3  # noqa: PLR2004

    def test_get_selection_rate_zero_recommendations(self) -> None:
        mem = ComponentMemory()
        assert mem.get_selection_rate("nonexistent") == 0.0


class TestPersistence:
    def test_save_and_load(self, tmp_path: object) -> None:
        from pathlib import Path

        path = str(Path(str(tmp_path)) / "memory.json")

        mem = ComponentMemory()
        mem.record_recommendation("A")
        mem.record_selection(["A", "B"])
        mem.save(path)

        loaded = ComponentMemory.load(path)
        assert loaded.usage_stats["A"].recommendation_count == 1
        assert loaded.usage_stats["A"].selection_count == 1
        assert loaded.usage_stats["B"].selection_count == 1
        assert "A|B" in loaded.co_selections
        assert loaded.co_selections["A|B"].count == 1

    def test_load_nonexistent_returns_empty(self) -> None:
        mem = ComponentMemory.load("/nonexistent/path/memory.json")
        assert len(mem.usage_stats) == 0
        assert len(mem.co_selections) == 0
