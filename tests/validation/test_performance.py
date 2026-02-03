"""Performance benchmarks for MCP server and pipeline."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture


class TestStartupPerformance:
    """MCP server startup time tests."""

    def test_pipeline_startup_under_3_seconds(
        self, benchmark: BenchmarkFixture
    ) -> None:
        """Pipeline initialization should complete in under 3 seconds."""

        async def init_pipeline() -> None:
            # Reset global state to force fresh initialization
            from skill_retriever.mcp import server

            server._pipeline = None
            server._graph_store = None
            server._vector_store = None
            server._metadata_store = None
            await server.get_pipeline()

        def run_init() -> None:
            asyncio.run(init_pipeline())

        # Run benchmark with warmup
        result = benchmark.pedantic(
            run_init,
            rounds=3,
            warmup_rounds=1,
        )

        # Assert startup SLA
        max_time = benchmark.stats["max"]
        assert max_time < 3.0, f"Startup time {max_time:.2f}s exceeds 3s SLA"


class TestQueryLatency:
    """Query latency and throughput tests."""

    def test_simple_query_under_500ms(
        self, seeded_pipeline, benchmark: BenchmarkFixture
    ) -> None:
        """Simple query should complete in under 500ms."""
        result = benchmark(seeded_pipeline.retrieve, "authentication agent", top_k=5)

        max_latency = benchmark.stats["max"] * 1000  # Convert to ms
        assert max_latency < 500, f"Query latency {max_latency:.0f}ms exceeds 500ms SLA"

    def test_complex_query_under_1000ms(
        self, seeded_pipeline, benchmark: BenchmarkFixture
    ) -> None:
        """Complex multi-hop query should complete in under 1000ms."""
        complex_query = (
            "I need JWT authentication with refresh tokens "
            "and OAuth integration for GitHub login"
        )
        result = benchmark(seeded_pipeline.retrieve, complex_query, top_k=10)

        max_latency = benchmark.stats["max"] * 1000
        assert max_latency < 1000, f"Complex query latency {max_latency:.0f}ms exceeds 1000ms"


class TestLoadStability:
    """Load testing and degradation detection."""

    def test_sequential_queries_no_degradation(self, seeded_pipeline) -> None:
        """10 sequential queries should not show performance degradation."""
        queries = [
            "JWT authentication",
            "GitHub repository analysis",
            "LinkedIn post writer",
            "OAuth login flow",
            "MCP server setup",
            "debugging agent",
            "code review tool",
            "email processing",
            "data analysis",
            "security audit",
        ]

        # Clear cache to ensure fresh queries
        seeded_pipeline.clear_cache()

        latencies: list[float] = []
        for query in queries:
            result = seeded_pipeline.retrieve(query, top_k=5)
            latencies.append(result.latency_ms)

        # First 5 vs last 5 comparison
        first_half_avg = sum(latencies[:5]) / 5
        second_half_avg = sum(latencies[5:]) / 5

        print(f"\nLatencies: {[f'{l:.1f}ms' for l in latencies]}")
        print(f"First 5 avg: {first_half_avg:.1f}ms")
        print(f"Last 5 avg: {second_half_avg:.1f}ms")

        # Allow 50% degradation tolerance (generous for cold start effects)
        degradation_ratio = second_half_avg / first_half_avg if first_half_avg > 0 else 1
        assert degradation_ratio < 1.5, (
            f"Performance degraded by {(degradation_ratio - 1) * 100:.0f}%"
        )

    def test_cached_queries_fast(self, seeded_pipeline) -> None:
        """Cached queries should be significantly faster than cold queries."""
        query = "authentication agent"

        # Cold query (first run)
        seeded_pipeline.clear_cache()
        cold_result = seeded_pipeline.retrieve(query, top_k=5)
        cold_latency = cold_result.latency_ms

        # Warm query (cached)
        warm_result = seeded_pipeline.retrieve(query, top_k=5)
        warm_latency = warm_result.latency_ms

        print(f"\nCold latency: {cold_latency:.1f}ms")
        print(f"Warm latency: {warm_latency:.1f}ms")

        assert warm_result.cache_hit, "Second query should be cache hit"
        assert warm_latency < cold_latency * 0.5, (
            f"Cached query not significantly faster: {warm_latency:.1f}ms vs {cold_latency:.1f}ms"
        )
