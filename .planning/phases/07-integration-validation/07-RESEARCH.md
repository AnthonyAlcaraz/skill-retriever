# Phase 7: Integration & Validation - Research

**Researched:** 2026-02-03
**Domain:** Retrieval evaluation, hyperparameter tuning, performance benchmarking
**Confidence:** HIGH

## Summary

This phase validates the end-to-end skill-retriever system against known-good component sets, tunes hyperparameters (PPR alpha, RRF k, thresholds), and ensures performance SLAs are met. The validation harness uses a query-to-expected-component dataset with Mean Reciprocal Rank (MRR) as the primary metric, comparing hybrid retrieval against vector-only and graph-only baselines.

The research covers four domains: (1) MRR calculation and baseline comparison methodology, (2) hyperparameter tuning via grid search with cross-validation, (3) performance benchmarking with pytest-benchmark, and (4) MCP server startup/load testing. The standard stack includes ranx for ranking evaluation (fast, verified against TREC Eval), pytest-benchmark for latency measurement, and FastMCP's in-memory testing for MCP validation.

**Primary recommendation:** Build a validation harness that stores query-component pairs as JSON fixtures, evaluates MRR using ranx, and runs grid search over alpha values with the validation set as the evaluation criterion.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| ranx | >=0.3.5 | Ranking evaluation metrics (MRR, NDCG, MAP) | Blazing fast (Numba), TREC Eval verified, supports statistical tests |
| pytest-benchmark | >=5.2 | Performance/latency benchmarking | pytest integration, regression detection, min/max/mean stats |
| pytest-asyncio | >=0.24 | Async test support for MCP server | Already in dev deps, required for FastMCP testing |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | (existing) | Generate deterministic test embeddings | Validation fixture creation |
| json | (stdlib) | Store validation pairs as JSON fixtures | Query-component pair storage |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| ranx | pytrec_eval | ranx is pure Python, faster; pytrec_eval requires C bindings |
| ranx | rank-eval | rank-eval last updated 2021, less maintained |
| pytest-benchmark | timeit | pytest-benchmark integrates with test suite, provides regression history |

**Installation:**
```bash
uv add ranx --dev
uv add pytest-benchmark --dev
```

## Architecture Patterns

### Recommended Project Structure
```
tests/
├── conftest.py                    # Shared fixtures
├── validation/                    # Phase 7 validation tests
│   ├── __init__.py
│   ├── conftest.py                # Validation-specific fixtures
│   ├── fixtures/
│   │   ├── validation_pairs.json  # 30+ query-expected-component pairs
│   │   └── baseline_results.json  # Historical baseline results
│   ├── test_mrr_evaluation.py     # MRR against validation set
│   ├── test_baselines.py          # Vector-only vs graph-only vs hybrid
│   ├── test_alpha_tuning.py       # PPR alpha grid search
│   └── test_performance.py        # Startup time, query latency, load
└── ...existing tests...
```

### Pattern 1: Validation Pair Fixture Structure
**What:** JSON fixture mapping queries to expected component IDs with relevance grades
**When to use:** All validation tests that compute MRR
**Example:**
```json
{
  "pairs": [
    {
      "query_id": "q_01",
      "query": "I need to authenticate users with JWT tokens",
      "expected": {
        "skill-jwt": 5,
        "agent-auth": 4,
        "skill-refresh": 3
      },
      "category": "authentication"
    },
    {
      "query_id": "q_02",
      "query": "github repository analysis tool",
      "expected": {
        "cmd-z-repo": 5,
        "skill-git": 3
      },
      "category": "development"
    }
  ]
}
```

### Pattern 2: Ranx Evaluation Fixture
**What:** Convert validation pairs to ranx Qrels/Run format
**When to use:** Computing MRR and NDCG metrics
**Example:**
```python
from ranx import Qrels, Run, evaluate

def build_qrels_from_pairs(pairs: list[dict]) -> Qrels:
    """Convert validation pairs to ranx Qrels format."""
    qrels_dict = {}
    for pair in pairs:
        qrels_dict[pair["query_id"]] = pair["expected"]
    return Qrels(qrels_dict)

def build_run_from_results(query_id: str, results: list[RankedComponent]) -> dict:
    """Convert pipeline results to ranx Run format."""
    return {comp.component_id: comp.score for comp in results}

# Evaluate MRR
mrr = evaluate(qrels, run, "mrr")
assert mrr >= 0.7, f"MRR {mrr} below 0.7 threshold"
```

### Pattern 3: Baseline Comparison Pattern
**What:** Run same queries through vector-only, graph-only, and hybrid modes
**When to use:** Proving hybrid outperforms single-mode baselines
**Example:**
```python
def test_hybrid_outperforms_baselines(pipeline, validation_pairs):
    """Hybrid retrieval should beat both baselines on MRR."""
    qrels = build_qrels_from_pairs(validation_pairs)

    # Vector-only baseline
    vector_run = run_vector_only(pipeline, validation_pairs)
    mrr_vector = evaluate(qrels, vector_run, "mrr")

    # Graph-only baseline
    graph_run = run_graph_only(pipeline, validation_pairs)
    mrr_graph = evaluate(qrels, graph_run, "mrr")

    # Hybrid (full pipeline)
    hybrid_run = run_hybrid(pipeline, validation_pairs)
    mrr_hybrid = evaluate(qrels, hybrid_run, "mrr")

    assert mrr_hybrid > mrr_vector, "Hybrid should beat vector-only"
    assert mrr_hybrid > mrr_graph, "Hybrid should beat graph-only"
```

### Pattern 4: Grid Search for Alpha Tuning
**What:** Systematic search over alpha values with MRR as objective
**When to use:** Documenting optimal PPR alpha for different query types
**Example:**
```python
def grid_search_alpha(
    pipeline: RetrievalPipeline,
    validation_pairs: list[dict],
    alpha_values: list[float] = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95],
) -> dict[float, float]:
    """Grid search PPR alpha, return alpha -> MRR mapping."""
    qrels = build_qrels_from_pairs(validation_pairs)
    results = {}

    for alpha in alpha_values:
        run = run_with_alpha(pipeline, validation_pairs, alpha)
        mrr = evaluate(qrels, run, "mrr")
        results[alpha] = mrr

    return results
```

### Pattern 5: pytest-benchmark for Latency SLAs
**What:** Use benchmark fixture to measure and assert latency constraints
**When to use:** MCP startup time, query latency tests
**Example:**
```python
import pytest

def test_mcp_startup_under_3_seconds(benchmark):
    """MCP server should start in under 3 seconds."""
    async def start_server():
        from skill_retriever.mcp import get_pipeline
        await get_pipeline()

    result = benchmark.pedantic(
        lambda: asyncio.run(start_server()),
        rounds=5,
        warmup_rounds=1,
    )

    # Assert startup time SLA
    assert benchmark.stats["max"] < 3.0, "Startup exceeds 3s SLA"

def test_query_latency_under_500ms(benchmark, pipeline):
    """Single query should complete in under 500ms."""
    result = benchmark(pipeline.retrieve, "authentication agent")
    assert result.latency_ms < 500
```

### Pattern 6: Sequential Load Test
**What:** Run 10 sequential queries, verify no degradation
**When to use:** MCP server stability testing
**Example:**
```python
def test_sequential_queries_no_degradation(pipeline):
    """10 sequential queries should not degrade."""
    queries = ["auth", "JWT", "github", "linkedin", "email"] * 2
    latencies = []

    for query in queries:
        result = pipeline.retrieve(query)
        latencies.append(result.latency_ms)

    # First 5 vs last 5 should not show significant degradation
    first_half_avg = sum(latencies[:5]) / 5
    second_half_avg = sum(latencies[5:]) / 5

    # Allow 20% degradation tolerance
    assert second_half_avg < first_half_avg * 1.2, "Performance degraded"
```

### Anti-Patterns to Avoid
- **Random test embeddings without seed:** Use `np.random.default_rng(42)` for deterministic validation
- **Testing MRR with too few pairs:** Need 30+ pairs for statistical significance
- **Hardcoding alpha values in code:** Store tuned values in config after grid search
- **Mixing unit and validation tests:** Keep validation tests in separate directory

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| MRR/NDCG calculation | Custom metric functions | ranx library | Verified against TREC Eval, handles edge cases |
| Latency measurement | time.time() wrappers | pytest-benchmark | Handles warmup, calibration, statistical analysis |
| Async test fixtures | Manual event loop management | pytest-asyncio | Automatic async handling with `asyncio_mode = "auto"` |
| Statistical significance | Custom t-test | ranx statistical tests | Supports paired t-test, Fisher's randomization |

**Key insight:** Ranking evaluation has decades of research behind it. Libraries like ranx implement the nuanced details (tie handling, cutoff semantics, empty result handling) that are easy to get wrong in custom implementations.

## Common Pitfalls

### Pitfall 1: Relevance Grade Mismatch
**What goes wrong:** Validation pairs use 0-5 grades but MRR expects binary relevance
**Why it happens:** MRR traditionally uses binary (relevant/not relevant), but ranx supports graded
**How to avoid:** Use binary grades (0/1) for MRR, save graded for NDCG
**Warning signs:** MRR values much lower than expected despite good visual results

### Pitfall 2: Missing Components in Test Graph
**What goes wrong:** Validation expects component X, but it's not in test fixtures
**Why it happens:** Fixtures diverge from validation pairs over time
**How to avoid:** Generate validation pairs FROM indexed components, not separately
**Warning signs:** Many queries return zero results during validation

### Pitfall 3: Cold Start Bias in Benchmark
**What goes wrong:** First query includes embedding model load, skews average
**Why it happens:** FastEmbed lazy-loads model on first use
**How to avoid:** Use `benchmark.pedantic()` with warmup_rounds=1
**Warning signs:** First benchmark run 10-100x slower than subsequent

### Pitfall 4: Cache Contamination
**What goes wrong:** Second baseline test uses cached results from first
**Why it happens:** Pipeline LRU cache persists across tests
**How to avoid:** Call `pipeline.clear_cache()` in fixture teardown
**Warning signs:** Graph-only test mysteriously fast, returns hybrid results

### Pitfall 5: Alpha Overfitting
**What goes wrong:** Grid search finds alpha=0.93 works best, but it's noise
**Why it happens:** Small validation set, single decimal precision
**How to avoid:** Use coarse grid [0.5, 0.6, 0.7, 0.8, 0.9], report ranges not exact values
**Warning signs:** Optimal alpha changes dramatically with 5 new validation pairs

### Pitfall 6: Empty Query Handling
**What goes wrong:** PPR returns empty dict when no seeds found, breaks baseline comparison
**Why it happens:** Graph-only baseline can't handle queries with no entity matches
**How to avoid:** Handle empty results gracefully, return worst-case MRR (0.0) for empty
**Warning signs:** Division by zero errors, NaN MRR values

## Code Examples

Verified patterns from official sources:

### MRR Calculation with ranx
```python
# Source: https://github.com/AmenRa/ranx
from ranx import Qrels, Run, evaluate

# Ground truth: query_id -> {doc_id: relevance_grade}
qrels = Qrels({
    "q_1": {"skill-jwt": 1, "agent-auth": 1},
    "q_2": {"cmd-z-repo": 1},
})

# System output: query_id -> {doc_id: score}
run = Run({
    "q_1": {"skill-jwt": 0.95, "skill-refresh": 0.8, "agent-auth": 0.75},
    "q_2": {"cmd-z-repo": 0.9, "cmd-z-dd": 0.5},
})

# Compute MRR
mrr = evaluate(qrels, run, "mrr")
print(f"MRR: {mrr}")  # Should be close to 1.0 if top result is relevant
```

### pytest-benchmark Fixture Usage
```python
# Source: https://pytest-benchmark.readthedocs.io/en/latest/usage.html
import pytest

def test_pipeline_latency(benchmark, pipeline):
    """Benchmark pipeline retrieval latency."""
    # benchmark() automatically handles timing, warmup, rounds
    result = benchmark(pipeline.retrieve, "authentication agent")

    # Access stats after benchmark completes
    mean_ms = benchmark.stats["mean"] * 1000
    max_ms = benchmark.stats["max"] * 1000

    # Assert SLA
    assert max_ms < 1000, f"Max latency {max_ms}ms exceeds 1000ms SLA"
```

### FastMCP In-Memory Testing
```python
# Source: https://gofastmcp.com/patterns/testing
import pytest
from fastmcp.client import Client

@pytest.fixture
async def mcp_client():
    """In-memory MCP client for testing."""
    from skill_retriever.mcp.server import mcp

    async with Client(transport=mcp) as client:
        yield client

async def test_search_tool_exists(mcp_client):
    """Verify search_components tool is registered."""
    tools = await mcp_client.list_tools()
    tool_names = [t.name for t in tools]
    assert "search_components" in tool_names

async def test_search_returns_results(mcp_client):
    """Verify search returns recommendations."""
    result = await mcp_client.call_tool(
        name="search_components",
        arguments={"query": "authentication", "top_k": 5}
    )
    assert result.data is not None
```

### Baseline Comparison Test Structure
```python
# Source: Best practices from hybrid retrieval evaluation methodology
import pytest
from ranx import Qrels, Run, evaluate

@pytest.fixture
def validation_qrels(validation_pairs):
    """Convert validation pairs to ranx Qrels."""
    qrels_dict = {p["query_id"]: p["expected"] for p in validation_pairs}
    return Qrels(qrels_dict)

def test_hybrid_outperforms_vector_only(
    pipeline, validation_pairs, validation_qrels
):
    """Hybrid should beat vector-only baseline on MRR."""
    # Run vector-only (skip PPR stage)
    vector_run_dict = {}
    for pair in validation_pairs:
        # Bypass graph, use vector results only
        results = vector_search_only(pair["query"], pipeline)
        vector_run_dict[pair["query_id"]] = {
            r.component_id: r.score for r in results
        }

    vector_run = Run(vector_run_dict)
    mrr_vector = evaluate(validation_qrels, vector_run, "mrr")

    # Run hybrid (full pipeline)
    hybrid_run_dict = {}
    for pair in validation_pairs:
        result = pipeline.retrieve(pair["query"])
        hybrid_run_dict[pair["query_id"]] = {
            c.component_id: c.score for c in result.context.components
        }

    hybrid_run = Run(hybrid_run_dict)
    mrr_hybrid = evaluate(validation_qrels, hybrid_run, "mrr")

    print(f"Vector-only MRR: {mrr_vector:.3f}")
    print(f"Hybrid MRR: {mrr_hybrid:.3f}")

    assert mrr_hybrid > mrr_vector, (
        f"Hybrid ({mrr_hybrid:.3f}) should beat vector-only ({mrr_vector:.3f})"
    )
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Custom MRR functions | ranx/rank-eval libraries | 2022 | TREC Eval verified, 10-100x faster |
| Manual timeit loops | pytest-benchmark | 2024 | Automatic calibration, regression detection |
| Subprocess MCP testing | In-memory FastMCP client | 2025 | No process overhead, deterministic |
| Single alpha value | Adaptive alpha (query-dependent) | 2026 | Already implemented in 04-02 |

**Deprecated/outdated:**
- pytrec_eval: Requires C bindings, harder to install on Windows
- Manual time.time() benchmarking: No warmup, no statistical analysis

## Open Questions

Things that couldn't be fully resolved:

1. **Optimal validation set size**
   - What we know: 30+ pairs minimum for statistical significance
   - What's unclear: Exact number for this component domain (skills, agents, commands)
   - Recommendation: Start with 30-40 pairs, add more if MRR variance is high

2. **Ground truth generation method**
   - What we know: Can generate synthetically or manually curate
   - What's unclear: Best balance of effort vs quality
   - Recommendation: Manual curation of 30 pairs from real Claude Code usage patterns

3. **Statistical significance threshold**
   - What we know: ranx supports paired t-test, p<0.05 standard
   - What's unclear: If dataset is too small for significance tests
   - Recommendation: Report MRR values with confidence intervals, skip p-values if N<30

## Sources

### Primary (HIGH confidence)
- [ranx GitHub](https://github.com/AmenRa/ranx) - MRR/NDCG API, Qrels/Run data structures
- [pytest-benchmark docs](https://pytest-benchmark.readthedocs.io/) - Fixture usage, timing configuration
- [FastMCP testing docs](https://gofastmcp.com/patterns/testing) - In-memory client pattern

### Secondary (MEDIUM confidence)
- [Evidently AI MRR explanation](https://www.evidentlyai.com/ranking-metrics/mean-reciprocal-rank-mrr) - MRR calculation details
- [Weaviate evaluation metrics](https://weaviate.io/blog/retrieval-evaluation-metrics) - Precision/Recall/NDCG relationships
- [Hybrid RAG benchmarking](https://community.netapp.com/t5/Tech-ONTAP-Blogs/Hybrid-RAG-in-the-Real-World-Graphs-BM25-and-the-End-of-Black-Box-Retrieval/ba-p/464834) - Baseline comparison methodology

### Tertiary (LOW confidence)
- [PPR alpha tuning research](https://link.springer.com/chapter/10.1007/978-3-319-58068-5_11) - Tuning methodology (academic, may not apply directly)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - ranx, pytest-benchmark are well-documented, verified
- Architecture: HIGH - Patterns follow established IR evaluation practices
- Pitfalls: MEDIUM - Based on general retrieval evaluation experience
- Code examples: HIGH - Verified against official documentation

**Research date:** 2026-02-03
**Valid until:** 2026-03-03 (30 days - stable domain)

---

## Phase-Specific Implementation Notes

### Validation Pair Categories

Based on existing implementation, validation pairs should cover:

1. **Authentication domain** (3-5 pairs): JWT, OAuth, login, refresh tokens
2. **Development tools** (3-5 pairs): Git, GitHub, repo analysis, debugging
3. **Content creation** (3-5 pairs): LinkedIn, Medium, writing, posts
4. **Analysis** (3-5 pairs): Deep research, Z1 philosophy, insights
5. **Infrastructure** (3-5 pairs): MCP servers, settings, hooks
6. **Multi-component** (5-10 pairs): Queries expecting multiple related components
7. **Negative cases** (3-5 pairs): Queries that should return empty or specific type

### Hyperparameters to Tune

Already implemented with defaults in codebase:
- **PPR alpha**: 0.85 default, 0.9 specific, 0.6 broad (in ppr_engine.py)
- **RRF k**: 60 (in score_fusion.py)
- **Flow pruning**: max 8 endpoints, max 10 paths, 0.01 threshold
- **Token budget**: 2000 max
- **High confidence threshold**: 0.9 for early exit

Grid search should validate these defaults and document:
1. MRR across alpha range [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
2. RRF k sensitivity [30, 60, 100]
3. Flow pruning endpoint sensitivity [5, 8, 10, 15]

### Success Criteria Mapping

| Criterion | Test | Metric |
|-----------|------|--------|
| 30+ validation pairs pass with MRR > 0.7 | test_mrr_evaluation.py | MRR from ranx |
| Hybrid outperforms baselines | test_baselines.py | MRR comparison |
| PPR alpha grid search documented | test_alpha_tuning.py | Grid search results |
| MCP starts < 3s | test_performance.py | pytest-benchmark max time |
| 10 sequential queries no degradation | test_performance.py | Latency comparison |
