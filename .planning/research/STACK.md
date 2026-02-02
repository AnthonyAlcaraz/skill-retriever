# Technology Stack

**Project:** skill-retriever (Graph-Based MCP Server for Claude Code Component Retrieval)
**Researched:** 2026-02-02
**Overall Confidence:** HIGH (all core libraries verified via PyPI/official docs/web search)

---

## Critical Finding: KuzuDB Archived

KuzuDB was archived on October 10, 2025. The company (Kuzu Inc.) stopped active development. The final release is **v0.11.3** with bundled extensions. Two community forks exist: **LadybugDB** (venture-backed, active development) and **Bighorn** (Kineviz). Neither fork has reached a stable release yet.

**Decision: Use KuzuDB 0.11.3 with migration abstraction.**

Rationale:
1. v0.11.3 remains fully installable from PyPI and functional (MIT license, no server dependency).
2. It includes vector search (HNSW), full-text search, and Cypher support -- all features this project needs.
3. An embedded DB with no server process aligns with MCP server deployment (single process, zero ops).
4. The graph layer will sit behind an interface (`memory/graph_store.py`) so migration to LadybugDB or another fork requires changing one module.
5. For 1300 components, KuzuDB's scale is more than sufficient. This is not a 10M-node problem.

**Migration triggers:** If LadybugDB ships a stable Python release, or if a KuzuDB bug blocks development with no workaround, migrate then. Do not pre-optimize for a fork that has no stable release.

**Confidence:** HIGH -- verified KuzuDB 0.11.3 on PyPI (Oct 10, 2025), confirmed vector extension bundled, confirmed Python API still works.

---

## Recommended Stack

### Core Runtime

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| Python | 3.13 | Runtime | Production-stable, full bug-fix support through Oct 2026. 3.14 is too new (Oct 2025 release) for all dependency wheels to be guaranteed. 3.13 sits in the sweet spot: modern features, broad library compatibility. | HIGH |
| uv | >=0.9.28 | Package/project manager | Rust-powered, 10-100x faster than pip. Native pyproject.toml support with lockfile. Industry standard for new Python projects in 2026. Replaces pip, virtualenv, pip-tools in one binary. | HIGH |
| Pydantic | >=2.12.5 | Data validation, entity models | Rust-core validation. Standard for Python data models. Required by FastMCP anyway. Native Python 3.14 support. | HIGH |

### Graph Database

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| KuzuDB | 0.11.3 (pinned) | Knowledge graph storage | Embedded (no server), Cypher queries, HNSW vector index, full-text search. Single-process deployment matches MCP server model. See Critical Finding above for archival risk mitigation. | HIGH |

**Pin to exact version.** No new releases will come. Pinning prevents accidental dependency resolution failures.

### Vector / Embeddings

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| FastEmbed | >=0.7.4 | Embedding generation | ONNX Runtime (no PyTorch dependency), 384-dim default model (bge-small-en-v1.5), lightweight (~50MB vs ~2GB for sentence-transformers). Supports sparse embeddings (SPLADE) for hybrid search. Perfect for an MCP server that must start fast and stay lean. | HIGH |

**Why NOT sentence-transformers:** Pulls in PyTorch (~2GB), overkill for an MCP server embedding component descriptions. FastEmbed gives equivalent quality at 1/40th the install size. If fine-tuning is needed later, sentence-transformers can be added as an optional dev dependency.

**Default model:** `BAAI/bge-small-en-v1.5` (384 dimensions). Fast, accurate, well-benchmarked on MTEB. For code-specific retrieval, evaluate `jinaai/jina-embeddings-v2-small-en` as an alternative (also supported by FastEmbed).

### Retrieval Pipeline

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| scikit-network | >=0.33.0 | Personalized PageRank | Native CSR sparse matrix input (no graph-to-matrix conversion overhead). Dedicated `PageRank` class with personalization parameter. BSD license, minimal deps (NumPy + SciPy only). Published in JMLR. | HIGH |
| SciPy | >=1.14 | Sparse matrix operations | CSR arrays for adjacency matrices, sparse linear algebra for flow pruning. Already a transitive dependency of scikit-network. | HIGH |
| NumPy | >=2.0 | Numerical operations | Array operations for PPR vectors, score normalization. Transitive dependency. | HIGH |

**Why scikit-network over NetworkX for PPR:** scikit-network accepts CSR sparse matrices directly as `fit(adjacency_matrix, weights=personalization_vector)`. NetworkX requires converting to its internal graph format and back, which is slow for repeated PPR calls. scikit-network's Cython-compiled power iteration is also faster.

**Why NOT fast-pagerank:** Good library but unmaintained (last commit years ago), no pip package with recent updates. scikit-network is actively maintained, published in JMLR, and provides the same CSR-direct interface with more algorithms (diffusion, classification) if needed later.

**Flow pruning implementation:** Custom. No off-the-shelf library exists for PathRAG-style flow pruning. Implement as a standalone module (`nodes/flow_pruner.py`) using SciPy sparse operations. The cross-vault-context.js in z-commands already implements this pattern in JavaScript -- port the algorithm to Python. Core idea: after PPR scores nodes, trace max-flow paths from query seeds to top-scored nodes, prune edges below a flow threshold.

### MCP Server Framework

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| FastMCP | >=2.0,<3.0 | MCP server framework | Standard Python MCP framework. Powers ~70% of MCP servers across all languages. Decorator-based API (`@mcp.tool`, `@mcp.resource`). Auto-generates schemas from type hints. Production-ready v2; v3 is beta, not yet stable. | HIGH |

**Why pin to v2:** FastMCP v3 is in beta (3.0.0b1). Production systems should use v2 until v3 reaches stable. Pin `fastmcp>=2.0,<3.0` and upgrade when v3 ships stable (expected mid-2026).

**Why NOT the official MCP Python SDK directly:** FastMCP wraps the official SDK and adds the high-level decorator API, auth, proxying, and testing utilities. The official SDK v2 (Q1 2026) will incorporate learnings from FastMCP but is not yet released. FastMCP v2 is the pragmatic choice today.

### Testing

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| pytest | >=9.0.2 | Test framework | Industry standard. Native pyproject.toml config. Fixture-based test setup maps well to graph DB test scenarios (create/teardown test graphs). | HIGH |
| pytest-asyncio | >=0.24 | Async test support | MCP server tools are async. pytest-asyncio provides `@pytest.mark.asyncio` and async fixtures. | MEDIUM |
| pytest-cov | >=6.0 | Coverage reporting | Standard coverage plugin. | HIGH |

### Dev Tools

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| Ruff | >=0.14.14 | Linter + formatter | Replaces Flake8 + Black + isort. 10-100x faster (Rust). Single tool for all code quality. From Astral (same team as uv). | HIGH |
| pyright | >=1.1.408 | Type checking | 3-5x faster than mypy on large codebases. Strict mode catches more bugs. Standard in VS Code via Pylance. Default Python version already set to 3.14. | HIGH |

**Why pyright over mypy:** Pyright checks all code by default (mypy skips unannotated functions). Pyright is faster. Pyright has better IDE integration. For a new project with full type annotations from day one, pyright is the stronger choice.

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Graph DB | KuzuDB 0.11.3 | Neo4j | Server-based, heavyweight. Overkill for embedded MCP server with 1300 nodes. Licensing complexity (Community vs Enterprise). |
| Graph DB | KuzuDB 0.11.3 | FalkorDB | Server-based (Redis module). Good for multi-service arch, wrong for single-process embedded MCP server. |
| Graph DB | KuzuDB 0.11.3 | LadybugDB | KuzuDB fork, no stable Python release yet. Monitor and migrate when ready. |
| Graph DB | KuzuDB 0.11.3 | DuckDB + graph extension | DuckDB is relational-first. Graph queries require workarounds. No native Cypher. KuzuDB is purpose-built for property graphs. |
| Embeddings | FastEmbed | sentence-transformers | 2GB PyTorch dependency. Slower cold start. Unnecessary for encoding short component descriptions. |
| Embeddings | FastEmbed | OpenAI API | External API call adds latency, cost, and network dependency. MCP server should work offline. |
| PPR | scikit-network | NetworkX | Slower for repeated PPR (graph format conversion overhead). scikit-network operates directly on CSR matrices. |
| PPR | scikit-network | fast-pagerank | Unmaintained. scikit-network provides same CSR interface with active maintenance. |
| PPR | scikit-network | igraph | C-based, fast, but heavier install. PPR API less ergonomic than scikit-network's `fit(matrix, weights)`. |
| MCP Framework | FastMCP v2 | Official MCP SDK | Lower-level API, more boilerplate. FastMCP wraps it with decorator syntax and test utilities. |
| MCP Framework | FastMCP v2 | FastMCP v3 | Beta. Not production-ready. Upgrade path exists when v3 stabilizes. |
| Type Checker | pyright | mypy | Slower, skips unannotated code by default, weaker IDE integration. |
| Formatter | Ruff | Black + isort + Flake8 | Three tools vs one. Ruff is faster and replaces all three. |

---

## Architecture-Relevant Stack Decisions

### KuzuDB stores both graph structure AND vector embeddings

KuzuDB's vector extension supports HNSW indexes on node properties. This means component embeddings live inside the same database as the knowledge graph. No separate vector store needed. Query pattern:

```cypher
-- Vector similarity search within graph context
CALL vector_search(component_embeddings, $query_embedding, 20)
RETURN node.name, node.type, score
```

This eliminates the need for ChromaDB, Qdrant, or any external vector DB. One database, one file, one process.

### Hybrid retrieval pipeline (PPR + vector)

The retrieval pipeline runs in two parallel branches that merge:

```
Query
  |
  ├── Branch 1: Vector search (KuzuDB HNSW) → top-K similar components
  |
  ├── Branch 2: PPR on graph (scikit-network) → top-K graph-relevant components
  |
  └── Merge: Score fusion (reciprocal rank fusion or weighted linear) → final ranked list
       |
       └── Flow pruning: Trace paths between query-relevant nodes, prune weak connections
```

### Embedding model lives in-process

FastEmbed with ONNX Runtime runs in the same Python process as the MCP server. No external service call for embeddings. Cold start: ~2-3 seconds for model load, then ~5ms per embedding. Cache the model instance at server startup.

---

## Installation

```bash
# Initialize project
uv init skill-retriever
cd skill-retriever

# Core dependencies
uv add "kuzu==0.11.3"
uv add "fastmcp>=2.0,<3.0"
uv add "fastembed>=0.7.4"
uv add "scikit-network>=0.33.0"
uv add "pydantic>=2.12.5"

# Dev dependencies
uv add --dev "pytest>=9.0.2"
uv add --dev "pytest-asyncio>=0.24"
uv add --dev "pytest-cov>=6.0"
uv add --dev "ruff>=0.14.14"
uv add --dev "pyright>=1.1.408"
```

### pyproject.toml key sections

```toml
[project]
name = "skill-retriever"
version = "0.1.0"
requires-python = ">=3.13"

[tool.ruff]
target-version = "py313"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "SIM", "TCH", "RUF"]

[tool.pyright]
pythonVersion = "3.13"
typeCheckingMode = "strict"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

---

## Version Pinning Strategy

| Dependency | Pin Strategy | Rationale |
|------------|-------------|-----------|
| kuzu | `==0.11.3` | Archived project, no new versions. Exact pin prevents breakage. |
| fastmcp | `>=2.0,<3.0` | v3 is beta. Stay on v2 stable line. |
| fastembed | `>=0.7.4` | Active development, semver-compatible updates safe. |
| scikit-network | `>=0.33.0` | Stable API, minimal breaking changes. |
| pydantic | `>=2.12` | Pydantic v2 API is stable. |
| Python | `>=3.13` | Broad compatibility. Do not require 3.14 yet. |

---

## What This Stack Does NOT Include (By Design)

| Omission | Reason |
|----------|--------|
| LangChain / LlamaIndex | No LLM orchestration needed. This is a retrieval server, not a chat agent. |
| ChromaDB / Qdrant / Pinecone | KuzuDB handles vector search natively. No separate vector DB. |
| Redis / Celery | No background job processing. MCP server handles requests synchronously. |
| Docker | MCP servers run as local processes (stdio transport). Docker adds complexity with no benefit. |
| SQLAlchemy / ORM | KuzuDB uses Cypher queries directly. No SQL, no ORM. |
| LLM API (OpenAI, Anthropic) | Retrieval is deterministic (graph + vector math). No LLM calls needed for search. |
| FastAPI / Flask | MCP protocol handles transport. No REST API layer needed. |

---

## Sources

- [KuzuDB PyPI](https://pypi.org/project/kuzu/) -- v0.11.3, Oct 10, 2025
- [KuzuDB archived - The Register](https://www.theregister.com/2025/10/14/kuzudb_abandoned/)
- [LadybugDB GitHub](https://github.com/LadybugDB/ladybug) -- KuzuDB fork
- [FastMCP documentation](https://gofastmcp.com/) -- v2.0 stable, v3.0 beta
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) -- official SDK, v2 expected Q1 2026
- [FastEmbed PyPI](https://pypi.org/project/fastembed/) -- v0.7.4, Dec 5, 2025
- [scikit-network docs](https://scikit-network.readthedocs.io/en/latest/) -- v0.33.0, PageRank with CSR input
- [Pydantic docs](https://docs.pydantic.dev/latest/) -- v2.12.5
- [pytest PyPI](https://pypi.org/project/pytest/) -- v9.0.2
- [Ruff PyPI](https://pypi.org/project/ruff/) -- v0.14.14, Jan 22, 2026
- [pyright PyPI](https://pypi.org/project/pyright/) -- v1.1.408, Jan 8, 2026
- [uv PyPI](https://pypi.org/project/uv/) -- v0.9.28, Jan 29, 2026
- [Python version status](https://devguide.python.org/versions/) -- 3.13 in full support, 3.14 stable
- [KuzuDB vector extension docs](https://docs.kuzudb.com/extensions/vector/) -- HNSW index on node properties
