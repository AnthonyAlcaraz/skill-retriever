# Project Research Summary

**Project:** skill-retriever
**Domain:** Graph-based MCP server for AI agent component retrieval (1300+ components)
**Researched:** 2026-02-02
**Confidence:** HIGH

## Executive Summary

This project builds a graph-based retrieval system that solves the context pollution problem plaguing AI agent development. Claude Code users face three compounding failures: they inject entire component libraries into context (wasting tokens), receive irrelevant recommendations (because the system doesn't understand dependencies), and miss critical components (because flat search doesn't traverse relationships). Research confirms that over-retrieval actively degrades LLM performance by 23%+ when 10% irrelevant content pollutes context. The solution is a knowledge graph with hybrid retrieval (vector similarity + Personalized PageRank) that returns minimal, complete component sets.

The recommended approach combines NetworkX for graph operations, FAISS for vector search, and FastEmbed for lightweight embeddings. This stack avoids the KuzuDB archival risk (project discontinued October 2025) while staying embedded and zero-ops. The retrieval pipeline follows a proven 7-stage architecture from production systems: query planning, parallel retrieval (PPR + vector + patterns), score fusion with flow-based pruning, reranking, and context assembly. The key differentiator is task-to-component-set mapping where users describe a goal and receive a complete set of compatible components, not a pile of search results.

Critical risks center on over-retrieval noise (the inverse U-curve where more graph traversal degrades quality), entity resolution failures (same concept appearing as disconnected nodes), and PPR alpha mistuning (wrong exploration depth). All three are preventable with architectural decisions made upfront: hard token budgets per query, two-phase entity deduplication in ingestion, and adaptive alpha tuning per query category. The system must be fast (sub-500ms MCP responses) and lean (max 5 MCP tools to avoid catalog bloat), which rules out runtime LLM calls and forces offline graph construction.

## Key Findings

### Recommended Stack

The stack prioritizes embedded components with zero external dependencies, suitable for an MCP server that must start fast and run as a single process. KuzuDB was the original choice but was archived in October 2025, forcing a pivot to NetworkX + FAISS as the foundation with a migration path to FalkorDBLite if Cypher becomes necessary later.

**Core technologies:**
- **Python 3.13**: Production-stable with full bug-fix support through October 2026. Avoids 3.14's bleeding edge while maintaining modern features.
- **NetworkX + FAISS**: Pure Python graph operations with battle-tested vector search. No server process, no external database, complete control over PPR implementation. Proven at scale in production retrieval systems.
- **FastEmbed (>=0.7.4)**: ONNX-based embeddings with ~50MB footprint vs 2GB for sentence-transformers. Uses bge-small-en-v1.5 (384-dim) as default, perfect for encoding component descriptions without PyTorch dependencies.
- **scikit-network (>=0.33.0)**: Native CSR sparse matrix input for Personalized PageRank. Faster than NetworkX for repeated PPR calls. BSD-licensed, minimal dependencies (NumPy + SciPy), published in JMLR.
- **FastMCP (>=2.0,<3.0)**: Standard MCP framework with decorator-based API. V2 is production-ready; v3 is beta. Powers 70%+ of MCP servers. Auto-generates schemas from type hints.
- **uv (>=0.9.28)**: Rust-powered package manager that's 10-100x faster than pip. Native pyproject.toml support with lockfiles. Industry standard for Python projects in 2026.

**Critical decision: KuzuDB archived, use abstraction layer.**
KuzuDB v0.11.3 remains installable from PyPI and functional (MIT license) with vector search (HNSW), full-text search, and Cypher support. The project was archived October 2025 but the final release works for 1300 components. Recommendation is to pin kuzu==0.11.3 with a graph store interface (`memory/graph_store.py`) that enables swapping to LadybugDB (venture-backed fork) or FalkorDBLite when they reach stability. Starting with NetworkX + FAISS provides the same capabilities without betting on an archived project.

### Expected Features

Research reveals a clear split between table stakes (every retrieval system does this), differentiators (what sets this apart), and anti-features (common mistakes to avoid).

**Must have (table stakes):**
- **Semantic search over components**: Embedding-based retrieval of component descriptions. Without this, users fall back to grep.
- **Component metadata indexing**: 7 component types (commands, prompts, hooks, agents, tools, MCP servers, workflows) with structured filtering.
- **Dependency resolution**: Surface what component A requires. Basic "show me deps" is table stakes; transitive graph traversal is differentiator territory.
- **Relevance ranking**: Without ranking, users drown in 1300 results. Combine semantic similarity with graph centrality.
- **Context-aware limiting**: Return 5-10 components max by default. Research shows adding 10% irrelevant content reduces LLM accuracy 23%. This is the core value proposition.
- **MCP protocol compliance**: Must implement tools/list, tools/call, capability declarations per MCP spec.

**Should have (differentiators):**
- **Graph-based dependency traversal**: KG-based dependency resolution with Cypher queries for transitive closure. Models DEPENDS_ON, CONFLICTS_WITH, ENHANCES, REPLACES relationships.
- **Task-to-component-set mapping**: User describes a task, system returns a complete set of components needed. This is the killer feature, not "find me one component."
- **Hybrid retrieval (vector + graph)**: Vector for semantic matching, PPR for structural relationships, merged with RRF or weighted fusion. Pure vector misses dependencies; pure graph misses semantics.
- **Anti-context-pollution scoring**: Each component has a token cost estimate. System optimizes for minimal context footprint while maximizing task coverage.
- **Component compatibility validation**: Before returning a set, validate that components don't conflict via CONFLICTS_WITH edges.
- **Explainable recommendations**: Graph paths are inherently explainable. "I recommended X because it handles Y which your task requires and depends on Z which I also included."

**Defer (v2+):**
- **Component freshness signals** (low effort but low priority)
- **Usage pattern learning** (cold start problem, seed with static analysis)
- **Query rewriting / intent clarification** (optimize after seeing real query patterns)
- **Abstraction level awareness** (commands wrap prompts wrap hooks; requires deeper graph modeling)

**Anti-features (explicitly do NOT build):**
- **Full package manager (install/update/version)**: npm/pip took decades. The system's job is recommendation, not installation. Claude Code handles file ops.
- **Component execution runtime**: MCP servers should not execute component code. That's the agent's job.
- **User accounts / ratings / reviews**: This is an internal tool for one user's library, not a public marketplace.
- **LLM-in-the-loop for every query**: LLM calls add 1-3s latency. Use LLM for offline tasks (graph enrichment), use fast retrieval at query time.

### Architecture Approach

The architecture follows three processing phases: offline ingestion (build graph), graph construction (index + embed), and online retrieval (query-time). The system separates read-heavy retrieval (sub-second response requirement) from write-rarely ingestion (minutes-to-hours, runs offline).

**Major components:**

1. **Ingestion Pipeline (nodes/extraction.py, nodes/graph_builder.py)**: Crawls repositories, extracts metadata via strategy pattern (different extractors for MCP servers, CLI tools, Python packages, Node packages), creates entities, builds relationships. Runs offline, triggered by `ingest_repo` MCP tool or scheduled job. Implements two-phase entity resolution (fuzzy string matching + embedding similarity >0.85) to prevent graph fragmentation.

2. **Memory Layer (memory/)**: Three subsystems per GSD rules. (1) Graph Store: NetworkX wrapper with PPR implementation using scikit-network's CSR-native interface. (2) Vector Store: FAISS index management for component embeddings. (3) Component Memory: DeepAgent-style usage tracking (times_recommended, times_selected, success_rate, co_selected_with) that feeds back into ranking.

3. **Retrieval Orchestrator (workflows/retrieval.py)**: 7-stage pipeline coordinator. Stage 1: Query planning (complexity classification, entity extraction, strategy selection). Stages 2-4: Parallel retrieval (PPR from seed nodes, vector similarity search, pattern matching). Stage 5: Score fusion with temporal decay and popularity signals. Stage 6: Reranking via cross-encoder. Stage 7: Context assembly with dependency DAG and explanation. Implements early exit for simple queries (skip graph traversal, vector-only).

4. **PPR Engine (nodes/ppr_engine.py)**: Personalized PageRank with adaptive alpha per query category. Specific queries (e.g., "database migration command") use alpha=0.2-0.3 for tight locality. Broad queries (e.g., "security audit pipeline") use alpha=0.08-0.15 for deep traversal. Precomputes PPR vectors for common seed nodes at ingestion time to hit <500ms latency target.

5. **MCP Server (mcp/)**: FastMCP-based server exposing max 5 tools to avoid context catalog bloat. Core tools: `search_components` (primary), `get_component_detail`, `install_components`, `check_dependencies`, `ingest_repo`. Schema definitions capped at 300 tokens total.

**Key architectural patterns:**
- **Staged pipeline with early exit**: Query planner classifies complexity; simple queries bypass graph traversal entirely.
- **Graph construction as ETL, not runtime**: Graph built offline during ingestion. Retrieval never blocks on graph mutations.
- **Bipartite graph structure**: Components and capabilities form two node types. Components PROVIDE capabilities. Enables capability search ("I need auth") to find all components providing that capability.
- **Flow-based pruning (PathRAG-inspired)**: After PPR scores nodes, trace max-flow paths from query seeds to top-scored nodes, prune edges below flow threshold. Prevents over-retrieval noise.
- **Incremental graph updates (LightRAG-inspired)**: New repos add/update only their nodes. No full rebuild after initial construction.

### Critical Pitfalls

The research identified 11 pitfalls across three severity tiers. The critical tier can cause rewrites or architectural failures.

1. **Over-retrieval noise destroys answer quality (inverse U-curve)**: Graph traversal exponentially increases retrieved subgraph size with each hop. Research shows GraphRAG achieved only 36-54% context relevance vs vanilla RAG's 62% because graph expansion added noise. Prevention: Implement flow-based pruning from day one (Phase 03). Set hard token budget per retrieval call (max 2000 tokens). Measure precision@k alongside recall@k. Prioritize shorter, higher-confidence paths.

2. **Entity resolution failures fragment the knowledge graph**: Same component concept appears as multiple disconnected nodes due to naming variations. Prevention: Build two-phase entity resolution into ingestion (Phase 02): (1) exact/fuzzy string matching, (2) semantic embedding similarity with 0.85+ cosine threshold. Maintain canonical name registry. Run graph connectivity metrics after each ingestion batch.

3. **PPR alpha parameter mistuning produces useless rankings**: Alpha controls random walk exploration depth. Too high (>0.3): walks stay local, missing multi-hop deps. Too low (<0.1): walks diffuse across entire graph, diluting relevance. Prevention: Implement adaptive alpha per query category (Phase 03). Build validation set of 20-30 query-to-expected-component mappings before tuning. Use grid search over [0.05, 0.10, 0.15, 0.20, 0.30] with mean reciprocal rank (MRR) as metric.

4. **Hybrid retrieval weighting defaults silently degrade**: Vector cosine similarity (0.7-0.95 range) and PPR scores (multiple orders of magnitude) are on different scales. Naive linear combination produces wrong rankings. Research shows improper fusion parameter caused hybrid to underperform vector-only baseline (MRR 0.390 vs 0.410). Prevention: Normalize scores to [0,1] within each channel before fusion (Phase 03). Use reciprocal rank fusion (RRF) as default combiner since it's rank-based and avoids scale mismatch. Measure whether hybrid beats each individual channel on calibration set.

5. **MCP response latency kills conversational flow**: Graph traversal + vector search + score fusion compound to 2-5 second responses. Adoption failure when sub-second is expected. Prevention: Precompute PPR vectors for common seed nodes at ingestion time (Phase 03). Run vector search and graph traversal in parallel, not sequentially. Cache top-100 most-requested component sets. Set latency budget: 500ms target, 1000ms hard limit.

## Implications for Roadmap

Research suggests an 8-phase build following the Iusztin virtual layers pattern adapted for Python AI projects with MCP serving. The critical path is Foundation → Domain Models → Ingestion → Memory → Retrieval → Orchestration → MCP Server.

### Phase 1: Foundation & Project Setup
**Rationale:** Establish tooling, dependencies, and project structure before any domain work. Embedding model pinning and graph database abstraction decisions made here cascade through all phases.

**Delivers:** `pyproject.toml` with uv dependency management, `src/skill_retriever/` layout, `utils/` module, Ruff + pyright configuration, pytest harness, embedding model pinned in config.

**Stack elements:** uv, Python 3.13, Pydantic >=2.12.5, dev tools (Ruff, pyright, pytest)

**Avoids:** Pitfall 11 (embedding model mismatch) by pinning model version in configuration from day one. Pitfall 9 (NetworkX as default) by establishing native Cypher convention early (if KuzuDB used).

**Research flags:** Standard Python project setup, no additional research needed.

### Phase 2: Domain Models & Ingestion Pipeline
**Rationale:** Define Pydantic entity models first (zero dependencies, domain layer) before any infrastructure. Ingestion produces the data that memory stores, so it must come before retrieval. Entity resolution is critical here to prevent graph fragmentation.

**Delivers:** `entities/` with ComponentMetadata, CapabilityEntity, PatternEntity, GraphNode, GraphEdge models. `nodes/extraction.py` with strategy pattern extractors for MCP servers, Python packages, Node packages, CLI tools. `nodes/graph_builder.py` with two-phase entity resolution (fuzzy + embedding similarity >0.85). `nodes/relation_builder.py` for edge construction.

**Features addressed:** T2 (metadata indexing), T3 (dependency resolution)

**Avoids:** Pitfall 2 (entity resolution failures) with two-phase deduplication. Pitfall 5 (schema drift) with write/read schema separation. Pitfall 10 (single-repo testing) by building 3 synthetic test repositories (davila7 structure, flat directory, nested categories).

**Research flags:** Needs research during planning. Extracting metadata from diverse repository structures is domain-specific work. Build validation harness with known-good component sets.

### Phase 3: Memory Layer
**Rationale:** Memory is a DIRECTORY per GSD rules, never a single file. Three subsystems: graph store, vector store, component memory. This layer stores what ingestion produces and serves what retrieval consumes. Must exist before retrieval can work.

**Delivers:** `memory/graph_store.py` (NetworkX wrapper, PPR via scikit-network CSR interface, abstraction layer for future KuzuDB/FalkorDBLite migration), `memory/vector_store.py` (FAISS index management, embedding via FastEmbed), `memory/component_memory.py` (DeepAgent-style usage tracking).

**Stack elements:** NetworkX, FAISS, FastEmbed >=0.7.4, scikit-network >=0.33.0

**Avoids:** Pitfall 1 (over-retrieval noise) by building PPR with flow-based pruning support from the start. Pitfall 7 (MCP latency) by precomputing PPR vectors for common seed nodes.

**Research flags:** Standard patterns. NetworkX PPR is well-documented. FAISS setup is straightforward. No additional research needed unless hitting performance issues.

### Phase 4: Retrieval Nodes
**Rationale:** Individual retrieval strategies (PPR, vector, patterns) are independent and can be built in parallel. Each is a self-contained node (domain layer) that reads from memory layer but doesn't know about orchestration.

**Delivers:** `nodes/query_planner.py` (complexity classification, entity extraction, strategy selection), `nodes/ppr_engine.py` (adaptive alpha PPR), `nodes/vector_search.py` (FAISS lookup), `nodes/pattern_matcher.py` (structural pattern matching), `nodes/flow_pruner.py` (PathRAG-style pruning), `nodes/temporal_scorer.py` (freshness decay), `nodes/reranker.py` (cross-encoder), `nodes/context_assembler.py` (format with explanations).

**Features addressed:** T1 (semantic search), T6 (relevance ranking), T7 (result limiting), D1 (graph traversal), D5 (hybrid retrieval)

**Avoids:** Pitfall 3 (PPR alpha mistuning) with adaptive alpha per query category. Pitfall 4 (hybrid weighting defaults) with RRF fusion and per-channel normalization. Pitfall 1 (over-retrieval) with flow pruning in this phase.

**Research flags:** Needs research during planning. PPR alpha tuning requires validation set of 20-30 query-component pairs. Flow pruning algorithm must be ported from JavaScript (`cross-vault-context.js`) to Python. Hybrid fusion weighting needs calibration.

### Phase 5: Retrieval Orchestrator
**Rationale:** Coordinates all retrieval nodes into 7-stage pipeline. Manages caching, timeout, early exit logic. Application layer that sequences domain nodes.

**Delivers:** `workflows/retrieval.py` with pipeline coordinator, caching layer, latency monitoring, early exit for simple queries.

**Features addressed:** D2 (task-to-component-set mapping), D4 (anti-context-pollution scoring)

**Avoids:** Pitfall 7 (MCP latency) with parallel execution of PPR + vector + pattern stages. Pitfall 1 (over-retrieval) with hard token budget enforcement (2000 tokens max per retrieval).

**Research flags:** Standard orchestration patterns. No additional research needed.

### Phase 6: MCP Server
**Rationale:** Exposes retrieval orchestrator through MCP protocol. Serving layer that consumes workflows. Must come after retrieval works.

**Delivers:** `mcp/server.py` with FastMCP decorators, max 5 tools (search_components, get_component_detail, install_components, check_dependencies, ingest_repo), schema definitions capped at 300 tokens total.

**Features addressed:** T8 (MCP protocol compliance)

**Avoids:** Pitfall 8 (context budget exhaustion) by limiting to 5 tools with 300-token schema ceiling. Pitfall 7 (MCP latency) with sub-500ms target enforced in orchestrator.

**Research flags:** Standard MCP patterns. FastMCP documentation is sufficient.

### Phase 7: Integration & Validation
**Rationale:** Wire everything together, build evaluation harness, tune hyperparameters (PPR alpha, hybrid fusion weights, token budgets). Validate against known-good component sets.

**Delivers:** End-to-end testing with 30+ query-component validation pairs, PPR alpha grid search results, hybrid fusion calibration, latency benchmarks, precision/recall metrics.

**Avoids:** Pitfall 3 (PPR mistuning) and Pitfall 4 (hybrid weighting) by tuning against validation set in this phase.

**Research flags:** No additional research. This phase validates earlier research findings against reality.

### Phase 8: Polish & Documentation
**Rationale:** Refinements, edge cases, documentation, deployment guides.

**Delivers:** README, usage examples, ingestion guides for different repository structures, performance tuning guide.

**Research flags:** None.

### Phase Ordering Rationale

- **Foundation must come first**: Embedding model pinning and database abstraction decisions cascade through all later phases. Getting these wrong forces refactoring.
- **Domain models before infrastructure**: Pydantic entities (Phase 2) define the data contracts that memory layer (Phase 3) implements. Clean dependency direction.
- **Ingestion before memory before retrieval**: Data flow is unidirectional. Can't store what hasn't been ingested. Can't retrieve what hasn't been stored.
- **Memory is a directory (3 subsystems)**: Graph store, vector store, component memory are independent concerns. Building memory as single file forces refactoring later.
- **Retrieval nodes before orchestration**: Individual strategies (PPR, vector, patterns) are parallelizable. Build them independently, then wire in orchestrator.
- **MCP server last**: Serving layer depends on working retrieval. Don't expose tools until retrieval quality is validated.
- **Validation as separate phase**: Tuning PPR alpha and fusion weights requires working end-to-end pipeline. Can't validate pieces in isolation.

### Research Flags

**Phases needing deeper research during planning:**
- **Phase 2 (Ingestion):** Repository metadata extraction strategies are domain-specific. Different component types (MCP servers vs CLI tools vs agents) have different metadata patterns. Build synthetic test repos alongside extractors.
- **Phase 4 (Retrieval Nodes):** PPR alpha tuning, hybrid fusion weighting, and flow pruning thresholds require validation data. Build 20-30 query-to-expected-component pairs before tuning. Port flow pruning algorithm from `cross-vault-context.js`.

**Phases with standard patterns (skip research-phase):**
- **Phase 1 (Foundation):** Standard Python project setup with uv, pyproject.toml, pytest.
- **Phase 3 (Memory):** NetworkX, FAISS, and FastEmbed setup is well-documented.
- **Phase 5 (Orchestration):** Pipeline coordination is standard application logic.
- **Phase 6 (MCP Server):** FastMCP patterns are well-established.
- **Phase 7 (Integration):** Validation with known-good test sets.
- **Phase 8 (Polish):** Documentation and refinement.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All core libraries verified via PyPI/official docs. KuzuDB archival risk mitigated with abstraction layer. NetworkX + FAISS are battle-tested. FastEmbed proven in production. |
| Features | MEDIUM-HIGH | Table stakes features well-established (semantic search, metadata indexing, MCP compliance). Differentiators (task-to-set mapping, graph traversal, hybrid retrieval) validated by research papers but novel combination. |
| Architecture | HIGH | 7-stage retrieval pipeline proven in production (`cross-vault-context.js`). Iusztin virtual layers pattern is industry-standard for Python AI. Component boundaries follow clean architecture. |
| Pitfalls | MEDIUM-HIGH | Critical pitfalls (over-retrieval, entity resolution, PPR tuning, hybrid fusion) confirmed by multiple academic papers and production experience. Specific mitigation strategies extrapolated from general principles. |

**Overall confidence:** HIGH

The technology stack is verified and stable. The architecture follows proven patterns from production systems (z-commands retrieval orchestrator) adapted to Python AI project structure. Critical pitfalls are well-documented in research literature with clear prevention strategies. The main uncertainty is in hyperparameter tuning (PPR alpha, fusion weights) which requires validation data built during Phase 2.

### Gaps to Address

**KuzuDB vs NetworkX tradeoff:** Research establishes that KuzuDB was archived October 2025, forcing a choice between (1) NetworkX + FAISS (pure Python, no external deps, full control), (2) KuzuDB 0.11.3 pinned with migration abstraction, or (3) FalkorDBLite (too new, launched late 2025). Recommendation is NetworkX + FAISS for initial build with graph store abstraction layer. Validate during Phase 3 implementation. If Cypher queries become necessary, migration path to FalkorDBLite exists.

**PPR alpha tuning requires validation set:** Cannot tune adaptive alpha without known-good query-to-expected-component pairs. Must build 20-30 validation examples during Phase 2 (Ingestion) by manually mapping queries to correct component sets. This is unavoidable work before Phase 4 (Retrieval Nodes) can be tuned.

**Hybrid fusion calibration needs metrics:** Research identifies that improper weighting causes hybrid to underperform baselines. Must measure precision@k, recall@k, and MRR on calibration set during Phase 7 (Integration). If hybrid doesn't beat vector-only and graph-only individually, the weighting is wrong. This requires iterative tuning.

**Flow pruning algorithm port:** The JavaScript implementation in `cross-vault-context.js` works in production. Must port the algorithm to Python (`nodes/flow_pruner.py`) during Phase 4. Core idea: after PPR scores nodes, trace max-flow paths from query seeds to top-scored nodes, prune edges below flow threshold. Implementation straightforward but requires validation that Python port produces equivalent results.

**Latency budget enforcement:** Research establishes <500ms target, 1000ms hard limit for MCP responses. Must instrument latency monitoring in Phase 5 (Orchestrator) and validate during Phase 7 (Integration). If latency exceeds budget, precompute more PPR vectors or cache more aggressively.

## Sources

### Primary (HIGH confidence)
- **z-commands production codebase:** `cross-vault-context.js` (PPR + flow pruning), `tdk-aggregator.js` (entity extraction + graph construction), retrieval orchestrator (7-stage pipeline pattern)
- **KuzuDB PyPI:** v0.11.3, Oct 10, 2025 — archived status confirmed
- **The Register:** KuzuDB abandoned article, Oct 14, 2025
- **FastMCP documentation:** v2.0 stable, v3.0 beta — MCP framework patterns
- **FastEmbed PyPI:** v0.7.4, Dec 5, 2025 — ONNX embeddings
- **scikit-network docs:** v0.33.0 — PageRank with CSR input
- **HippoRAG 2:** arXiv 2502.14802 — PPR-based retrieval with knowledge graphs
- **PathRAG:** arXiv 2502.14902 — Flow-based pruning for graph RAG, inverse U-curve
- **GraphRAG-Bench:** arXiv 2506.06331 — Context relevance drops with graph expansion (36-54% vs 62%)
- **GraphRAG Survey:** arXiv 2501.00309 — Comprehensive survey of noise/retrieval tradeoffs
- **HybridRAG:** arXiv 2408.04948 — Fusion weighting challenges (MRR 0.390 vs 0.410)
- **MCP Tools Specification:** Official protocol, 2025-03-26

### Secondary (MEDIUM confidence)
- **Tool RAG:** Red Hat Emerging Technologies, Nov 26, 2025 — Hybrid retrieval patterns
- **DeepAgent:** WWW 2026 paper, GitHub RUC-NLPIR — Memory architecture, usage tracking
- **COLT:** arXiv 2405.16089v1 — Completeness-oriented tool retrieval
- **LightRAG:** EMNLP 2025 — Dual-level entity-relation graph retrieval, incremental updates
- **DepsRAG:** arXiv 2405.20455v3 — KG-based dependency management
- **Context Pollution Research:** Kurtis Kemple blog — 23% accuracy degradation measurement
- **LLM Agentic Failure Modes:** arXiv 2512.07497v1 — Failure mode taxonomy
- **FalkorDB blog:** FalkorDBLite embedded Python graph database announcement
- **FalkorDB GraphRAG-SDK:** Official GitHub — Knowledge graph MCP patterns
- **CodeRabbit:** Ballooning context in MCP era article
- **Claude Code Issue #17668:** MCP context isolation problem

### Tertiary (LOW confidence)
- **Tool and Agent Selection preprint:** Bipartite tool-agent retrieval, not peer-reviewed
- **LadybugDB GitHub:** KuzuDB fork, too early to evaluate stability
- **MCP Server Best Practices 2026:** CData blog — design patterns
- **BytesRack blog:** MCP latency concerns — speculative

---
*Research completed: 2026-02-02*
*Ready for roadmap: yes*
