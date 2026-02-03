# Roadmap: Skill Retriever

## Overview

This roadmap delivers a graph-based MCP server that accepts natural language task descriptions and returns minimal, dependency-complete component sets from any Claude Code component repository. The build follows Iusztin virtual layers (entities, nodes, memory, workflows, mcp) with data flowing unidirectionally: ingestion produces graph data, memory stores it, retrieval nodes query it, the orchestrator coordinates queries, and the MCP server exposes everything to Claude Code. Seven phases move from project scaffolding through domain models, memory infrastructure, retrieval logic, orchestration, serving, and final validation.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Foundation** - Project scaffolding, dependencies, embedding model pinning, dev tooling
- [x] **Phase 2: Domain Models & Ingestion** - Pydantic entities and repository crawl/parse/extract pipeline
- [x] **Phase 3: Memory Layer** - Graph store, vector store, and component memory subsystems
- [x] **Phase 4: Retrieval Nodes** - PPR engine, vector search, flow pruning, and score fusion
- [x] **Phase 5: Retrieval Orchestrator** - Pipeline coordination, dependency resolution, conflict detection
- [ ] **Phase 6: MCP Server & Installation** - FastMCP tools, component installation, rationale generation
- [ ] **Phase 7: Integration & Validation** - End-to-end wiring, hyperparameter tuning, evaluation harness

## Phase Details

### Phase 1: Foundation
**Goal**: Developer can run tests and import the package with all dependencies resolved, embedding model pinned, and project structure established
**Depends on**: Nothing (first phase)
**Requirements**: (none directly -- enabling infrastructure for all subsequent phases)
**Success Criteria** (what must be TRUE):
  1. `uv run pytest` executes successfully with zero errors on an empty test suite
  2. `from skill_retriever import __version__` works from a Python REPL
  3. Ruff and pyright pass with zero warnings on the codebase
  4. Embedding model version is pinned in a config file and loadable at import time
**Plans**: 1 plan

Plans:
- [x] 01-01-PLAN.md -- Scaffold project, install dependencies, configure dev tools, pin embedding model, verify all success criteria

### Phase 2: Domain Models & Ingestion
**Goal**: System can crawl any component repository and produce structured, deduplicated entity data ready for graph and vector storage
**Depends on**: Phase 1
**Requirements**: INGS-01, INGS-02, INGS-03, INGS-04
**Success Criteria** (what must be TRUE):
  1. Running the ingestion pipeline against davila7/claude-code-templates produces ComponentMetadata objects for all 7 component types (agents, skills, commands, settings, MCPs, hooks, sandbox)
  2. Running ingestion against a flat-directory test repository (non-davila7 structure) produces valid entities without code changes
  3. Each ingested component has name, type, tags, description, and git health signals (last update, commit frequency) populated
  4. Two-phase entity resolution (fuzzy string + embedding similarity >0.85) prevents duplicate nodes for the same concept
  5. Any component's full definition (description, parameters, usage, dependencies) is retrievable by ID after ingestion
**Plans**: 3 plans

Plans:
- [x] 02-01-PLAN.md -- Pydantic entity models (ComponentType, ComponentMetadata, GraphNode, GraphEdge, EdgeType) + new deps
- [x] 02-02-PLAN.md -- Repository crawlers, extraction strategies (davila7/flat/generic), frontmatter parser, git signals
- [x] 02-03-PLAN.md -- Two-phase entity resolution pipeline (fuzzy + embedding dedup) [TDD]

### Phase 3: Memory Layer
**Goal**: Ingested component data persists in graph and vector stores with PPR and similarity search operational
**Depends on**: Phase 2
**Requirements**: GRPH-01
**Success Criteria** (what must be TRUE):
  1. Components and their relationships (DEPENDS_ON, ENHANCES, CONFLICTS_WITH) are stored as nodes and directed edges in the graph store
  2. Vector store returns semantically similar components when queried with a text description (top-5 cosine similarity)
  3. PPR computation from a seed node returns ranked nodes within 200ms for a 1300-node graph
  4. Component memory tracks recommendation/selection counts and co-selection patterns (DeepAgent-style)
  5. Graph store abstraction layer allows swapping NetworkX backend without changing calling code
**Plans**: 3 plans

Plans:
- [x] 03-01-PLAN.md -- Graph store (NetworkX wrapper, PPR via built-in pagerank, Protocol abstraction)
- [x] 03-02-PLAN.md -- Vector store (FAISS IndexFlatIP with ID mapping, cosine similarity search)
- [x] 03-03-PLAN.md -- Component memory (usage tracking, co-selection patterns, JSON persistence)

### Phase 4: Retrieval Nodes
**Goal**: Individual retrieval strategies (vector, graph, pattern) each return relevant components independently
**Depends on**: Phase 3
**Requirements**: RETR-01, RETR-02, RETR-03, RETR-04
**Success Criteria** (what must be TRUE):
  1. User can search components by natural language and receive semantically relevant results (vector search node)
  2. User can filter results by component type and receive only components of that type
  3. System returns ranked top-N results with relevance scores (default 5-10)
  4. PPR engine with adaptive alpha produces different traversal depths for specific vs broad queries
  5. Flow pruning reduces retrieved subgraph size by at least 40% compared to unpruned PPR output while retaining top-ranked components
**Plans**: 3 plans

Plans:
- [x] 04-01-PLAN.md -- Query planner and vector search node
- [x] 04-02-PLAN.md -- PPR engine with adaptive alpha and flow pruner
- [x] 04-03-PLAN.md -- Score fusion (RRF), reranker, and context assembler

### Phase 5: Retrieval Orchestrator
**Goal**: System coordinates all retrieval strategies into a single pipeline that returns complete, conflict-free component sets for any task description
**Depends on**: Phase 4
**Requirements**: GRPH-02, GRPH-03, GRPH-04
**Success Criteria** (what must be TRUE):
  1. Given a task description, system returns a complete component set with all transitive dependencies resolved (no missing pieces)
  2. System detects and surfaces conflicts between recommended components before returning results
  3. Simple queries (single component lookup) complete in under 500ms; complex queries (multi-hop dependency resolution) complete in under 1000ms
  4. Pipeline enforces a hard token budget (max 2000 tokens) per retrieval call to prevent context pollution
**Plans**: 2 plans

Plans:
- [x] 05-01-PLAN.md -- Pipeline coordinator with early exit, caching, and latency monitoring
- [x] 05-02-PLAN.md -- Dependency resolver and conflict detector

### Phase 6: MCP Server & Installation
**Goal**: Claude Code can call the system as an MCP server and install recommended components into .claude/ directory
**Depends on**: Phase 5
**Requirements**: INTG-01, INTG-02, INTG-03, INTG-04
**Success Criteria** (what must be TRUE):
  1. MCP server exposes max 5 tools (search_components, get_component_detail, install_components, check_dependencies, ingest_repo) and responds to tools/list correctly
  2. Calling search_components with a task description returns ranked recommendations with graph-path rationale explaining each selection
  3. Calling install_components places chosen components into the correct .claude/ subdirectory structure
  4. Each recommendation includes estimated context token cost, and the system optimizes for minimal footprint
  5. Total MCP tool schema definitions stay under 300 tokens
**Plans**: TBD

Plans:
- [ ] 06-01: FastMCP server with tool definitions and schema
- [ ] 06-02: Installation engine and rationale generator

### Phase 7: Integration & Validation
**Goal**: End-to-end system is tuned, validated against known-good component sets, and ready for daily use
**Depends on**: Phase 6
**Requirements**: (cross-cutting validation of all requirements)
**Success Criteria** (what must be TRUE):
  1. 30+ query-to-expected-component validation pairs pass with MRR above 0.7
  2. Hybrid retrieval (vector + graph) outperforms vector-only and graph-only baselines on the validation set
  3. PPR alpha grid search results are documented and the chosen values produce stable rankings
  4. MCP server starts in under 3 seconds and handles 10 sequential queries without degradation
**Plans**: TBD

Plans:
- [ ] 07-01: Validation harness and evaluation metrics
- [ ] 07-02: Hyperparameter tuning (PPR alpha, fusion weights, token budgets)
- [ ] 07-03: End-to-end smoke tests and performance benchmarks

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation | 1/1 | Complete | 2026-02-02 |
| 2. Domain Models & Ingestion | 3/3 | Complete | 2026-02-03 |
| 3. Memory Layer | 3/3 | Complete | 2026-02-03 |
| 4. Retrieval Nodes | 3/3 | Complete | 2026-02-03 |
| 5. Retrieval Orchestrator | 2/2 | Complete | 2026-02-03 |
| 6. MCP Server & Installation | 0/2 | Not started | - |
| 7. Integration & Validation | 0/3 | Not started | - |
