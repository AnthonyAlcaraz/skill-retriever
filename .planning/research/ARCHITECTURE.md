# Architecture Patterns

**Domain:** Graph-based skill/component retrieval MCP server
**Researched:** 2026-02-02
**Overall confidence:** HIGH (core patterns), MEDIUM (graph DB choice due to KuzuDB deprecation)

## Recommended Architecture

### System Overview

A hybrid retrieval system that ingests component repositories (1300+ across 7 types), builds a knowledge graph, and serves component recommendations through an MCP interface. The architecture follows three processing phases: offline ingestion, graph construction, and online retrieval.

```
                                      ┌──────────────────┐
                                      │   MCP Clients     │
                                      │  (Claude Code)    │
                                      └────────┬─────────┘
                                               │
                                      ┌────────▼─────────┐
                                      │   MCP Server      │
                                      │  (Serving Layer)  │
                                      └────────┬─────────┘
                                               │
                              ┌────────────────▼────────────────┐
                              │     Retrieval Orchestrator       │
                              │  (7-stage pipeline coordinator)  │
                              └──┬──────┬──────┬──────┬────────┘
                                 │      │      │      │
                    ┌────────────▼┐ ┌───▼───┐ ┌▼─────┐│┌──────────┐
                    │Query Planner│ │  PPR  │ │Vector││ │ Pattern  │
                    │             │ │Engine │ │Search│││ │ Matcher  │
                    └─────────────┘ └───┬───┘ └──┬───┘│ └────┬─────┘
                                        │        │    │      │
                              ┌─────────▼────────▼────▼──────▼──┐
                              │         Knowledge Graph          │
                              │   (FalkorDB or NetworkX+FAISS)   │
                              └──────────────┬──────────────────┘
                                             │
                              ┌──────────────▼──────────────────┐
                              │        Ingestion Pipeline        │
                              │  (repo crawl → extract → index)  │
                              └─────────────────────────────────┘
```

### Component Boundaries

| Component | Responsibility | Communicates With | Iusztin Layer |
|-----------|---------------|-------------------|---------------|
| **Ingestion Pipeline** | Crawl repos, extract metadata, build graph | Knowledge Graph, Vector Store | nodes/ |
| **Knowledge Graph** | Store entities (components, skills, patterns) + relationships (depends-on, similar-to, co-occurs-with) | PPR Engine, Pattern Matcher, Vector Store | memory/ |
| **Vector Store** | Embed component descriptions + code signatures for semantic search | Retrieval Orchestrator | memory/ |
| **Query Planner** | Classify query complexity, select retrieval strategy | Orchestrator | nodes/ |
| **PPR Engine** | Run Personalized PageRank from seed nodes to find graph-proximal components | Knowledge Graph | nodes/ |
| **Pattern Matcher** | Match structural patterns (co-occurrence, dependency chains, type composition) | Knowledge Graph | nodes/ |
| **Temporal Scorer** | Weight results by freshness, popularity trends | Retrieval Orchestrator | nodes/ |
| **Reranker** | Combine and rerank multi-signal results | Retrieval Orchestrator | nodes/ |
| **Context Assembler** | Format final component recommendations with rationale | Retrieval Orchestrator | nodes/ |
| **Retrieval Orchestrator** | Coordinate the full pipeline, manage caching | All retrieval nodes | workflows/ |
| **MCP Server** | Expose tools: `search_components`, `recommend_set`, `explain_component` | Orchestrator | mcp/ |

### Data Flow

**Offline (Ingestion) Flow:**

```
1. Repo Discovery
   Repository URLs/paths → enumerate files → classify component type

2. Metadata Extraction
   Per component: parse README, package.json/pyproject.toml, entry files
   Extract: name, description, dependencies, exports, language, patterns used

3. Entity Creation
   Each component → node in graph (with type: skill, tool, agent, workflow, etc.)
   Each dependency → directed edge (depends-on)
   Each co-occurrence in same project → undirected edge (co-occurs-with)

4. Embedding Generation
   Component description + code signatures → vector embeddings
   Store in vector index alongside graph node IDs

5. Pattern Extraction
   Scan across repos for recurring component combinations
   Create "pattern" nodes linking frequently co-occurring components
```

**Online (Retrieval) Flow:**

```
Query: "I need authentication with OAuth and session management"
                    │
                    ▼
┌─── Stage 1: Query Planning ──────────────────────────────┐
│ Classify: complexity=moderate, intent=component_search    │
│ Extract entities: [authentication, OAuth, session]        │
│ Select strategy: hybrid (PPR + vector + pattern)          │
└──────────────────────┬───────────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
┌── Stage 2 ──┐ ┌─ Stage 3 ─┐ ┌─ Stage 4 ──────┐
│ PPR from     │ │ Vector     │ │ Pattern match   │
│ seed nodes   │ │ similarity │ │ (auth+OAuth     │
│ → graph-     │ │ search     │ │  known combo)   │
│ proximal     │ │ → top-K    │ │ → matched sets  │
│ components   │ │ semantic   │ │                 │
└──────┬───────┘ └─────┬──────┘ └───────┬────────┘
       │               │               │
       └───────────────┼───────────────┘
                       ▼
┌─── Stage 5: Score Fusion ────────────────────────────────┐
│ Combine PPR scores + vector distances + pattern matches   │
│ Apply temporal decay (prefer actively maintained)          │
│ Apply popularity signal (stars, downloads)                 │
└──────────────────────┬───────────────────────────────────┘
                       ▼
┌─── Stage 6: Reranking ──────────────────────────────────┐
│ Cross-encoder or LLM rerank for query-document relevance │
│ Dependency compatibility check (version conflicts)        │
└──────────────────────┬───────────────────────────────────┘
                       ▼
┌─── Stage 7: Context Assembly ────────────────────────────┐
│ Format top-N components with:                             │
│ - Why recommended (graph path, pattern, similarity)       │
│ - Dependency graph between recommended components         │
│ - Installation instructions                               │
│ - Known alternatives                                      │
└──────────────────────────────────────────────────────────┘
```

## Key Architecture Decisions

### Decision 1: Graph Database Selection

**Problem:** KuzuDB was archived in October 2025. The existing `social-graph.kuzu` in the z-commands ecosystem is a sunk cost.

**Options evaluated:**

| Option | Embedded? | Python? | Cypher? | Status |
|--------|-----------|---------|---------|--------|
| KuzuDB | Yes | Yes | Yes | ARCHIVED - do not use |
| FalkorDB | No (server) | Yes | Yes | Active, GraphRAG focus |
| FalkorDBLite | Yes (subprocess) | Yes | Yes | New, zero-config |
| NetworkX + FAISS | Yes (in-process) | Yes | No | Stable, no external deps |
| Neo4j | No (server) | Yes | Yes | Enterprise, heavy |
| LadybugDB | Yes | TBD | TBD | Fork of Kuzu, early stage |

**Recommendation:** Start with **NetworkX (graph) + FAISS (vectors)** as the foundation. This gives zero external dependencies, pure Python, and full control over the PPR implementation. Upgrade to FalkorDBLite if query complexity demands Cypher support later.

**Rationale:**
- The graph is read-heavy, write-rarely (rebuilt during ingestion, queried during retrieval)
- PPR on NetworkX is straightforward (scipy sparse matrix, 20 lines)
- FAISS handles vector similarity with proven performance
- No server process to manage for an MCP server that should be self-contained
- The existing z-commands orchestrator already proved this pattern works at scale with a JSON graph

**Confidence:** MEDIUM. FalkorDBLite is promising but too new (launched late 2025). NetworkX is boring and reliable.

### Decision 2: Bipartite Graph Structure (PwC Pattern)

The knowledge graph uses a bipartite-inspired structure where **components** and **capabilities** form two primary node types, with edges representing "provides" relationships. This draws from the PwC tool-agent retrieval paper where tools and agents share a unified vector space connected by ownership edges.

```
Component Nodes          Capability Nodes         Pattern Nodes
┌──────────┐            ┌───────────────┐        ┌─────────────┐
│ oauth-lib│──provides──│ authentication│        │ auth-stack  │
│          │──provides──│ token-mgmt    │        │ (oauth-lib +│
└──────────┘            └───────────────┘        │  session-mgr│
┌──────────┐            ┌───────────────┐        │  + user-db) │
│session-  │──provides──│ session-mgmt  │        └─────────────┘
│manager   │──provides──│ middleware    │
└──────────┘            └───────────────┘

Edges between components:
  depends-on (directed): oauth-lib → http-client
  co-occurs-with (undirected): oauth-lib ↔ session-manager
  alternative-to (undirected): oauth-lib ↔ passport-js
```

This structure enables three retrieval modes:
1. **Capability search:** "I need authentication" → find all components providing that capability
2. **Component expansion:** "I'm using oauth-lib" → find co-occurring components via graph traversal
3. **Pattern matching:** "Show me common auth stacks" → retrieve pattern nodes with their component sets

### Decision 3: Retrieval Pipeline (HippoRAG-Inspired)

The pipeline follows the proven 7-stage pattern from the existing z-commands retrieval orchestrator, adapted with insights from HippoRAG 2 and LightRAG.

**Stage adaptations from research:**

| Stage | Source Pattern | Adaptation for Skill Retriever |
|-------|---------------|-------------------------------|
| Query Planning | z-commands query-planner | Add intent classification: search vs. recommend vs. explain |
| PPR | HippoRAG 2 | Seed from extracted capability entities, not just text entities |
| Dual-Level Retrieval | LightRAG | Low-level: specific component lookup. High-level: "what components do I need for X?" |
| Flow Pruning | z-commands flow-pruner | Prune graph paths that cross component-type boundaries unnecessarily |
| Temporal Scoring | z-commands temporal-scorer | Weight by last-commit-date, not just last-seen |
| Reranking | LightRAG reranker (2025) | Cross-encoder on (query, component-description) pairs |
| Context Assembly | z-commands context-assembler | Output structured component sets with dependency DAG |

### Decision 4: DeepAgent-Style Tool Memory

Track component usage patterns across retrieval sessions:

```python
class ComponentMemory:
    component_id: str
    times_recommended: int
    times_selected: int        # User actually used recommendation
    success_rate: float        # Selected / Recommended
    co_selected_with: list[str]  # What other components were selected alongside
    last_recommended: datetime
```

This feedback loop improves recommendations over time. Components with high success rates get boosted. Co-selection patterns feed back into the graph as stronger co-occurrence edges.

### Decision 5: Colin-Style Freshness Tracking

Components have a freshness score based on:
- Last commit to source repo
- Last time component was recommended
- Dependency health (are its deps actively maintained?)
- Breaking changes detected (major version bumps)

Stale components get demoted in retrieval. Components with known security issues get flagged.

## Component Interaction Diagram

```
┌─────────────────────────────────────────────────────┐
│                    MCP Server                        │
│  Tools: search_components, recommend_set,            │
│         explain_component, ingest_repo               │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│              Retrieval Orchestrator                   │
│  Coordinates pipeline, manages cache, tracks usage   │
│                                                      │
│  ┌──────────┐ ┌─────┐ ┌────────┐ ┌───────────────┐ │
│  │  Query   │→│ PPR │→│ Vector │→│ Pattern Match │ │
│  │ Planner  │ │     │ │ Search │ │               │ │
│  └──────────┘ └──┬──┘ └───┬────┘ └──────┬────────┘ │
│                  │        │             │           │
│              ┌───▼────────▼─────────────▼───┐      │
│              │      Score Fusion +           │      │
│              │      Temporal Scoring         │      │
│              └──────────────┬────────────────┘      │
│                             │                       │
│              ┌──────────────▼────────────────┐      │
│              │   Reranker → Context Assembly │      │
│              └───────────────────────────────┘      │
└─────────────────────────────────────────────────────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
┌────────────────┐ ┌─────────┐ ┌──────────────┐
│ Knowledge Graph│ │ Vector  │ │  Component   │
│  (NetworkX)    │ │ Store   │ │   Memory     │
│                │ │ (FAISS) │ │ (usage stats)│
│ Nodes:         │ │         │ │              │
│  - component   │ │ Embeds: │ │ Tracks:      │
│  - capability  │ │  - desc │ │  - success   │
│  - pattern     │ │  - code │ │  - co-select │
│                │ │  - deps │ │  - freshness │
│ Edges:         │ │         │ │              │
│  - provides    │ └─────────┘ └──────────────┘
│  - depends-on  │
│  - co-occurs   │
│  - alternative │
└────────────────┘
          ▲
          │
┌─────────┴──────────────────────────────────────────┐
│              Ingestion Pipeline                      │
│                                                      │
│  ┌──────────┐ ┌──────────┐ ┌─────────┐ ┌────────┐ │
│  │   Repo   │→│ Metadata │→│  Entity │→│Relation│ │
│  │ Crawler  │ │ Extractor│ │ Creator │ │ Builder│ │
│  └──────────┘ └──────────┘ └─────────┘ └────────┘ │
│                                                      │
│  ┌──────────┐ ┌──────────┐                          │
│  │ Embedding│→│  Pattern │                          │
│  │Generator │ │ Detector │                          │
│  └──────────┘ └──────────┘                          │
└─────────────────────────────────────────────────────┘
```

## Patterns to Follow

### Pattern 1: Staged Pipeline with Early Exit

**What:** Each retrieval stage checks whether its output adds value. Simple queries (exact component name lookup) exit after vector search. Complex queries ("build me an auth system") run the full PPR + pattern matching pipeline.

**When:** Always. The query planner classifies intent and complexity upfront.

**Why:** The z-commands orchestrator demonstrated that 50%+ of queries are simple and skip PPR entirely. This saves significant latency.

```python
# In workflows/retrieval.py
async def retrieve(query: str, mode: str = "balanced") -> RetrievalResult:
    plan = query_planner.plan(query)

    if plan.complexity == "simple":
        # Direct vector lookup, skip graph traversal
        results = await vector_store.search(query, top_k=10)
        return context_assembler.assemble(results, mode)

    # Full pipeline for moderate/complex queries
    seeds = entity_extractor.extract(query)
    ppr_scores = ppr_engine.run(seeds)
    vector_results = await vector_store.search(query, top_k=30)
    pattern_matches = pattern_matcher.match(seeds)

    fused = score_fusion.combine(ppr_scores, vector_results, pattern_matches)
    reranked = reranker.rerank(query, fused)
    return context_assembler.assemble(reranked, mode)
```

### Pattern 2: Graph Construction as ETL, Not Runtime

**What:** The knowledge graph is rebuilt during ingestion (offline), not constructed at query time. Ingestion runs as a separate workflow triggered by `ingest_repo` MCP tool or scheduled job.

**When:** On repo addition, periodic refresh, or manual trigger.

**Why:** Graph construction requires LLM calls for entity extraction and is inherently slow. Retrieval must be fast (sub-second for MCP tool responses). Separating these concerns means the retrieval path never blocks on graph mutations.

### Pattern 3: Universal Repo Parser with Strategy Pattern

**What:** Different component types (MCP server, CLI tool, LangChain agent, etc.) need different extraction strategies. Use a registry of parsers that each know how to extract metadata from their component type.

**When:** During ingestion.

```python
# In nodes/extraction.py
class ExtractionStrategy(Protocol):
    def can_handle(self, repo_structure: dict) -> bool: ...
    def extract(self, repo_path: str) -> ComponentMetadata: ...

STRATEGIES = [
    MCPServerExtractor(),      # Detects MCP server pattern
    PythonPackageExtractor(),  # pyproject.toml / setup.py
    NodePackageExtractor(),    # package.json
    CLIToolExtractor(),        # Detects CLI patterns
    GenericExtractor(),        # Fallback: README + file structure
]
```

### Pattern 4: Incremental Graph Updates (LightRAG-inspired)

**What:** When a new repo is ingested, only add/update its nodes and edges. Do not rebuild the entire graph. Maintain a version counter per node.

**When:** After initial full build, all subsequent updates are incremental.

**Why:** With 1300+ components, full rebuilds take minutes. Incremental updates take seconds.

## Anti-Patterns to Avoid

### Anti-Pattern 1: Monolithic Graph Query

**What:** Sending the entire graph context to an LLM for "reasoning" about component relationships.
**Why bad:** Token explosion. A 1300-node graph serialized is 500K+ tokens. LLMs cannot reason over this.
**Instead:** Use PPR to select a relevant subgraph (30-50 nodes), then serialize only that subgraph for context assembly.

### Anti-Pattern 2: Vector-Only Retrieval

**What:** Using only embedding similarity to find components, ignoring graph structure.
**Why bad:** Misses transitive dependencies ("if you use A, you always need B") and compositional patterns ("auth systems typically combine X + Y + Z"). Vector search finds semantically similar descriptions but not structurally related components.
**Instead:** Hybrid retrieval where vector similarity is one signal among PPR scores and pattern matches.

### Anti-Pattern 3: Runtime LLM Calls in Retrieval Path

**What:** Calling an LLM during the retrieval pipeline (e.g., for entity extraction at query time).
**Why bad:** Adds 1-5 seconds of latency per retrieval. MCP tool calls should respond in <500ms.
**Instead:** Use heuristic entity extraction (regex + keyword matching + embedding lookup) at query time. Reserve LLM calls for the offline ingestion pipeline where latency tolerance is high.

### Anti-Pattern 4: Single Embedding Space for Everything

**What:** Embedding component names, descriptions, code, and READMEs all into one vector space.
**Why bad:** Code embeddings and natural language embeddings have different distributions. Mixing them degrades retrieval quality.
**Instead:** Use separate embedding indices for descriptions (natural language) and code signatures (code-aware embeddings). Fuse scores at retrieval time.

### Anti-Pattern 5: Over-Abstracting the Graph Schema

**What:** Creating elaborate ontologies with dozens of node and edge types before having data.
**Why bad:** Schema rigidity before understanding actual data distribution. Most components will cluster into 3-4 common patterns.
**Instead:** Start with minimal schema (component, capability, pattern nodes + 4 edge types). Extend when data demands it.

## Build Order (Dependencies Between Components)

The build order follows the Iusztin virtual layers pattern, adapted for this system's specific dependency chain.

```
Phase 01: Foundation
  └── pyproject.toml, src/ layout, utils/
  └── No dependencies on other phases

Phase 02: Domain Models (entities/)
  └── ComponentMetadata, CapabilityEntity, PatternEntity
  └── GraphNode, GraphEdge, RetrievalResult
  └── Pydantic models, zero external deps
  └── BLOCKS: everything else (all components consume these types)

Phase 03: Ingestion Pipeline (nodes/extraction.py, nodes/graph_builder.py)
  └── Repo crawler, metadata extractors, entity creator, relation builder
  └── DEPENDS ON: Phase 02 (entity types)
  └── BLOCKS: Phase 04 (graph must exist before retrieval works)

Phase 04: Memory Layer (memory/)
  └── memory/graph_store.py — NetworkX wrapper, PPR implementation
  └── memory/vector_store.py — FAISS index management
  └── memory/component_memory.py — Usage tracking (DeepAgent pattern)
  └── DEPENDS ON: Phase 02 (entity types), Phase 03 (data to store)
  └── BLOCKS: Phase 05 (retrieval needs memory layer)
  └── NOTE: 3 subsystems minimum (graph, vector, usage). Memory is a directory.

Phase 05: Retrieval Nodes (nodes/retrieval/)
  └── query_planner.py, ppr_engine.py, pattern_matcher.py
  └── temporal_scorer.py, reranker.py, context_assembler.py
  └── DEPENDS ON: Phase 04 (reads from graph + vector stores)
  └── BLOCKS: Phase 06 (orchestrator coordinates these nodes)

Phase 06: Retrieval Orchestrator (workflows/retrieval.py)
  └── 7-stage pipeline coordination
  └── Caching layer, timeout management
  └── DEPENDS ON: Phase 05 (all retrieval nodes)
  └── BLOCKS: Phase 07 (MCP server exposes orchestrator)

Phase 07: MCP Server (mcp/)
  └── Tool definitions: search_components, recommend_set, explain_component, ingest_repo
  └── DEPENDS ON: Phase 06 (orchestrator), Phase 03 (ingestion for ingest_repo tool)

Phase 08: Testing + Evaluation
  └── Built alongside each phase, not deferred
  └── Evaluation harness with known-good component sets
```

### Critical Path

```
Phase 02 → Phase 03 → Phase 04 → Phase 05 → Phase 06 → Phase 07
  (types)   (ingest)   (storage)  (retrieval)  (orchestrate)  (serve)
```

Phases 03 and 04 can partially overlap: the graph store interface (Phase 04) can be defined while extraction strategies (Phase 03) are still being built. But ingestion must produce data before retrieval can be tested.

### Parallelization Opportunities

- Phase 05 retrieval nodes (PPR, vector search, pattern matcher) are independent and can be built in parallel
- Phase 08 tests are built alongside each phase
- Embedding generation (Phase 03) and graph construction (Phase 04) can run in parallel during ingestion

## Scalability Considerations

| Concern | At 100 components | At 1,300 components | At 10,000 components |
|---------|-------------------|---------------------|----------------------|
| Graph size | NetworkX in-memory, trivial | NetworkX in-memory, <100MB | Consider FalkorDBLite |
| PPR latency | <10ms | <50ms | <200ms (sparse matrix) |
| Vector search | FAISS flat index | FAISS IVF index | FAISS HNSW |
| Ingestion time | Minutes | 10-30 min | 1-3 hours |
| Storage | <50MB total | <500MB total | <2GB total |

The system is designed for the 1,300 component target. NetworkX + FAISS handles this comfortably. If the corpus grows beyond 10K, migrate the graph store to FalkorDBLite without changing the retrieval logic (the graph store has an interface boundary).

## Sources

**HIGH confidence (existing codebase, verified):**
- z-commands retrieval orchestrator: `~/repos/z-commands/automation/linkedin-export/retrieval/orchestrator.js` - 7-stage pipeline pattern
- z-commands TDK aggregator: `~/repos/z-commands/automation/linkedin-export/tdk-aggregator.js` - Entity extraction + graph construction
- z-commands cross-vault-context: `~/repos/z-commands/automation/linkedin-export/cross-vault-context.js` - PPR + flow pruning integration

**HIGH confidence (peer-reviewed / official):**
- [HippoRAG 2 (arXiv 2502.14802)](https://arxiv.org/abs/2502.14802) - PPR-based retrieval with knowledge graphs, NeurIPS lineage
- [LightRAG (EMNLP 2025)](https://github.com/HKUDS/LightRAG) - Dual-level entity-relation graph retrieval
- [LEGO-GraphRAG (VLDB 2025)](https://www.vldb.org/pvldb/vol18/p3269-cao.pdf) - Modular graph RAG framework

**MEDIUM confidence (verified with official sources):**
- [KuzuDB archived October 2025](https://www.theregister.com/2025/10/14/kuzudb_abandoned/) - The Register, confirmed by GitHub archive status
- [FalkorDBLite](https://www.falkordb.com/blog/falkordblite-embedded-python-graph-database/) - Embedded Python graph, official blog
- [FalkorDB GraphRAG-SDK](https://github.com/FalkorDB/GraphRAG-SDK) - Official GitHub
- [Graphiti + FalkorDB MCP](https://www.falkordb.com/blog/mcp-knowledge-graph-graphiti-falkordb/) - Knowledge graph MCP pattern

**LOW confidence (single source, needs validation):**
- [Tool and Agent Selection preprint](https://www.preprints.org/frontend/manuscript/9402a980820b7b420ea80a1871a9c0d4/download_pub) - Bipartite tool-agent retrieval, not peer-reviewed
- [LadybugDB](https://github.com/ladybugdb) - KuzuDB fork, too early to evaluate stability
