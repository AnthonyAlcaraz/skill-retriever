# Feature Landscape: Graph-Based Skill/Component Retrieval System

**Domain:** MCP server for AI agent component retrieval (1300+ components, 7 types, KuzuDB knowledge graph, hybrid retrieval)
**Researched:** 2026-02-02
**Overall confidence:** MEDIUM-HIGH

---

## Table Stakes

Features users expect. Missing any of these and the system fails its core value proposition of "find me the right components for this task."

| # | Feature | Why Expected | Complexity | Depends On | Notes |
|---|---------|-------------|------------|------------|-------|
| T1 | **Semantic search over components** | Every retrieval system does this. Without it, users fall back to grep. npm search, VS Code marketplace, Smithery all provide keyword + semantic search as baseline. | Med | Embedding pipeline | Use dense embeddings (e.g., BGE-large-en-v1.5 as DeepAgent uses). Must handle component names, descriptions, and code snippets. |
| T2 | **Component metadata indexing** | Package managers index name, version, description, author, tags. VS Code marketplace indexes categories, ratings, install counts. Users expect structured filtering alongside free-text search. | Low | Schema design | 7 component types, tags, complexity level, author, date. Store as node properties in KuzuDB. |
| T3 | **Dependency resolution** | npm resolves transitive deps automatically. pip resolves (poorly) but resolves. If you recommend component A that requires component B, you must surface B. DepsRAG showed KG-based dep resolution works. | High | Graph schema | Core differentiator territory, but basic "show me what this needs" is table stakes. Full transitive resolution is differentiator. |
| T4 | **Component detail retrieval** | `npm info`, VS Code extension details page, MCP `tools/list` all return structured metadata on demand. Users need to inspect before selecting. | Low | T2 | Return full component definition: description, parameters, usage examples, dependencies, type. |
| T5 | **Filtering by component type** | VS Code uses `@category:`, npm uses keywords, MCP registries use categories. With 7 component types (commands, prompts, hooks, etc.), type filtering is mandatory. | Low | T2 | Map to: `@type:command`, `@type:prompt`, `@type:hook`, etc. |
| T6 | **Relevance ranking** | Every search system ranks results. npm by download count, VS Code by installs + rating, Google by PageRank. Without ranking, users drown in 1300 results. | Med | T1 | Combine semantic similarity with graph-based signals (centrality, usage frequency). |
| T7 | **Context-aware result limiting** | Tool RAG (Red Hat) showed that presenting too many tools degrades LLM performance by 23%+ (context pollution research). The system must return a bounded, optimal set, not a dump. | Med | T1, T6 | Default to 5-10 components max. Research shows adding 10% irrelevant content reduces LLM accuracy by 23%. |
| T8 | **MCP protocol compliance** | The system IS an MCP server. Must implement `tools/list`, `tools/call`, capability declaration per MCP spec. Non-negotiable for integration with Claude Code, Cursor, etc. | Med | None | Follow MCP SDK patterns. Declare capabilities including `listChanged`. |

---

## Differentiators

Features that set this apart from "just search a list." These solve the actual pain points: context pollution, missing dependencies, wrong abstraction level.

| # | Feature | Value Proposition | Complexity | Depends On | Notes |
|---|---------|-------------------|------------|------------|-------|
| D1 | **Graph-based dependency traversal** | npm resolves flat deps. This resolves *graph* deps -- components that work together, components that conflict, transitive chains. DepsRAG proved KG-based dep resolution with Cypher queries works. COLT showed graph neural networks capture "collaborative relationships" between tools that flat retrieval misses. | High | T3, Graph schema | KuzuDB Cypher queries for transitive closure. Model: `DEPENDS_ON`, `CONFLICTS_WITH`, `ENHANCES`, `REPLACES`. |
| D2 | **Task-to-component-set mapping** | User describes a task, system returns a *complete set* of components needed. COLT calls this "completeness-oriented retrieval" -- the gap between finding individual tools and finding the right *combination*. DeepAgent's "scene" concept groups collaborative tools. | High | D1, T1 | This is the killer feature. Not "find me a component" but "find me everything I need for X." Requires modeling component co-occurrence patterns. |
| D3 | **Abstraction level awareness** | A slash command wraps a prompt which uses hooks. If user asks for high-level functionality, return the command. If they ask for the building block, return the prompt. Package managers do not do this. IDE marketplaces partially do (extension packs vs individual extensions). | Med | Graph schema | Model containment hierarchy: command > prompt > hook > resource. Query at appropriate granularity. |
| D4 | **Anti-context-pollution scoring** | Each component returned has a "context cost" estimate (token count when injected into agent context). System optimizes for minimal context footprint while maximizing task coverage. Directly addresses the 23% accuracy degradation from irrelevant context. | Med | T7, D2 | Unique to this domain. No existing system optimizes for LLM context budget. Anthropic's context engineering research validates this as critical. |
| D5 | **Hybrid retrieval (vector + graph)** | Pure vector search misses structural relationships. Pure graph traversal misses semantic similarity. Combining them (as Tool RAG recommends with "dense and hybrid retrieval") gives both precision and recall. | High | T1, D1 | Vector for semantic matching, graph for structural relationships, merge and rerank. AgentCore Gateway does something similar with semantic tool selection + MCP. |
| D6 | **Component compatibility validation** | Before returning a set, validate that components do not conflict. Like npm's peer dependency warnings but for agent components. "These two hooks both modify the same lifecycle event" or "this command expects a prompt format that differs from what this prompt outputs." | Med | D1, Graph schema | Model `CONFLICTS_WITH` edges. Run validation pass before returning results. |
| D7 | **Usage pattern learning** | Track which component combinations are actually used together (from git history, from explicit feedback). VS Code does this with workspace-based recommendations. npm does it with download co-occurrence. Feed patterns back into ranking. | Med | D2, Storage | Cold start problem: seed with static analysis of existing repos. Over time, learn from actual usage. |
| D8 | **Query rewriting / intent clarification** | Tool RAG identifies query rewriting as a key enhancement. When user query is ambiguous ("I need auth"), the system can rewrite to multiple specific queries ("JWT generation" + "session management" + "OAuth flow") and merge results. DeepAgent does this as part of its reasoning loop. | Med | T1 | LLM-assisted rewriting. Can be a simple prompt or a dedicated reranking step. |
| D9 | **Explainable recommendations** | "I recommended component X because: it handles Y which your task requires, it depends on Z which I also included, and it has been used alongside W in 15 other configurations." KG-based recommendation systems excel at explainability because you can trace the graph path. | Med | D1, D2 | Major advantage over opaque vector similarity. Graph paths are inherently explainable. |
| D10 | **Component freshness and health signals** | Like npm's download trends, GitHub stars, last update date. Surface which components are actively maintained vs abandoned. VS Code marketplace shows ratings and update frequency. | Low | T2 | Pull from git metadata: last commit, commit frequency, open issues if tracked. |

---

## Anti-Features

Features to explicitly NOT build. These are common mistakes in this domain that would add complexity without proportional value, or actively harm the system.

| # | Anti-Feature | Why Avoid | What to Do Instead |
|---|-------------|-----------|-------------------|
| A1 | **Full package manager (install/update/version)** | npm/pip took decades to get right. Version resolution is an NP-hard problem. The system's job is *recommendation*, not *installation*. Claude Code already handles file operations. | Return component definitions and file paths. Let the consuming agent handle installation/copying. |
| A2 | **Component execution runtime** | MCP servers should not execute arbitrary component code. That is the agent's job. Mixing retrieval with execution creates security and reliability issues. AgentCore separates Gateway (discovery) from Runtime (execution) deliberately. | Return component specs. The agent decides when/how to execute. |
| A3 | **User accounts / ratings / reviews** | This is an internal tool for one user's component library, not a public marketplace. Building social features is enormous scope for zero value. VS Code marketplace needs this because it has millions of users. You have one. | Use git history and co-occurrence data as implicit quality signals. |
| A4 | **Real-time component monitoring / telemetry** | Observability platforms like Arize exist for agent runtime monitoring. Building telemetry into a retrieval server is scope creep. | Log queries and results for offline analysis. Feed back into D7 (usage pattern learning) asynchronously. |
| A5 | **Natural language component creation** | DeepAgent's Autonomous API & Tool Creation (AATC) generates tools on the fly. Fascinating research, but this system retrieves *existing* components, not generates new ones. Component creation belongs in the authoring workflow, not the retrieval system. | Surface gaps ("no component found for X") as actionable feedback for the user to create one. |
| A6 | **Multi-tenant isolation / RBAC** | Enterprise MCP registries (VS Code Private Marketplace, AgentCore) need this. A personal component retrieval system does not. Every hour spent on auth is an hour not spent on retrieval quality. | Single-user mode. If multi-tenant is ever needed, add it as a separate layer. |
| A7 | **Automatic component updating / syncing** | Auto-pulling upstream changes, managing versions across repos. This is a package manager feature (A1). The graph should be rebuilt on demand or on a schedule, not continuously synced. | Provide a `rebuild` / `ingest` command that scans repos and updates the graph. Run manually or via cron. |
| A8 | **LLM-in-the-loop for every query** | Tempting to use Claude for query understanding, reranking, and explanation on every call. But LLM calls add 1-3 seconds latency and cost. For a tool that Claude Code calls mid-task, speed matters. | Use LLM for offline tasks (graph enrichment, embedding generation). Use fast vector + graph retrieval at query time. Reserve LLM reranking for ambiguous queries only. |

---

## Feature Dependencies

```
T2 (metadata indexing) ─────────────────────────────────┐
   │                                                     │
   ├── T4 (component details)                            │
   ├── T5 (type filtering)                               │
   └── D10 (health signals)                              │
                                                         │
T1 (semantic search) ──────┬── T6 (relevance ranking) ──┤
   │                       │                             │
   │                       └── T7 (result limiting) ─────┤
   │                                                     │
   ├── D5 (hybrid retrieval) ───────────────┐            │
   │                                        │            │
   └── D8 (query rewriting)                 │            │
                                            │            │
T3 (dependency resolution) ─┬── D1 (graph traversal) ───┤
                            │       │                    │
                            │       ├── D2 (task-to-set) │
                            │       │       │            │
                            │       │       ├── D4 (anti-pollution scoring)
                            │       │       │
                            │       │       └── D7 (usage patterns)
                            │       │
                            │       ├── D6 (compatibility)
                            │       │
                            │       └── D9 (explainability)
                            │
                            └── D3 (abstraction levels)

T8 (MCP compliance) ── independent, parallel track
```

**Critical path:** T2 + T1 + T3 (foundation) --> D1 + D5 (graph + hybrid) --> D2 (task-to-set) --> D4 (anti-pollution).

Everything else branches off this spine.

---

## MVP Recommendation

**For MVP, prioritize the critical path that solves the #1 pain point (context pollution):**

1. **T8 - MCP compliance** -- without this, nothing integrates
2. **T2 - Metadata indexing** -- ingest the 1300+ components into KuzuDB
3. **T1 - Semantic search** -- basic "find components by description"
4. **T3 - Dependency resolution** -- model which components need which others
5. **T6 + T7 - Ranking + limiting** -- return top-N, not a dump
6. **D1 - Graph traversal** -- the first differentiator, transitive deps via Cypher
7. **D2 - Task-to-component-set** -- the killer feature

**Defer to post-MVP:**

- D3 (abstraction levels): Requires deeper graph modeling. Add after MVP validates the core retrieval loop.
- D4 (anti-pollution scoring): Needs token counting infrastructure. Add once D2 works.
- D7 (usage patterns): Cold start problem. Seed with static analysis, learn over time.
- D8 (query rewriting): Optimize after seeing real query patterns.
- D9 (explainability): Nice-to-have. Graph paths give you this mostly for free once D1 works.
- D10 (health signals): Low effort but low priority. Add when polishing.

---

## Competitive Landscape Summary

| System | What It Does Well | What It Lacks (Our Opportunity) |
|--------|------------------|-------------------------------|
| **MCP `tools/list`** | Standard protocol, universal client support | Flat list, no intelligence, no dependencies, no ranking |
| **npm/pip** | Mature dep resolution, massive registries | No semantic search, no task-oriented retrieval, no LLM context awareness |
| **VS Code Marketplace** | Categories, ratings, recommendations, extension packs | No graph relationships, no completeness-oriented retrieval |
| **Smithery / MCP registries** | 4000+ servers, semantic search, categories | Server-level not component-level, no dependency modeling |
| **Tool RAG (Red Hat)** | Hybrid retrieval, reranking, query rewriting | Research-stage, no graph structure, document-oriented not component-oriented |
| **DeepAgent** | Dense retrieval over 16K tools, memory architecture, ToolPO training | Requires fine-tuning, overkill for 1300 components, no graph deps |
| **COLT** | Graph contrastive learning for tool combinations, "scene" concept | Academic, requires training data, GNN complexity |
| **AgentCore Gateway** | Semantic tool selection, MCP-native, enterprise-grade | Cloud service, not local, no KG, tool-level not component-level |
| **DepsRAG** | KG-based dep resolution, Cypher queries, multi-agent | Package-ecosystem focused, not agent-component focused |

**Our unique position:** None of these systems combine graph-based dependency traversal with semantic retrieval specifically for AI agent components, optimized for LLM context budgets. DepsRAG comes closest in approach (KG + Cypher + RAG) but targets package ecosystems. COLT comes closest in vision (completeness-oriented tool sets) but requires GNN training. We take the best ideas from both and apply them to the specific domain of Claude Code component retrieval.

---

## Sources

- [MCP Tools Specification](https://modelcontextprotocol.io/specification/2025-03-26/server/tools) - HIGH confidence
- [Tool RAG: Red Hat Emerging Technologies](https://next.redhat.com/2025/11/26/tool-rag-the-next-breakthrough-in-scalable-ai-agents/) - MEDIUM confidence
- [DeepAgent (WWW 2026)](https://github.com/RUC-NLPIR/DeepAgent) - HIGH confidence
- [COLT: Completeness-Oriented Tool Retrieval](https://arxiv.org/html/2405.16089v1) - HIGH confidence
- [DepsRAG: KG-Based Dependency Management](https://arxiv.org/html/2405.20455v3) - HIGH confidence
- [Amazon Bedrock AgentCore Gateway](https://aws.amazon.com/blogs/machine-learning/introducing-amazon-bedrock-agentcore-gateway-transforming-enterprise-ai-agent-tool-development/) - HIGH confidence
- [Context Pollution Research - Anthropic](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) - HIGH confidence
- [Context Pollution Measurement](https://kurtiskemple.com/blog/measuring-context-pollution/) - MEDIUM confidence
- [LLM Agentic Failure Modes](https://arxiv.org/html/2512.07497v1) - HIGH confidence
- [VS Code Extension Marketplace](https://code.visualstudio.com/docs/editor/extension-marketplace) - HIGH confidence
- [Smithery MCP Registry](https://smithery.ai/) - MEDIUM confidence
- [Official MCP Registry](https://registry.modelcontextprotocol.io/) - HIGH confidence
- [npm vs pip Dependency Management](https://medium.com/@kabira_79251/npm-vs-pip-package-dependency-management-comparison-22a2b761a1db) - LOW confidence
