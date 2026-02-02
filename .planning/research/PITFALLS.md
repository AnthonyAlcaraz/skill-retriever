# Domain Pitfalls

**Domain:** Graph-based component retrieval MCP server (KuzuDB + PPR + vector search)
**Researched:** 2026-02-02
**Overall confidence:** MEDIUM-HIGH (cross-verified across academic papers, production systems, and community reports)

---

## Critical Pitfalls

Mistakes that cause rewrites or major architectural issues.

### Pitfall 1: Over-Retrieval Noise Destroys Answer Quality (The Inverse U-Curve)

**What goes wrong:** Retrieving too many graph paths/subgraphs floods the LLM context with redundant information, actively degrading response quality. Performance follows an inverse U-shaped curve: too few paths miss critical relationships, but too many introduce noise that overwhelms the signal. PathRAG research (2025) measured this directly: GraphRAG and LightRAG achieved 36-54% context relevance compared to vanilla RAG's 62% because graph traversal expanded scope at the cost of precision.

**Why it happens:** Graph traversal is greedy by nature. Each additional hop exponentially increases the retrieved subgraph size. Without pruning, every tangential relationship gets included. Developers tune for recall ("did I get all relevant components?") and neglect precision ("did I get ONLY relevant components?").

**Consequences:** Context pollution (the exact problem this project exists to solve), slower response times, higher token costs, and worse recommendations. The three compounding failures from PROJECT.md all trace back to this.

**Warning signs:**
- Recommendations include components that are semantically adjacent but functionally irrelevant to the query
- Token consumption per retrieval call keeps climbing as the graph grows
- Adding more components to the graph makes recommendations worse, not better
- Retrieval latency grows non-linearly with graph size

**Prevention:** Implement flow-based pruning from the start, not as an optimization pass. PathRAG's approach assigns reliability scores to each path and caps the number of paths returned. Set a hard token budget for retrieved context (e.g., max 2000 tokens per retrieval call). Measure precision@k alongside recall@k from day one. Implement the pruning algorithm with distance awareness to prioritize shorter, higher-confidence paths.

**Phase mapping:** Phase 03 (Retrieval Engine). Build pruning into the retrieval pipeline architecture, not bolted on later.

**Confidence:** HIGH (multiple papers confirm: PathRAG arXiv:2502.14902, GraphRAG-Bench arXiv:2506.06331, comprehensive survey arXiv:2501.00309)

---

### Pitfall 2: Entity Resolution Failures Fragment the Knowledge Graph

**What goes wrong:** The same component concept appears as multiple disconnected nodes because the ingestion pipeline fails to recognize variants. Example: a "code-reviewer" agent, a "code-review" skill, and a "review-code" command all reference the same functional capability but become three isolated subgraphs. The graph degrades into disconnected islands of facts instead of a traversable knowledge structure.

**Why it happens:** Component repositories use inconsistent naming conventions. Different authors name related components differently. Abbreviations, hyphenation, word order, and description phrasing all vary. Without entity resolution, each string variation becomes a new node.

**Consequences:** Graph traversal fails to find relationships that exist in reality. PPR random walks terminate at dead ends instead of discovering connected components. Dependency resolution misses cross-type relationships (the agent that actually needs that MCP). The 1300+ components from davila7 alone will have dozens of near-duplicates.

**Warning signs:**
- Graph has many small disconnected subgraphs instead of a few large connected components
- The same functional capability appears under multiple node names
- PPR scores cluster near zero for nodes that should be highly relevant
- Manual inspection reveals obvious relationships the graph missed

**Prevention:** Build entity resolution into the ingestion pipeline as a mandatory step, not a post-processing optimization. Use a two-phase approach: (1) exact/fuzzy string matching for names, (2) semantic embedding similarity for descriptions with a threshold (0.85+ cosine similarity). Maintain a canonical name registry that maps variants to a single node. Run graph connectivity metrics after each ingestion batch: if connected component count is high relative to node count, entity resolution is failing.

**Phase mapping:** Phase 02 (Ingestion). Entity resolution is a core ingestion concern, not a graph concern.

**Confidence:** HIGH (KG construction surveys confirm: MDPI 2024, Cherre 2021, MLCPD arXiv:2510.16357)

---

### Pitfall 3: PPR Alpha Parameter Mistuning Produces Useless Rankings

**What goes wrong:** Personalized PageRank's teleport probability (alpha) controls the random walk's exploration depth. Too high (alpha > 0.3): walks stay local, missing multi-hop dependencies. Too low (alpha < 0.1): walks diffuse across the entire graph, diluting relevance scores. The result is either too-narrow recommendations that miss dependencies, or too-broad recommendations that include everything.

**Why it happens:** Alpha tuning requires task-specific validation data that doesn't exist at build time. Developers pick a default (usually 0.15 from the original PageRank paper) and never revisit. Component retrieval queries vary in specificity (some need 1-hop neighbors, some need 3-hop dependency chains), so no single alpha works for all queries.

**Consequences:** All recommendations converge to the same "popular" components regardless of query. Dependency chains beyond 2 hops never surface. PPR becomes equivalent to a simple popularity sort, wasting the graph structure entirely.

**Warning signs:**
- Top-k recommendations barely change across different queries
- Components with high connectivity always rank high, regardless of relevance
- Increasing k just adds noise without improving recall of actual dependencies
- PPR scores follow a power law with a sharp cliff after the top 5

**Prevention:** Implement adaptive alpha per query category. Short, specific queries (e.g., "database migration command") need higher alpha (0.2-0.3) for tight locality. Broad capability queries (e.g., "set up a full security audit pipeline") need lower alpha (0.08-0.15) for deeper traversal. Build a validation set of 20-30 query-to-expected-component mappings before tuning. Use grid search over alpha in [0.05, 0.10, 0.15, 0.20, 0.30] with mean reciprocal rank (MRR) as the metric.

**Phase mapping:** Phase 03 (Retrieval Engine) for implementation, Phase 07 (Evaluation) for tuning.

**Confidence:** MEDIUM (PPR surveys confirm the sensitivity: arXiv:2403.05198. Specific tuning for component retrieval is novel territory with no direct precedent.)

---

### Pitfall 4: Hybrid Retrieval Weighting Defaults That Silently Degrade

**What goes wrong:** The balance between vector search scores and graph traversal scores uses a fixed weighting that works for one query type but fails for others. Research shows an improperly tuned fusion parameter caused hybrid retrieval to underperform a vector-only baseline (MRR 0.390 vs 0.410). The system appears to work but delivers worse results than simpler approaches.

**Why it happens:** Vector similarity and PPR scores are on different scales and distributions. Naive linear combination (score = w1*vector + w2*ppr) assumes both scores are comparable, but they are not. Vector cosine similarity clusters between 0.7-0.95 for relevant results; PPR scores span several orders of magnitude. A component ranking high on PPR (structurally important in the graph) may be irrelevant to the query semantically, and vice versa.

**Consequences:** The system picks components that are either structurally central but semantically wrong (graph-dominated) or semantically similar but missing key dependencies (vector-dominated). Users lose trust in recommendations because "obvious" components get ranked below obscure ones.

**Warning signs:**
- Hybrid retrieval performs worse than either vector-only or graph-only on your validation set
- Score distributions from vector and graph channels have wildly different ranges
- One retrieval channel dominates 80%+ of final rankings regardless of query
- Removing the graph component doesn't noticeably change results

**Prevention:** Normalize scores to [0, 1] within each channel before fusion. Use reciprocal rank fusion (RRF) instead of score addition as the default combiner, since RRF is rank-based and avoids scale mismatch. Build a calibration set of queries where you know the correct answer and measure whether hybrid beats each individual channel. If it doesn't, the weighting is wrong. Start with equal weights and adjust based on query type: structural queries (dependency resolution) favor graph; semantic queries (capability matching) favor vector.

**Phase mapping:** Phase 03 (Retrieval Engine). This must be designed into the fusion architecture, not tuned ad-hoc.

**Confidence:** HIGH (HybridRAG arXiv:2408.04948, Superlinked benchmarks, AIMultiple 2026 survey all confirm the weighting sensitivity)

---

## Moderate Pitfalls

Mistakes that cause delays or technical debt.

### Pitfall 5: Schema Drift When Ingesting Multiple Repository Formats

**What goes wrong:** The universal ingestion pipeline assumes a stable structure, but component repositories evolve independently. A repository that used flat directories restructures into nested categories. New component types appear (e.g., "profiles" or "personas" alongside the existing 7 types). Metadata fields get renamed or removed. The ingestion pipeline silently produces partial or malformed graph nodes.

**Why it happens:** "Universal" ingestion is built against the current davila7 structure and tested only against it. Other repositories use different conventions. Even davila7 will evolve over time. Without schema validation on ingestion, drift goes undetected until downstream retrieval fails.

**Warning signs:**
- Ingestion logs show increasing "unknown field" or "missing field" warnings
- Component count after re-ingestion differs from previous runs without explanation
- Some components have empty descriptions or missing type classifications
- Graph query results include nodes with null/placeholder properties

**Prevention:** Separate write schema (flexible, accepts unknown fields into a metadata bag) from read schema (strict, validates required properties). Run post-ingestion validation that checks: every node has a name, type, and description; every edge connects valid nodes; no orphan nodes exist. Implement a dead-letter queue for components that fail validation instead of silently dropping them. Version the ingestion schema and log which version produced each node.

**Phase mapping:** Phase 02 (Ingestion). Schema validation is a core ingestion concern.

**Confidence:** MEDIUM (data engineering patterns from Estuary, DZone, Matia confirm the general problem. Component-repository-specific drift is speculative but highly probable given the diversity of repository structures.)

---

### Pitfall 6: Circular and Implicit Dependencies in the Component Graph

**What goes wrong:** Component A says it requires Component B, which requires Component C, which requires Component A. The dependency resolver enters an infinite loop or returns the entire graph. Alternatively, dependencies are implicit (an agent's system prompt references an MCP by name in its instructions, but no formal dependency edge exists), so the resolver misses them entirely.

**Why it happens:** Component authors don't declare dependencies formally. Relationships must be inferred from content (prompts mentioning tool names, configs referencing server names). Inference produces both false positives (mentions don't always mean dependencies) and false negatives (indirect dependencies through shared concepts). Circular dependencies arise naturally when two components are complementary (a test runner agent and a test framework MCP).

**Consequences:** Circular dependencies crash the resolver or produce infinite recommendation sets. Implicit dependencies cause the "missing pieces" failure mode: user installs a command but it fails because the MCP it quietly requires isn't present.

**Warning signs:**
- Dependency resolution hangs or takes orders of magnitude longer than expected
- Users report installed components that reference other components they don't have
- The graph has cycles detected by topological sort
- Some components work only when manually paired with others

**Prevention:** Run cycle detection (Tarjan's algorithm) after every graph update. For circular dependencies, model them as "co-dependency" edges rather than hierarchical blockers. For implicit dependencies, use a two-pass inference: (1) parse component content for references to other component names/types, (2) validate inferred edges against actual component behavior. Set a maximum dependency chain depth (e.g., 5 hops) and flag anything deeper for manual review. Store both explicit and inferred edges with confidence scores, and use only high-confidence edges for automatic dependency resolution.

**Phase mapping:** Phase 02 (Ingestion) for detection, Phase 04 (Dependency Resolution) for handling.

**Confidence:** MEDIUM (dependency hell is well-documented in package managers: Gradle, pip, npm. Component-level dependency inference is novel territory.)

---

### Pitfall 7: MCP Response Latency Kills the Conversational Flow

**What goes wrong:** The MCP server takes 2-5 seconds to respond because it runs graph traversal, vector search, and score fusion synchronously. Claude Code waits, the user waits, and the "on-demand retrieval during conversation" promise breaks. Users stop using the tool because it's faster to manually browse the repository.

**Why it happens:** Graph traversal (PPR with convergence iterations), vector search (embedding + ANN lookup), and score fusion are each fast independently but compound when chained synchronously. KuzuDB's NetworkX integration adds Python overhead compared to native Cypher. Cold starts add further delay if the graph isn't preloaded.

**Consequences:** Adoption failure. A tool that takes 3+ seconds per query in a conversational flow will be abandoned regardless of recommendation quality.

**Warning signs:**
- P50 latency exceeds 1 second per retrieval call
- Users start typing queries to Claude Code directly instead of using the MCP tool
- Profile shows most time spent in graph traversal, not vector search
- Memory usage spikes when the full graph loads on first query

**Prevention:** Precompute PPR vectors for common seed nodes at ingestion time, not query time. Use KuzuDB's native algo extension for PageRank instead of NetworkX (documented performance difference). Run vector search and graph traversal in parallel, not sequentially. Cache the top-100 most-requested component sets. Set a latency budget: 500ms target, 1000ms hard limit. If the graph is too large for real-time PPR, precompute topic-level PPR vectors and do final refinement at query time.

**Phase mapping:** Phase 03 (Retrieval Engine) for architecture, Phase 07 (Evaluation) for benchmarking.

**Confidence:** MEDIUM (MCP latency concerns from CodeRabbit 2026, BytesRack 2026. Specific KuzuDB PPR latency is unverified.)

---

### Pitfall 8: Context Budget Exhaustion from Tool Catalog Bloat

**What goes wrong:** The MCP server exposes too many tools/endpoints, each consuming context tokens for their schema definitions. A 100-tool MCP burns 600+ tokens per conversation turn just on definitions before any actual retrieval happens. Combined with the retrieved component descriptions themselves, the total context consumption approaches the budget limits that this project was built to solve.

**Why it happens:** Feature creep in the MCP interface. Each new capability (search, browse, install, validate, compare, dependency-tree) becomes a separate tool with its own schema. The MCP protocol requires advertising all available tools upfront.

**Consequences:** Irony: the tool built to prevent context pollution becomes a source of context pollution. Net context budget for the user's actual work shrinks.

**Warning signs:**
- MCP tool definitions consume more than 500 tokens in Claude Code's context
- Users disable the MCP because it "uses too much context"
- Adding new MCP tools measurably reduces Claude Code's performance on unrelated tasks

**Prevention:** Limit the MCP to 3-5 tools maximum: `search_components`, `get_component_detail`, `install_components`, `check_dependencies`. Resist the urge to expose every internal capability as a separate tool. Use a single `search_components` tool with rich parameters instead of separate tools for different search modes. Measure total tool definition token count and set a hard ceiling (300 tokens for all tool schemas combined).

**Phase mapping:** Phase 06 (Serving/MCP Layer). Design the MCP interface after the retrieval engine works, not before.

**Confidence:** HIGH (GitHub issue #17668 on Claude Code context isolation, CodeRabbit 2026 context engineering article)

---

## Minor Pitfalls

Mistakes that cause annoyance but are fixable.

### Pitfall 9: KuzuDB NetworkX Bridge as Default Path

**What goes wrong:** Developers default to the NetworkX bridge for all graph algorithms because it's more familiar than Cypher, introducing Python object overhead for every graph operation. Performance degrades by 2-10x compared to native Cypher queries.

**Prevention:** Use KuzuDB's native algo extension for all production graph algorithms. Reserve NetworkX bridge for prototyping and one-off analysis only. The official docs explicitly note this performance difference.

**Phase mapping:** Phase 01 (Foundation). Establish this convention early.

**Confidence:** HIGH (KuzuDB official documentation)

---

### Pitfall 10: Testing Against a Single Repository Only

**What goes wrong:** All tests use davila7/claude-code-templates as the only input. Ingestion passes, retrieval works, dependencies resolve. Then a user points the tool at a different repository structure and everything breaks: different directory layout, different naming convention, different metadata format.

**Prevention:** Build a test harness with at least 3 synthetic repository structures during Phase 02: (1) davila7 structure, (2) flat directory with README-based metadata, (3) nested category directories with YAML frontmatter. If ingestion works for all three, the universal claim holds.

**Phase mapping:** Phase 02 (Ingestion). Build synthetic test repos alongside the ingestion pipeline.

**Confidence:** MEDIUM (standard software testing principle applied to this domain)

---

### Pitfall 11: Embedding Model Mismatch Between Ingestion and Query Time

**What goes wrong:** Component descriptions are embedded with one model at ingestion time, but queries use a different model (or a different version of the same model after an update). Cosine similarity scores become meaningless because the embedding spaces don't align.

**Prevention:** Pin the embedding model version in configuration. Store the model identifier alongside every embedding in the vector store. On model change, trigger a full re-embedding of all components. Never mix embeddings from different models in the same vector index.

**Phase mapping:** Phase 01 (Foundation). Embedding model pinning is a configuration concern.

**Confidence:** HIGH (well-documented vector search pitfall)

---

## Phase-Specific Warnings

| Phase | Likely Pitfall | Mitigation |
|-------|---------------|------------|
| 01 Foundation | Embedding model not pinned; KuzuDB NetworkX as default | Pin model in config; establish native Cypher convention |
| 02 Ingestion | Entity resolution skipped; schema validation absent; single-repo testing | Two-phase entity resolution; write/read schema separation; 3+ test repo structures |
| 03 Retrieval Engine | Over-retrieval noise; PPR alpha mistuning; hybrid weighting defaults | Flow-based pruning from day one; adaptive alpha; RRF fusion; latency budget |
| 04 Dependency Resolution | Circular deps crash resolver; implicit deps missed | Cycle detection (Tarjan); two-pass inference with confidence; max depth cap |
| 05 Orchestration | No orchestration pitfall identified | Standard patterns apply |
| 06 MCP Layer | Tool catalog bloat consuming context budget | Max 5 tools; 300-token schema ceiling |
| 07 Evaluation | No validation set for tuning; misleading recall metrics | Build 30+ query-expected-component pairs before tuning; measure precision alongside recall |

---

## Sources

### Academic Papers
- [PathRAG: Pruning Graph-based RAG with Relational Paths](https://arxiv.org/abs/2502.14902) - Flow-based pruning, redundancy as the core problem (HIGH confidence)
- [Retrieval-Augmented Generation with Graphs (GraphRAG) Survey](https://arxiv.org/abs/2501.00309) - Comprehensive survey of noise/retrieval trade-offs (HIGH confidence)
- [How Significant Are the Real Performance Gains? GraphRAG Evaluation](https://arxiv.org/html/2506.06331v1) - Context relevance drops with graph expansion (HIGH confidence)
- [HybridRAG: Integrating KGs and Vector Retrieval](https://arxiv.org/abs/2408.04948) - Hybrid fusion challenges (HIGH confidence)
- [Efficient Algorithms for PPR: A Survey](https://arxiv.org/html/2403.05198v1) - PPR computational challenges and tuning (HIGH confidence)
- [MLCPD: Universal AST Schema for Multi-Language Code](https://arxiv.org/html/2510.16357) - Universal parsing challenges (MEDIUM confidence)
- [Construction of Knowledge Graphs: State and Challenges](https://www.mdpi.com/2078-2489/15/8/509) - Entity resolution and graph quality (HIGH confidence)
- [When to Use Graphs in RAG](https://arxiv.org/html/2506.05690v2) - GraphRAG performance boundaries (HIGH confidence)

### Industry and Community
- [CodeRabbit: Ballooning Context in the MCP Era](https://www.coderabbit.ai/blog/handling-ballooning-context-in-the-mcp-era-context-engineering-on-steroids) - Context budget management (MEDIUM confidence)
- [Claude Code Issue #17668: MCP Context Isolation](https://github.com/anthropics/claude-code/issues/17668) - Tool catalog bloat problem (HIGH confidence)
- [MCP Server Best Practices 2026](https://www.cdata.com/blog/mcp-server-best-practices-2026) - MCP design patterns (MEDIUM confidence)
- [Stop the Latency: MCP on Dedicated Hardware](https://dev.to/bytesrack/stop-the-latency-why-mcp-servers-belong-on-dedicated-hardware-not-lambda-functions-169n) - MCP latency concerns (LOW confidence)
- [KuzuDB Official Documentation](https://docs.kuzudb.com/get-started/graph-algorithms/) - Native algo extension vs NetworkX (HIGH confidence)
- [Schema Drift: The Silent Killer of Data Pipelines](https://www.gambilldataengineering.com/data-engineering/how-to-survive-schema-drift-the-silent-killer-of-data-pipelines) - Schema drift patterns (MEDIUM confidence)

### Vault / Production Experience
- cross-vault-context.js PPR + flow pruning implementation (production-validated retrieval patterns)
- tdk-graph.json entity extraction and dedup patterns (production-validated entity resolution)
