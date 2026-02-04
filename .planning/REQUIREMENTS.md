# Requirements: Skill Retriever

**Defined:** 2026-02-02
**Core Value:** Given a task description, return the minimal correct set of components with all dependencies resolved.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Ingestion

- [x] **INGS-01**: System can crawl and parse any component repository structure (davila7 as reference, but universal)
- [x] **INGS-02**: System indexes component metadata (name, type, tags, author, description) into graph store
- [x] **INGS-03**: System returns full component definition on demand (description, parameters, usage, dependencies)
- [x] **INGS-04**: System extracts git health signals (last update, commit frequency) per component

### Retrieval

- [x] **RETR-01**: User can search components by natural language description (vector-based semantic search)
- [x] **RETR-02**: User can filter results by component type (agent, skill, command, MCP, hook, setting, sandbox)
- [x] **RETR-03**: System returns ranked top-N results (default 5-10) with relevance scores
- [x] **RETR-04**: System combines vector search with graph traversal (PPR + flow pruning) for hybrid retrieval

### Graph & Dependencies

- [x] **GRPH-01**: System models component dependencies as directed graph edges (DEPENDS_ON, ENHANCES, CONFLICTS_WITH)
- [x] **GRPH-02**: System resolves transitive dependency chains via Cypher queries (multi-hop)
- [x] **GRPH-03**: Given a task description, system returns complete component set needed (task-to-set mapping)
- [x] **GRPH-04**: System validates component compatibility and surfaces conflicts before recommending

### Integration

- [x] **INTG-01**: System serves as MCP server (tools/list, tools/call) callable from Claude Code
- [x] **INTG-02**: System installs chosen components into .claude/ directory structure
- [x] **INTG-03**: Each recommendation includes graph-path rationale explaining why it was selected
- [x] **INTG-04**: System estimates context token cost per component and optimizes for minimal footprint

## v2 Requirements

Partial implementation complete. Some features implemented, others deferred.

### Auto-Sync

- [x] **SYNC-01**: System watches configured repositories for changes (webhook or polling)
- [x] **SYNC-02**: System auto-reingests repositories when upstream changes detected
- [x] **SYNC-03**: Incremental ingestion updates only changed components (not full rebuild)

### Advanced Retrieval

- **RETR-05**: Query rewriting/intent clarification for ambiguous queries (LLM-assisted) — *Deferred*
- [x] **RETR-06**: Abstraction level awareness (return command vs agent vs hook based on query complexity and project maturity)

### Learning

- **LRNG-01**: Track component selection/rejection patterns from usage — *Deferred*
- **LRNG-02**: Feed usage patterns back into ranking (collaborative filtering) — *Deferred*
- [x] **LRNG-03**: Track co-occurrence patterns (components frequently installed together)

### Health

- [x] **HLTH-01**: Surface component maintenance status (active vs abandoned)
- **HLTH-02**: Flag deprecated components with suggested replacements — *Deferred*

## Out of Scope

| Feature | Reason |
|---------|--------|
| Full package manager (versioning, updates) | npm took decades; our job is recommendation, not installation management |
| Component execution runtime | MCP servers should not execute arbitrary code; agent decides when/how |
| User accounts / ratings / reviews | Single-user tool, not a public marketplace |
| Real-time telemetry / monitoring | Log offline, feed back asynchronously |
| Natural language component creation | Retrieval only, not synthesis |
| Multi-tenant / RBAC | Personal tool; add layer later if needed |
| Auto-syncing with upstream repos | Provide rebuild/ingest command instead |
| LLM-in-the-loop for every query | Too slow (1-3s latency); reserve for offline enrichment |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| INGS-01 | Phase 2 | Complete |
| INGS-02 | Phase 2 | Complete |
| INGS-03 | Phase 2 | Complete |
| INGS-04 | Phase 2 | Complete |
| RETR-01 | Phase 4 | Complete |
| RETR-02 | Phase 4 | Complete |
| RETR-03 | Phase 4 | Complete |
| RETR-04 | Phase 4 | Complete |
| GRPH-01 | Phase 3 | Complete |
| GRPH-02 | Phase 5 | Complete |
| GRPH-03 | Phase 5 | Complete |
| GRPH-04 | Phase 5 | Complete |
| INTG-01 | Phase 6 | Complete |
| INTG-02 | Phase 6 | Complete |
| INTG-03 | Phase 6 | Complete |
| INTG-04 | Phase 6 | Complete |

**Coverage:**
- v1 requirements: 16 total (all complete)
- v2 requirements implemented: 6 (SYNC-01, SYNC-02, SYNC-03, RETR-06, LRNG-03, HLTH-01)
- v2 requirements deferred: 4 (RETR-05, LRNG-01, LRNG-02, HLTH-02)

---
*Requirements defined: 2026-02-02*
*Last updated: 2026-02-04 after SYNC-01/SYNC-02 implementation*
