# Requirements: Skill Retriever

**Defined:** 2026-02-02
**Core Value:** Given a task description, return the minimal correct set of components with all dependencies resolved.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Ingestion

- [ ] **INGS-01**: System can crawl and parse any component repository structure (davila7 as reference, but universal)
- [ ] **INGS-02**: System indexes component metadata (name, type, tags, author, description) into graph store
- [ ] **INGS-03**: System returns full component definition on demand (description, parameters, usage, dependencies)
- [ ] **INGS-04**: System extracts git health signals (last update, commit frequency) per component

### Retrieval

- [ ] **RETR-01**: User can search components by natural language description (vector-based semantic search)
- [ ] **RETR-02**: User can filter results by component type (agent, skill, command, MCP, hook, setting, sandbox)
- [ ] **RETR-03**: System returns ranked top-N results (default 5-10) with relevance scores
- [ ] **RETR-04**: System combines vector search with graph traversal (PPR + flow pruning) for hybrid retrieval

### Graph & Dependencies

- [ ] **GRPH-01**: System models component dependencies as directed graph edges (DEPENDS_ON, ENHANCES, CONFLICTS_WITH)
- [ ] **GRPH-02**: System resolves transitive dependency chains via Cypher queries (multi-hop)
- [ ] **GRPH-03**: Given a task description, system returns complete component set needed (task-to-set mapping)
- [ ] **GRPH-04**: System validates component compatibility and surfaces conflicts before recommending

### Integration

- [ ] **INTG-01**: System serves as MCP server (tools/list, tools/call) callable from Claude Code
- [ ] **INTG-02**: System installs chosen components into .claude/ directory structure
- [ ] **INTG-03**: Each recommendation includes graph-path rationale explaining why it was selected
- [ ] **INTG-04**: System estimates context token cost per component and optimizes for minimal footprint

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Advanced Retrieval

- **RETR-05**: Query rewriting/intent clarification for ambiguous queries (LLM-assisted)
- **RETR-06**: Abstraction level awareness (return command vs prompt vs hook based on query granularity)

### Learning

- **LRNG-01**: Track component co-occurrence patterns from usage
- **LRNG-02**: Feed usage patterns back into ranking (collaborative filtering)

### Health

- **HLTH-01**: Surface component maintenance status (active vs abandoned)
- **HLTH-02**: Flag deprecated components with suggested replacements

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
| INGS-01 | Phase 2 | Pending |
| INGS-02 | Phase 2 | Pending |
| INGS-03 | Phase 2 | Pending |
| INGS-04 | Phase 2 | Pending |
| RETR-01 | Phase 4 | Pending |
| RETR-02 | Phase 4 | Pending |
| RETR-03 | Phase 4 | Pending |
| RETR-04 | Phase 4 | Pending |
| GRPH-01 | Phase 3 | Pending |
| GRPH-02 | Phase 5 | Complete |
| GRPH-03 | Phase 5 | Complete |
| GRPH-04 | Phase 5 | Complete |
| INTG-01 | Phase 6 | Pending |
| INTG-02 | Phase 6 | Pending |
| INTG-03 | Phase 6 | Pending |
| INTG-04 | Phase 6 | Pending |

**Coverage:**
- v1 requirements: 16 total
- Mapped to phases: 16
- Unmapped: 0

---
*Requirements defined: 2026-02-02*
*Last updated: 2026-02-02 after roadmap creation*
