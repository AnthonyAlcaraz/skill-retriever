# Skill Retriever

## What This Is

An MCP server that builds a knowledge graph of Claude Code components (agents, skills, commands, MCPs, hooks, settings, sandbox) from any component repository, then uses hybrid retrieval (PPR + flow pruning + vector search) to recommend the optimal component set for a given task and project context. Claude Code calls it on demand, gets ranked recommendations with dependency-aware rationale, user picks, and the system installs.

## Core Value

Given a task description, return the minimal correct set of components with all dependencies resolved — no missing pieces, no context pollution, no wrong abstraction level.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Ingest component repositories (davila7/claude-code-templates as reference, but universal)
- [ ] Build knowledge graph of component relationships in KuzuDB
- [ ] Hybrid retrieval: pattern matching + vector search + graph traversal (PPR + flow pruning)
- [ ] Detect project context (existing codebase, framework, tech stack)
- [ ] Accept natural language task descriptions
- [ ] Return ranked component recommendations with rationale
- [ ] Resolve component dependencies (an agent that needs an MCP, a command that needs a hook)
- [ ] Detect conflicts between components (overlapping functionality, incompatible settings)
- [ ] Install chosen components into .claude/ directory structure
- [ ] Serve as MCP server callable from Claude Code

### Out of Scope

- Real-time continuous monitoring (on-demand only for v1) — adds complexity without clear value yet
- Component authoring/publishing — this is retrieval, not creation
- Component quality scoring/reviews — trust the marketplace, add quality signals later
- Custom component generation — retrieval only, not synthesis

## Context

**Source marketplace:** davila7/claude-code-templates — 1300+ components across 7 types:
- Agents (~329): Role-based system prompts (code-reviewer, security-auditor, etc.)
- Skills (~664): Progressive-disclosure instructions with references/ subdirectories
- Commands (~228): Slash commands organized by domain
- Settings (~64): JSON config files
- MCPs (~64): Model Context Protocol server configs
- Hooks (~52): Pre/post tool triggers
- Sandbox (3): Execution environments

**Marketplace structure:** Components organized by category (ai-research, database, security, etc.). Bundles exist in marketplace.json grouping related components. No formal dependency graph between individual components.

**Existing patterns to leverage:**
- PPR + flow pruning retrieval orchestrator (cross-vault-context.js, retrieval/)
- TDK knowledge graph pattern (tdk-graph.json — entity extraction + dedup)
- KuzuDB already in stack (social-graph.kuzu — 71MB graph database)
- Tool Selection Problem vault analysis — five-level maturity model, Level 5 = graph/ontology-driven
- PwC Tool-to-Agent bipartite graph — unified vector space with ownership edges
- DeepAgent tool memory — track usage patterns and success rates
- Colin context engine — dependency-aware skill freshness tracking

**The three compounding failure modes this solves:**
1. Context pollution — too many irrelevant components waste Claude's context window
2. Missing dependencies — install a command but miss the MCP it needs
3. Wrong abstraction — pick a complex agent when a simple command suffices

## Constraints

- **Graph DB**: KuzuDB (embedded, Cypher support, already in stack)
- **Interface**: MCP server protocol (callable from Claude Code)
- **Language**: Python (Iusztin virtual layers architecture)
- **Retrieval model**: PPR + flow-based pruning (proven in production retrieval orchestrator)
- **Universal ingestion**: Must handle any component repository, not hardcoded to davila7 structure

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| KuzuDB over Neo4j | Already in stack, embedded (no separate server), Cypher support | — Pending |
| MCP server over CLI | Claude Code native integration, on-demand retrieval during conversation | — Pending |
| PPR + flow pruning over pure vector search | Graph traversal resolves multi-hop dependencies that vector similarity misses | — Pending |
| Universal ingestion from start | Repository structure varies; clean graph model means only ingestion layer changes | — Pending |
| Python with Iusztin layers | Aligns with standard agentic project architecture, clean domain separation | — Pending |
| All 7 component types in v1 | Components interact across types (command needs MCP, agent needs hook) — partial coverage breaks dependency resolution | — Pending |

---
*Last updated: 2026-02-02 after initialization*
