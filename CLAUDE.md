# Skill Retriever Integration

**MCP server providing component recommendations for Claude Code tasks.**

When working on coding tasks, this system can suggest relevant skills, agents, commands, hooks, and MCPs from a curated index of 2,500+ components across 56 repositories.

## When to Query Skill Retriever

**Automatically query** when the user asks to:
- Build a new feature (search for relevant skills/agents)
- Set up a workflow (git, CI/CD, deployment)
- Add integrations (APIs, databases, external services)
- Implement patterns (authentication, testing, logging)
- Solve a recurring problem (debugging, performance, security)

**Skip querying** for:
- Simple one-line fixes
- Questions/explanations (not implementation tasks)
- Tasks with explicit "don't use external components"

## How to Query

```
search_components(query="<task description>", top_k=5)
```

Example queries:
- "git commit automation with conventional commits"
- "code review with security analysis"
- "test generation for Python functions"
- "deployment to AWS Lambda"

## Using Results

1. **Review recommendations** — Each result includes:
   - `id`: Component identifier
   - `name`: Human-readable name
   - `type`: skill, agent, command, hook, or mcp
   - `score`: Relevance score (0-1)
   - `rationale`: Why this component was recommended
   - `health`: Active/stale/abandoned status
   - `security`: Risk assessment (see Security section below)
   - `token_cost`: Context window impact

2. **Check security before installing** — Surface warnings for HIGH/CRITICAL:
   ```
   Security: ⚠️ CRITICAL (shell_injection, env_harvest_all)
   → "This component uses eval() with user input and accesses all env vars. Proceed?"
   ```

3. **Check dependencies** before installing:
   ```
   check_dependencies(component_ids=["id1", "id2"])
   ```

3. **Install** if user approves:
   ```
   install_components(component_ids=["id1"], target_dir=".claude")
   ```

## Feedback Loop

The system learns from usage patterns:

- **Automatic**: `install_components` records success/failure
- **Manual**: Use `report_outcome` after execution:
  ```
  report_outcome(component_id="...", outcome="used")      # Worked well
  report_outcome(component_id="...", outcome="removed")   # Didn't fit
  report_outcome(component_id="...", outcome="deprecated") # Superseded
  ```

This feedback improves future recommendations.

## Existing Z Commands Integration

When executing Z commands (`/z0` through `/z12`), consider these component searches:

| Command | Relevant Search Query |
|---------|----------------------|
| `/z0` | "web scraping course extraction authentication" |
| `/z1` | "deep analysis document processing vault" |
| `/z2` | "linkedin content generation social media" |
| `/z3` | "audio transcription speech to text" |
| `/z4` | "presentation slides deck generation" |
| `/z5` | "pdf processing document chunking" |
| `/z7` | "linkedin response comment generation" |
| `/z9` | "linkedin analytics engagement tracking" |
| `/z10` | "customer analysis mbr insights" |
| `/z11` | "competitive analysis gtm strategy" |

Example integration flow:
1. User invokes `/z2` (social media generation)
2. Before executing, query: `search_components("linkedin post generation social media content")`
3. If relevant skills found, suggest to user: "Found 2 skills that could help: linkedin-writer (0.89), content-optimizer (0.82). Install?"
4. Proceed with Z command, optionally using installed components

## GSD Integration

During `/gsd:plan-phase`, query for components relevant to the phase goal:

```python
# Phase: "Implement user authentication"
search_components("authentication oauth jwt session management")

# Phase: "Add API rate limiting"
search_components("api rate limiting throttling middleware")
```

Include recommended components in the PLAN.md as optional dependencies.

## Component Types by Task

| Task Category | Preferred Types |
|--------------|-----------------|
| Quick automation | command, hook |
| Complex workflow | skill, agent |
| External integration | mcp |
| Event-driven | hook |
| Parallel execution | agent |

## Abstraction Level Hints

The system detects query complexity:
- **Simple queries** (<300 chars) → suggests commands, hooks, settings
- **Moderate queries** → suggests skills, commands
- **Complex queries** (>600 chars) → suggests agents, MCPs, skills

Override with explicit hints:
- Include "agent" or "autonomous" → prioritizes agents
- Include "command" or "quick" → prioritizes commands
- Include "mcp" or "server" → prioritizes MCPs

## Security Scanning (SEC-01)

**Every search result includes security status.** The system scans for:

| Category | Risk | Examples |
|----------|------|----------|
| Data Exfiltration | HIGH | HTTP POST with data, webhook endpoints |
| Credential Access | CRITICAL | `process.env` harvesting, SSH key access |
| Privilege Escalation | CRITICAL | Shell injection, `eval()`, `sudo`, `rm -rf` |
| Obfuscation | HIGH | Hex encoding, unicode escapes |

**How to handle security warnings:**

```
# Search returns:
auth-jwt-skill
  Score: 0.89
  Security: ⚠️ MEDIUM (env_sensitive_keys)
  Findings: 1 - accesses JWT_SECRET from env

crypto-utils
  Score: 0.72
  Security: ✅ SAFE
```

**Actions based on risk level:**

| Risk | Action |
|------|--------|
| SAFE | Install without warning |
| LOW | Mention in passing |
| MEDIUM | Surface to user: "accesses X, expected?" |
| HIGH | Warn user, explain finding |
| CRITICAL | **Strong warning**, require explicit approval |

**Security tools:**

```python
# Scan specific component
security_scan(component_id="owner/repo/skill/name")

# Audit all indexed components
security_audit(risk_level="medium")  # Reports MEDIUM and above

# Backfill security scans (for components indexed before SEC-01)
backfill_security_scans(force_rescan=False)
```

**Known false positives:**
- `shell_injection` triggers on bash code blocks in markdown (e.g., `gh pr view $PR`)
- `webhook_post` flags legitimate Discord/Slack integrations
- Review CRITICAL findings manually before dismissing

## Troubleshooting

### Ingestion Failures

```python
# Check heal status
get_heal_status()
# Returns: failures with retry count, healable status

# Clear stuck failures
clear_heal_failures()

# Re-ingest failed repo
ingest_repo(repo_url="https://github.com/owner/repo")
```

**Common failure types:**
| Type | Cause | Fix |
|------|-------|-----|
| `CLONE_FAILED` | Network/auth issue | Check URL, retry |
| `NO_COMPONENTS` | Repo has no recognizable components | Expected, skip |
| `RATE_LIMITED` | GitHub API limit | Wait, retry later |
| `PARSE_ERROR` | Malformed component file | Report issue |

### Supported File Extensions

The ingestion pipeline recognizes these component file formats:

| Pattern | Strategy |
|---------|----------|
| `*.md` | All markdown strategies (Davila7, Flat, Generic, Plugin) |
| `*.md.txt` | All markdown strategies (added for repos like honnibal/claude-skills) |
| `*.py` | PythonModuleStrategy (docstring-bearing modules) |
| `package.json` | PackageJsonStrategy (npm packages) |
| `README.md` | ReadmeFallbackStrategy (catch-all) |

Markdown files must have YAML frontmatter with a `name` field to be indexed.

### Search Returns Empty

1. Check index is loaded: `sync_status()`
2. Verify component exists: `get_component_detail(component_id="...")`
3. Try broader query terms
4. Check type filter isn't too restrictive

### Installation Fails

```python
# Check dependencies first
check_dependencies(component_ids=["id1"])
# Returns: conflicts, missing deps

# If component not found in metadata:
ingest_repo(repo_url="<source repo>")
```

## Data Paths

| Data | Location |
|------|----------|
| Index | `~/.skill-retriever/data/` |
| Graph | `~/.skill-retriever/data/graph.json` |
| Vectors | `~/.skill-retriever/data/vectors/` |
| Metadata | `~/.skill-retriever/data/metadata.json` |
| Outcome tracker | `~/.skill-retriever/data/outcome-tracker.json` |
| Component memory | `~/.skill-retriever/data/component-memory.json` |
| Ingestion cache | `~/.skill-retriever/data/ingestion-cache.json` |
| Repo registry | `~/.skill-retriever/repo-registry.json` |
| Installed components | `.claude/` in target project |

## FalkorDB Graph Backend

The graph store supports a hybrid FalkorDB + NetworkX architecture:
- **FalkorDB** = persistent graph database (Cypher queries, survives restarts)
- **NetworkX** = in-memory mirror for PPR, path algorithms
- **Fallback** = if FalkorDB is unavailable, uses JSON (`graph.json`) like before

### Setup

FalkorDB runs via Docker (shared with TDK system):

```bash
# Check if FalkorDB is running
docker ps | grep falkordb

# If not running, start it
docker run -d --name falkordb_migration -p 6379:6379 falkordb/falkordb:latest
```

### Configuration

Environment variables (defaults work for local dev):

| Variable | Default | Description |
|----------|---------|-------------|
| `FALKORDB_HOST` | `localhost` | FalkorDB host |
| `FALKORDB_PORT` | `6379` | FalkorDB port |

### Migration

One-time migration from `graph.json` to FalkorDB:

```bash
uv run python scripts/migrate_to_falkordb.py
```

### How It Works

1. On startup, MCP server tries to connect to FalkorDB
2. If connected: syncs all nodes/edges into NetworkX mirror, uses write-through for updates
3. If FalkorDB is down: falls back to `NetworkXGraphStore` (loads from `graph.json`)
4. PPR and path algorithms always run on the NetworkX mirror (FalkorDB lacks built-in PPR)
