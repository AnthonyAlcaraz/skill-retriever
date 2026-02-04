# Skill Retriever Integration

**MCP server providing component recommendations for Claude Code tasks.**

When working on coding tasks, this system can suggest relevant skills, agents, commands, hooks, and MCPs from a curated index of 1,000+ components across 23 repositories.

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
   - `token_cost`: Context window impact

2. **Check dependencies** before installing:
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

## Data Paths

| Data | Location |
|------|----------|
| Index | `~/.skill-retriever/` |
| Outcome tracker | `~/.skill-retriever/outcome-tracker.json` |
| Feedback engine | `~/.skill-retriever/feedback-engine.json` |
| Installed components | `.claude/` in target project |
