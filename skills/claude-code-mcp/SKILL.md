---
name: claude-code-mcp
description: Run Claude Code as a one-shot MCP server for agent-in-agent delegation
version: "1.0"
author: steipete
tags:
  - mcp
  - agent-delegation
  - claude-code
  - code-editing
  - multi-agent
tools:
  - claude_code
---

# Claude Code MCP - Agent-in-Agent Delegation

## Overview

Claude Code MCP wraps the Claude CLI as an MCP server, enabling any MCP client (Cursor, Windsurf, Claude Desktop, custom agents) to delegate complex coding tasks to Claude Code running in one-shot mode with permissions bypassed. This creates a powerful "agent-in-agent" pattern where a primary agent can offload file editing, multi-step operations, git workflows, and system access tasks to Claude Code as a sub-agent.

The key advantage: Claude Code is often better and faster at file editing than host editors. Multiple commands can be queued rather than executed one by one, saving context window space and reducing compacts. You can use a cheaper model in the outer agent and delegate expensive operations to Claude Code.

## Installation

```json
{
  "mcpServers": {
    "claude-code-mcp": {
      "command": "npx",
      "args": ["-y", "@steipete/claude-code-mcp@latest"]
    }
  }
}
```

Prerequisites:
- Node.js v20+
- Claude CLI installed (`npm install -g @anthropic-ai/claude-code`)
- One-time acceptance of `--dangerously-skip-permissions` flag

## Key Commands / Usage

The server exposes a single powerful `claude_code` tool that accepts any prompt:

```
# From any MCP client, invoke the tool:
claude_code("Fix the TypeScript compilation errors in src/auth.ts")
claude_code("Run the test suite and fix any failures")
claude_code("Create a git branch, make changes, and open a PR")
```

Environment variables:
- `CLAUDE_CLI_NAME`: Override the Claude CLI binary name or path (default: `claude`)
- `MCP_CLAUDE_DEBUG`: Enable debug logging (`true` for verbose output)

First-time setup requires running `claude --dangerously-skip-permissions` manually to accept terms.

## Architecture

The server launches Claude Code in one-shot mode (`--dangerously-skip-permissions`) for each invocation. The prompt is passed directly, and Claude Code handles all file system operations, git commands, and code editing autonomously. The response is returned through the MCP protocol back to the calling agent.

The architecture deliberately keeps things simple: one tool, one prompt, one response. The complexity lives in Claude Code's own capabilities rather than in the MCP server wrapper.

## Integration with OS Agent

In an OS agent architecture, claude-code-mcp serves as the "coding specialist" sub-agent. The orchestrating agent handles planning and routing while delegating implementation tasks to Claude Code. This pattern works well with model routing: use a cheaper model (Gemini, GPT-4o) for the outer loop and let Claude Code handle the heavy lifting with its own model. The outer agent can unstick itself from editor limitations by routing through Claude Code for file operations, git workflows, and multi-file refactors.

## Source

- Repository: https://github.com/steipete/claude-code-mcp
- Stars: 1087
- Language: JavaScript
