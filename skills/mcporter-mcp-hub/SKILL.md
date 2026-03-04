---
name: mcporter
description: TypeScript runtime, CLI, and code-generation toolkit for the Model Context Protocol
version: "1.0"
author: steipete
tags:
  - mcp
  - orchestration
  - discovery
  - typescript
  - cli
  - code-generation
tools:
  - mcporter list
  - mcporter call
  - mcporter generate-cli
  - mcporter emit-ts
  - mcporter auth
  - mcporter config
---

# MCPorter - MCP Hub and Orchestration Toolkit

## Overview

MCPorter is the definitive toolkit for working with MCP (Model Context Protocol) servers. It provides zero-config discovery of MCP servers already configured on your system, direct calling of any tool across servers, composition of multi-server automations in TypeScript, and single-command CLI generation from any MCP server definition. It merges configs from Cursor, Claude Code/Desktop, Codex, Windsurf, OpenCode, and VS Code automatically, supporting stdio, HTTP, and SSE transports with OAuth caching built in.

The core insight: instead of hand-writing plumbing for each MCP server, MCPorter discovers what you have, exposes typed interfaces, and lets you call tools with a single command or TypeScript line.

## Installation

```bash
# Use directly with npx (no install required)
npx mcporter list
npx mcporter call context7.resolve-library-id libraryName=react

# Or install globally
npm install -g mcporter
```

## Key Commands / Usage

```bash
# List all discovered MCP servers and their tools
mcporter list
mcporter list context7 --schema
mcporter list --json

# Call any tool on any server
mcporter call linear.create_comment issueId:ENG-123 body:'Looks good!'
mcporter call 'linear.create_comment(issueId: "ENG-123", body: "Looks good!")'

# Generate a standalone CLI from an MCP server
mcporter generate-cli

# Emit TypeScript type definitions for an MCP server
mcporter emit-ts

# Ad-hoc connections to any endpoint
mcporter list https://mcp.linear.app/mcp --all-parameters
mcporter list --stdio "bun run ./local-server.ts" --env TOKEN=xyz

# OAuth authentication
mcporter auth https://vercel.com/mcp
```

## Architecture

MCPorter uses a layered discovery system. `createRuntime()` merges home config (`~/.mcporter/mcporter.json[c]`), project config, and imports from IDE-specific MCP configurations. It pools connections across transports so multiple calls reuse the same session. `createServerProxy()` exposes tools as camelCase methods with automatic JSON-schema validation, default injection, and `CallResult` helpers (`.text()`, `.markdown()`, `.json()`, `.content()`).

The listing output is formatted as TypeScript function signatures with JSDoc comments, making it copy-pasteable directly into agent code. Required parameters show inline; optional parameters are hidden unless `--all-parameters` is passed.

## Integration with OS Agent

MCPorter serves as the central hub in an OS agent architecture. It discovers and brokers access to every MCP server on the system through a single interface. An agent can enumerate available capabilities via `mcporter list --json`, call any tool via the unified `mcporter call` syntax, and generate typed clients for frequently used servers. This eliminates the need for agents to manage individual MCP server connections, making MCPorter the "service mesh" for MCP-based agent tooling.

## Source

- Repository: https://github.com/steipete/mcporter
- Stars: 1710
- Language: TypeScript
