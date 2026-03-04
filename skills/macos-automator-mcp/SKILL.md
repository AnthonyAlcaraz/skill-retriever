---
name: macos-automator-mcp
description: MCP server for executing AppleScript and JXA (JavaScript for Automation) on macOS with 200+ pre-built recipes
version: "1.0"
author: steipete
tags:
  - macos
  - applescript
  - jxa
  - mcp
  - automation
  - knowledge-base
tools:
  - execute_script
  - get_scripting_tips
---

# macOS Automator MCP - AppleScript and JXA via MCP

## Overview

macOS Automator MCP is an MCP server that transforms an AI assistant into a macOS automation engine capable of executing AppleScript and JavaScript for Automation (JXA) scripts. It ships with a knowledge base of 200+ pre-programmed automation sequences covering common macOS operations: toggling dark mode, extracting URLs from Safari, managing Finder windows, controlling Mail, manipulating files, and much more. Scripts can be provided inline, as file paths, or by referencing knowledge base entries by ID.

The server handles both script languages, supports argument passing (positional via `arguments` or named via `input_data` with placeholder substitution), configurable timeouts, multiple output format modes, and debug logging of placeholder substitutions.

## Installation

```json
{
  "mcpServers": {
    "macos_automator": {
      "command": "npx",
      "args": ["-y", "@steipete/macos-automator-mcp@latest"]
    }
  }
}
```

Prerequisites:
- Node.js 18+
- macOS with Automation permissions (System Settings > Privacy & Security > Automation)
- Accessibility permissions for UI scripting via System Events

## Key Commands / Usage

**execute_script** - Run AppleScript or JXA:

```json
// Inline AppleScript
{
  "tool": "execute_script",
  "script_content": "tell application \"Finder\" to get name of every window",
  "language": "applescript"
}

// Inline JXA
{
  "tool": "execute_script",
  "script_content": "Application('Safari').windows[0].currentTab.url()",
  "language": "javascript"
}

// From file
{
  "tool": "execute_script",
  "script_path": "/Users/me/scripts/backup.applescript"
}

// From knowledge base
{
  "tool": "execute_script",
  "kb_script_id": "toggle-dark-mode"
}

// With named inputs (knowledge base placeholder substitution)
{
  "tool": "execute_script",
  "kb_script_id": "send-email",
  "input_data": {
    "recipient": "user@example.com",
    "subject": "Automated Report",
    "body": "Here is the daily summary."
  }
}
```

**get_scripting_tips** - Discover available knowledge base scripts:
```json
{"tool": "get_scripting_tips", "query": "safari"}
```

Options: `timeout_seconds` (default 60), `output_format_mode` (auto/human_readable/structured_error/direct), `include_executed_script_in_output`, `include_substitution_logs`.

## Architecture

The server runs as a standard MCP stdio server via Node.js. When `execute_script` is called, it determines the script source (inline, file path, or knowledge base ID), applies any placeholder substitutions for `input_data`/`arguments`, selects the appropriate language handler, and executes via `osascript` with configurable output formatting flags. The knowledge base is a curated collection of scripts indexed by ID with metadata including `argumentsPrompt` descriptions.

For AppleScript, output modes control the `-s` flag: `h` for human-readable, `s` for structured errors, `ss` for structured output and errors. JXA defaults to direct output (no `-s` flags). The `get_scripting_tips` tool provides discovery of the 200+ built-in scripts with search by keyword.

## Integration with OS Agent

macOS Automator MCP is the "system scripting" layer of a macOS OS agent. While Peekaboo handles visual automation and AXorcist handles accessibility-level interaction, macOS Automator provides the AppleScript/JXA execution layer for operations that are most naturally expressed as scripts: file management, application control, system preferences, email composition, calendar manipulation, and inter-app communication. The 200+ knowledge base scripts mean an agent can perform common macOS operations without writing scripts from scratch. Combined with the other steipete tools, it enables an agent to automate any aspect of the macOS desktop.

## Source

- Repository: https://github.com/steipete/macos-automator-mcp
- Stars: 630
- Language: TypeScript
