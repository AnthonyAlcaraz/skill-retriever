---
name: conduit-mcp
description: MCP server for file system operations, web content processing, image compression, and archive management
version: "1.0"
author: steipete
tags:
  - mcp
  - file-operations
  - web-scraping
  - markdown-conversion
  - image-compression
  - archive
tools:
  - read (content, metadata, search)
  - write (content, move, copy, delete)
  - batch operations
---

# Conduit MCP - File and Web Operations Server

## Overview

Conduit MCP is a production-ready MCP server that provides comprehensive file system operations, web content processing, and data management capabilities to AI assistants. It operates within configurable allowed path territories for security, and handles file reading/writing, web page scraping with Markdown conversion (via Mozilla Readability), image compression (via Sharp), archive pack/unpack, checksum calculation, partial byte-range reading, and batch operations for efficiency.

The server is designed for agents that need reliable file I/O and web content retrieval without the complexity of managing separate tools for each operation.

## Installation

```json
{
  "mcpServers": {
    "conduit": {
      "command": "npx",
      "args": ["-y", "@steipete/conduit-mcp@beta"],
      "env": {
        "CONDUIT_ALLOWED_PATHS": "~/Documents:~/Projects:/tmp",
        "LOG_LEVEL": "INFO"
      }
    }
  }
}
```

Or for development:
```bash
git clone https://github.com/steipete/conduit-mcp.git
cd conduit-mcp && npm install
```

## Key Commands / Usage

The server exposes `read` and `write` tools with multiple operations:

**Reading content:**
```json
{"tool": "read", "operation": "content", "sources": ["~/Documents/report.txt", "https://example.com/article"], "format": "markdown"}
```

**Reading metadata:**
```json
{"tool": "read", "operation": "metadata", "sources": ["~/Documents/report.txt"]}
```

**Searching files:**
```json
{"tool": "read", "operation": "search", "path": "~/Projects", "pattern": "*.ts", "content_match": "TODO"}
```

**Format options:** `text` (default for text files), `base64` (binary-safe), `markdown` (web pages cleaned via Readability), `checksum` (cryptographic fingerprints with MD5/SHA1/SHA256/SHA512).

**Writing files, moving, copying, deleting, and archive operations** all follow the same structured JSON pattern through the `write` tool.

## Architecture

Conduit MCP enforces a security boundary through `CONDUIT_ALLOWED_PATHS`. All file operations are validated against these paths, and symlinks are followed to prevent escapes. Web content fetching uses Mozilla Readability for HTML-to-Markdown conversion, producing clean readable content from any URL. Image processing uses Sharp for compression without quality loss. The server supports batch operations where multiple read/write commands can be issued in a single request for efficiency.

## Integration with OS Agent

Conduit MCP provides the fundamental file I/O and web content layer for an OS agent. While Peekaboo handles visual interaction and SweetLink handles browser automation, Conduit handles the data plane: reading configuration files, writing generated code, fetching web documentation, processing images, and managing archives. The allowed-paths security model makes it safe for agents to operate without risking system-wide file access. Combined with MCPorter for discovery, an agent can use Conduit for all file-based operations across any project workspace.

## Source

- Repository: https://github.com/steipete/conduit-mcp
- Stars: 58
- Language: TypeScript
