---
name: sweetlink
description: In-tab browser automation -- like Playwright but works in your current authenticated tab
version: "1.0"
author: steipete
tags:
  - browser-automation
  - devtools
  - playwright-alternative
  - in-tab
  - chrome
  - testing
  - smoke-tests
tools:
  - sweetlink open
  - sweetlink sessions
  - sweetlink smoke
  - sweetlink devtools
  - sweetlink daemon
---

# SweetLink - In-Tab Browser Automation

## Overview

SweetLink provides Playwright-style browser automation that works in your current authenticated tab rather than spinning up a headless browser. This solves the fundamental problem with traditional browser automation: authentication. Instead of managing cookies, tokens, and login flows in a separate browser instance, SweetLink drives the browser session you already have open, complete with all your authenticated state, extensions, and cookies.

The architecture uses a daemon process that attaches to a DevTools-enabled Chrome instance and a CLI that sends commands through the daemon. Session management, console/network telemetry, screenshot capture, and DOM selector discovery all work through this channel.

## Installation

```bash
# Requires Node.js 22+, pnpm, and mkcert for TLS
brew install mkcert nss
pnpm install
pnpm run build

# Trust the local certificate (one-time)
pnpm sweetlink trust-ca
```

## Key Commands / Usage

```bash
# Start the SweetLink daemon
pnpm exec sweetlink daemon

# Launch or reuse a controlled Chrome window
pnpm sweetlink open --controlled --path /dashboard

# Target a specific URL
pnpm sweetlink open --url http://localhost:4100/dashboard

# View active sessions (codename, heartbeat, socket state, errors)
pnpm sweetlink sessions

# Run smoke tests across configured routes
pnpm sweetlink smoke --routes main

# Force-click OAuth consent buttons
pnpm sweetlink devtools authorize
```

Configuration via `sweetlink.json` (or `sweetlink.config.json`), supporting `appUrl`, `prodUrl`, `daemonUrl`, health checks, cookie mappings, and smoke route presets.

## Architecture

SweetLink consists of two components. The **daemon** is a long-lived service that launches or attaches to a DevTools-enabled Chrome instance, forwards console/network telemetry, and executes remote evaluations. The **CLI** parses commands, resolves runtime defaults from `sweetlink.json`, and communicates with the daemon over secure WebSockets.

The flow: start daemon once, `sweetlink open` requests a session token, daemon launches or reuses Chrome with hydrated cookies, CLI receives health check confirmations, then commands like `smoke` or `devtools authorize` stream through the daemon via DevTools Protocol or Puppeteer. When the CLI exits, the browser stays alive for the next command. Next.js DevTools integration automatically calls `/_next/mcp` for structured error summaries.

## Integration with OS Agent

SweetLink fills a critical gap in OS agent architecture: authenticated web interaction. While Playwright and Puppeteer require managing separate browser sessions, SweetLink lets an agent interact with web applications in the user's actual browser context. This is essential for SaaS dashboards, internal tools, and any web application behind authentication. Combined with MCPorter for MCP orchestration and Peekaboo for screen capture, SweetLink enables agents to drive web applications exactly as a human would -- including handling OAuth flows, session cookies, and SPAs.

## Source

- Repository: https://github.com/steipete/sweetlink
- Stars: 74
- Language: TypeScript
