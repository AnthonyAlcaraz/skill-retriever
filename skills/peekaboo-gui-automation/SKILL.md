---
name: peekaboo
description: macOS screen capture, AI analysis, and full GUI automation via CLI and MCP server
version: "3.0"
author: steipete
tags:
  - macos
  - screen-capture
  - gui-automation
  - accessibility
  - mcp
  - ai-vision
  - swift
tools:
  - peekaboo see
  - peekaboo click
  - peekaboo type
  - peekaboo press
  - peekaboo hotkey
  - peekaboo scroll
  - peekaboo swipe
  - peekaboo drag
  - peekaboo move
  - peekaboo window
  - peekaboo app
  - peekaboo menu
  - peekaboo menubar
  - peekaboo dock
  - peekaboo dialog
  - peekaboo image
  - peekaboo list
  - peekaboo agent
---

# Peekaboo - macOS Screen Capture and GUI Automation

## Overview

Peekaboo is the most comprehensive macOS GUI automation toolkit available. Version 3 provides pixel-accurate screen capture (windows, screens, menu bar) with optional Retina 2x scaling, a natural-language agent that chains tools autonomously (see, click, type, scroll, hotkey, menu, window, app, dock, space), menu and menubar discovery with structured JSON, multi-provider AI vision (GPT-5.1, Claude 4.x, Grok 4-fast, Gemini 2.5, Ollama), and both a native CLI and MCP server exposing the same tool set.

Peekaboo bridges the gap between "seeing the screen" and "acting on it" that most automation tools miss. An agent can capture a window, get annotated element IDs, then click, type, and scroll using those IDs in a closed loop.

## Installation

```bash
# Homebrew (macOS app + CLI)
brew install steipete/tap/peekaboo

# MCP server (Node 22+, no global install)
npx -y @steipete/peekaboo
```

Requires macOS 15+ (Sequoia), Screen Recording + Accessibility permissions.

## Key Commands / Usage

```bash
# Capture full screen at Retina scale
peekaboo image --mode screen --retina --path ~/Desktop/screen.png

# See an app's UI and get annotated element IDs
peekaboo see --app Safari --json-output

# Click by element label from a snapshot
peekaboo click --on "Reload this page" --snapshot "$SNAPSHOT"

# Type text into a focused field
peekaboo type --text "Hello world" --delay-ms 50

# Keyboard shortcuts
peekaboo hotkey cmd,shift,t

# Window management
peekaboo window list
peekaboo window move --app Safari --x 0 --y 0 --width 1920 --height 1080

# App lifecycle
peekaboo app launch Safari
peekaboo app quit TextEdit

# Natural-language automation
peekaboo agent "Open Notes and create a TODO list with three items"
peekaboo agent --model gpt-5.1 --max-steps 20 "Fill out the form in Safari"
```

## Architecture

Peekaboo is built in Swift 6.2 targeting macOS 15+. The core uses macOS Accessibility APIs for element discovery and interaction, ScreenCaptureKit for pixel-accurate capture, and a snapshot system that annotates captured images with element IDs for click targeting. The MCP server wraps the same Swift tooling via a Node.js transport layer. The agent mode chains tools in a loop: see -> reason -> act -> verify, using configurable AI providers for the reasoning step.

## Integration with OS Agent

Peekaboo is the "eyes and hands" of a macOS OS agent. It provides the complete perception-action loop: capture the screen state, analyze it with AI vision, then act through clicks, typing, and keyboard shortcuts. Combined with axorcist (accessibility queries) and macos-automator-mcp (AppleScript/JXA), Peekaboo enables an agent to operate any macOS application as a human would. The MCP server mode means any MCP-compatible agent can use Peekaboo's capabilities without additional integration work.

## Source

- Repository: https://github.com/steipete/peekaboo
- Stars: 2018
- Language: Swift
