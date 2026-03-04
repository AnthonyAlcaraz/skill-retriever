---
name: axorcist
description: Swift wrapper for macOS Accessibility APIs with chainable, fuzzy-matched queries for UI inspection and automation
version: "1.0"
author: steipete
tags:
  - macos
  - accessibility
  - ui-automation
  - swift
  - axui-element
  - fuzzy-matching
  - gui-inspection
tools:
  - AXorcist.runCommand (query)
  - AXorcist.runCommand (action)
  - AXorcist.runCommand (observe)
  - AXorcist.runCommand (batch)
  - axorcist CLI
---

# AXorcist - macOS Accessibility API Wrapper

## Overview

AXorcist harnesses the macOS Accessibility APIs to provide type-safe, chainable, fuzzy-matched queries that read, click, and inspect any UI element on the system. It wraps `AXUIElement` in modern Swift patterns (async/await, structured concurrency, MainActor isolation) and exposes a command-based architecture for query, action, observation, and batch operations. Whether you are building automation tools, testing frameworks, or assistive technologies, AXorcist provides the foundational element discovery and interaction layer.

The key differentiator from raw Accessibility API usage: AXorcist adds fuzzy matching for element search, batch attribute fetching for performance, hierarchy navigation with caching, and a comprehensive command envelope system that makes operations composable and reproducible.

## Installation

```bash
# Build from source (requires macOS 14+, Xcode 16+, Swift 6.2)
git clone https://github.com/steipete/axorcist.git
cd axorcist
swift build

# As a Swift Package dependency
.package(url: "https://github.com/steipete/axorcist.git", from: "1.0.0")
```

Requires Accessibility permissions: System Settings > Privacy & Security > Accessibility.

## Key Commands / Usage

**Swift API:**
```swift
import AXorcist

let axorcist = AXorcist.shared

// Query for a button in Safari
let command = AXCommandEnvelope(
    commandID: "find-button",
    command: .query(QueryCommand(
        appName: "Safari",
        searchCriteria: [.role(.button), .title("Reload")]
    ))
)
let response = axorcist.runCommand(command)

// Perform an action on an element
let clickCommand = AXCommandEnvelope(
    commandID: "click-reload",
    command: .action(ActionCommand(
        element: foundElement,
        action: .press
    ))
)

// Element navigation
let element = Element(axUIElement)
let title = element.title
let role = element.role
let children = element.children()
try element.performAction(.press)
try element.setValue("Hello World")

// Permission handling
let hasPermissions = AXPermissionHelpers.hasAccessibilityPermissions()
let granted = await AXPermissionHelpers.requestPermissions()
for await hasPermissions in AXPermissionHelpers.permissionChanges() {
    // React to permission changes in real time
}
```

**CLI:**
```bash
# Query elements in an app
axorcist query --app Safari --role button --title "Reload"

# Click an element
axorcist action --app Safari --role button --title "Reload" --action press

# List all elements in a window
axorcist query --app TextEdit --all
```

## Architecture

AXorcist is built on three core classes. `AXorcist` (singleton, MainActor-isolated) is the central orchestrator that processes `AXCommandEnvelope` commands and returns `AXResponse` objects. `Element` wraps `AXUIElement` with type-safe property access, automatic CF-to-Swift value conversion, hierarchy navigation with caching, and batch attribute fetching. `AXPermissionHelpers` provides async/await permission handling with real-time monitoring via AsyncStream.

The command system supports four operation types: `query` (find elements by criteria with fuzzy matching), `action` (perform operations like press, setValue), `observe` (watch for accessibility notifications), and `batch` (execute multiple operations in one call). All operations use compile-time type safety for accessibility attributes.

## Integration with OS Agent

AXorcist is the "accessibility backbone" of a macOS OS agent. While Peekaboo provides screen capture and high-level GUI automation, AXorcist provides the low-level element discovery and inspection layer that powers precise interaction. An agent can query any application's UI tree, find specific elements by role/title/value with fuzzy matching, read element properties, and perform actions. Combined with Peekaboo (visual automation) and macos-automator-mcp (AppleScript/JXA), AXorcist enables agents to interact with macOS applications at both the visual and semantic levels.

## Source

- Repository: https://github.com/steipete/axorcist
- Stars: 170
- Language: Swift
