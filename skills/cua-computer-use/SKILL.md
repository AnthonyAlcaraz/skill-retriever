---
name: cua-computer-use
description: Open infrastructure for computer-use agents with sandboxes, SDKs, and benchmarks
version: "1.0"
author: trycua
tags:
  - computer-use
  - sandbox
  - infrastructure
  - macos-virtualization
  - benchmarks
  - agent-sdk
tools:
  - cua
  - cuabot
  - lume
  - cua-bench
---

# CUA: Computer-Use Agent Infrastructure

## Overview
CUA (Computer-Use Agent) is open-source infrastructure for building, benchmarking, and deploying agents that control full desktops. It provides sandboxed virtual machines (macOS, Linux, Windows), Python SDKs for agent development, and benchmarks for evaluation. CuaBot gives any coding agent (Claude Code, OpenClaw) a seamless sandbox with H.265 streaming, shared clipboard, and audio. Lume handles macOS virtualization natively on Apple Silicon.

## Installation

### CuaBot (Quickstart)
```bash
npx cuabot                   # Setup onboarding
cuabot claude                # Run Claude Code in sandbox
cuabot openclaw              # Run OpenClaw in sandbox
cuabot chromium              # Run browser in sandbox
```

### Python SDK
```bash
pip install cua
# Requires Python 3.12 or 3.13
```

### Lume (macOS Virtualization)
```bash
brew install trycua/tap/lume
lume create --os macos
lume start
```

## Key Commands / Usage

### Agent Development
```python
from computer import Computer
from agent import ComputerAgent

computer = Computer(os_type="linux", provider_type="cloud")
agent = ComputerAgent(
    model="anthropic/claude-sonnet-4-5-20250929",
    computer=computer
)

async for result in agent.run([
    {"role": "user", "content": "Open Firefox and search for Cua"}
]):
    print(result)
```

### CuaBot Commands
```bash
cuabot --screenshot           # Take screenshot
cuabot --type "hello"         # Type text
cuabot --click 100 200       # Click at coordinates
cuabot --click 100 200 right # Right-click
```

### Benchmarking
```bash
cua-bench run --benchmark osworld --agent my_agent
```

## Architecture
CUA has four components: (1) **CuaBot** - co-op computer-use for any agent, with native window rendering via H.265 streaming, (2) **Cua SDK** - Python library for building agents that see screens, click buttons, and complete tasks, (3) **Lume** - macOS/Linux virtualization layer using Apple's Virtualization.framework for fast VM creation, (4) **Cua-Bench** - standardized benchmarks and RL environments for agent evaluation. The system supports both cloud (hosted VMs) and local (Lume) deployment.

## OS Agent Integration
CUA provides the infrastructure layer that other OS agents run on top of. Instead of building your own VM management, screenshot pipeline, or input injection, CUA handles the plumbing. Use cases: (1) sandbox any AI coding agent for safe computer-use, (2) run benchmarks to evaluate agent performance, (3) deploy production computer-use agents with proper isolation. Built-in support for agent-browser (web) and agent-device (iOS, Android) extensions.

## Source
- Repository: https://github.com/trycua/cua
- Stars: 12,488
- Language: Python
- Documentation: https://cua.ai/docs
