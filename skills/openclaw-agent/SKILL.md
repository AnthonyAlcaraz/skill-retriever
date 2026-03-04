---
name: openclaw-agent
description: Multi-channel personal AI assistant (185k stars) supporting WhatsApp, Telegram, Slack, Discord, iMessage, and more
version: "1.0"
author: openclaw
tags:
  - personal-assistant
  - multi-channel
  - whatsapp
  - telegram
  - slack
  - discord
  - imessage
  - always-on
tools:
  - openclaw
  - openclaw-cli
  - openclaw-gateway
---

# OpenClaw: Personal AI Assistant

## Overview
OpenClaw is the most popular open-source personal AI assistant (185k+ GitHub stars). It runs on your own devices and answers on the channels you already use: WhatsApp, Telegram, Slack, Discord, Google Chat, Signal, iMessage, Microsoft Teams, WebChat, plus extensions like BlueBubbles, Matrix, and Zalo. It can speak and listen on macOS/iOS/Android, render a live Canvas, and works with any LLM provider (Anthropic recommended). The Gateway is the control plane; the product is an always-on, single-user assistant that feels local and fast.

## Installation

```bash
# Requires Node.js 22+
npm install -g openclaw@latest

# Run the onboarding wizard (recommended path)
openclaw onboard --install-daemon

# Start the gateway
openclaw gateway --port 18789 --verbose
```

### From Source
```bash
git clone https://github.com/openclaw/openclaw.git
cd openclaw
pnpm install
pnpm ui:build
pnpm build
pnpm openclaw onboard --install-daemon
```

## Key Commands / Usage

### Core Commands
```bash
# Onboard and install daemon (launchd/systemd)
openclaw onboard --install-daemon

# Send a message
openclaw message send --to +1234567890 --message "Hello from OpenClaw"

# Talk to the assistant
openclaw agent --message "Ship checklist" --thinking high

# Health check
openclaw doctor

# Update
openclaw update --channel stable
```

### Channel Setup
```bash
# Pair WhatsApp
openclaw channel pair whatsapp

# Pair Telegram
openclaw channel pair telegram --bot-token YOUR_TOKEN

# Pair Slack
openclaw channel pair slack

# Pair Discord
openclaw channel pair discord --bot-token YOUR_TOKEN
```

### Security (DM Access)
```bash
# Default: pairing mode (unknown senders get a code)
openclaw pairing approve telegram ABC123

# Open mode (explicitly opt-in)
# Set dmPolicy="open" in config
```

### Development Channels
- **stable**: tagged releases, npm dist-tag `latest`
- **beta**: prerelease tags, npm dist-tag `beta`
- **dev**: head of `main`, npm dist-tag `dev`

## Architecture
OpenClaw has three layers: (1) **Gateway** - the always-on daemon that manages channel connections, message routing, and model failover (launchd on macOS, systemd on Linux), (2) **Agent** - LLM-powered reasoning with skills (extensible plugins), memory, and tool use, (3) **Channels** - adapters for each messaging platform handling authentication, message format conversion, and delivery. Model configuration supports OAuth-based subscriptions (Anthropic Pro/Max, OpenAI ChatGPT) with automatic failover between providers. The CLI wizard guides setup step by step.

## OS Agent Integration
OpenClaw provides the communication and interface layer for OS agents. Integration patterns: (1) use as the conversational front-end for desktop agents, receiving tasks via WhatsApp/Telegram and dispatching to computer-use agents, (2) extend with custom skills for domain-specific capabilities, (3) the always-on daemon architecture means your AI assistant is available 24/7 across all your devices, (4) combine with CuaBot for sandboxed computer-use (`cuabot openclaw` runs OpenClaw inside a CUA sandbox). Recommended model: Anthropic Opus 4.6 for prompt-injection resistance and long-context strength.

## Source
- Repository: https://github.com/openclaw/openclaw
- Stars: 185,784
- Language: TypeScript
- Documentation: https://docs.openclaw.ai
