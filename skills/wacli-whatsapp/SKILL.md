---
name: wacli
description: WhatsApp CLI for message sync, search, send, and contact/group management via whatsmeow
version: "0.2"
author: steipete
tags:
  - whatsapp
  - messaging
  - cli
  - sync
  - search
  - automation
  - go
tools:
  - wacli auth
  - wacli sync
  - wacli messages search
  - wacli send text
  - wacli send file
  - wacli history backfill
  - wacli media download
  - wacli groups list
  - wacli groups rename
  - wacli chats list
  - wacli doctor
---

# wacli - WhatsApp CLI

## Overview

wacli is a WhatsApp CLI built on the whatsmeow library (WhatsApp Web protocol) focused on best-effort local sync of message history with continuous capture, fast offline search, message sending, and contact/group management. It stores messages locally in `~/.wacli` with SQLite FTS5 for full-text search, supports multiple output formats (human-readable default, `--json` for machine consumption), and handles media downloads, history backfilling from the primary device, and group administration.

This is a third-party tool using the WhatsApp Web protocol and is not affiliated with WhatsApp.

## Installation

```bash
# Homebrew
brew install steipete/tap/wacli

# Build from source
git clone https://github.com/steipete/wacli.git
cd wacli
go build -tags sqlite_fts5 -o ./dist/wacli ./cmd/wacli
```

## Key Commands / Usage

```bash
# Authenticate (shows QR code, then bootstraps initial sync)
wacli auth

# Continuous sync (non-interactive, requires prior auth)
wacli sync --follow

# Search messages (offline, FTS5)
wacli messages search "meeting tomorrow" --json

# List chats
wacli chats list --limit 100 --json

# Send a text message
wacli send text --to 1234567890 --message "hello"

# Send a file with caption
wacli send file --to 1234567890 --file ./report.pdf --caption "Q4 report"
wacli send file --to 1234567890 --file /tmp/abc123 --filename report.pdf

# Download media from a message
wacli media download --chat 1234567890@s.whatsapp.net --id <message-id>

# Backfill older messages (requires primary device online)
wacli history backfill --chat 1234567890@s.whatsapp.net --requests 10 --count 50

# Group management
wacli groups list --json
wacli groups rename --jid 123456789@g.us --name "New name"

# Diagnostics
wacli doctor
```

Environment overrides: `WACLI_DEVICE_LABEL` (linked device label shown in WhatsApp), `WACLI_DEVICE_PLATFORM` (defaults to CHROME).

## Architecture

Built in Go using the whatsmeow library for the WhatsApp Web multi-device protocol. Messages are stored locally in SQLite with FTS5 enabled for full-text search. The `auth` command handles QR-based pairing and initial sync. The `sync --follow` command maintains a persistent connection for real-time message capture. History backfilling sends requests to the primary device (phone must be online) to fetch older messages on a per-chat basis, using the oldest locally stored message as the anchor.

Default storage is `~/.wacli` (override with `--store DIR`). Output is human-readable by default; pass `--json` for structured machine-readable output suitable for agent consumption.

## Integration with OS Agent

wacli provides the WhatsApp communication channel for an OS agent. An agent can monitor incoming messages via `sync --follow` with JSON output, search conversation history for context, send responses or notifications, and manage group memberships. The offline search capability means the agent can answer questions about past conversations without making API calls. Combined with imsg (iMessage) and gogcli (Gmail/Google), wacli completes the three-channel messaging layer. The `--json` output mode on all commands makes integration with agent pipelines straightforward.

## Source

- Repository: https://github.com/steipete/wacli
- Stars: 365
- Language: Go
