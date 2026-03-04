---
name: imsg
description: CLI for Apple Messages.app to send and receive iMessage and SMS with attachment metadata
version: "1.0"
author: steipete
tags:
  - imessage
  - sms
  - macos
  - messaging
  - automation
  - swift
tools:
  - imsg chats
  - imsg history
  - imsg watch
  - imsg send
---

# imsg - iMessage and SMS Automation

## Overview

imsg is a macOS CLI for Apple's Messages.app that enables sending, reading, and streaming iMessage and SMS messages programmatically. It provides read-only database access for receiving messages (via `~/Library/Messages/chat.db`) and uses AppleScript for sending -- no private APIs. Features include phone normalization to E.164 for reliable buddy lookup, optional attachment metadata output (mime, name, path, missing flag), filtering by participants and date ranges, event-driven watch via filesystem events, and JSON output for tooling integration.

This tool bridges the gap between AI agents and the iMessage ecosystem, enabling automated responses, message monitoring, and notification workflows.

## Installation

```bash
# Build from source (requires macOS 14+ with Messages.app signed in)
git clone https://github.com/steipete/imsg.git
cd imsg && make build
# Binary at ./bin/imsg
```

Permissions required:
- **Full Disk Access** for your terminal (to read `~/Library/Messages/chat.db`)
- **Automation permission** for your terminal to control Messages.app (for sending)
- For SMS relay: enable "Text Message Forwarding" on your iPhone to this Mac

## Key Commands / Usage

```bash
# List recent conversations
imsg chats --limit 20 --json

# View message history for a chat
imsg history --chat-id 1 --limit 50 --attachments --json

# Filter by date range
imsg history --chat-id 1 --start 2026-01-01T00:00:00Z --end 2026-02-01T00:00:00Z --json

# Filter by participants
imsg history --chat-id 1 --participants +15551234567,+15559876543

# Live stream new messages (event-driven, low latency)
imsg watch --chat-id 1 --attachments --debounce 250ms --json

# Watch all chats
imsg watch --json

# Send a text message
imsg send --to "+14155551212" --text "Meeting at 3pm" --service imessage

# Send with attachment
imsg send --to "+14155551212" --text "See attached" --file ~/Desktop/pic.jpg --service imessage
```

JSON output includes: `id`, `chat_id`, `guid`, `reply_to_guid`, `sender`, `is_from_me`, `text`, `created_at`, `attachments` (array with filename, mime_type, total_bytes, is_sticker, original_path, missing), and `reactions`.

## Architecture

Built in Swift targeting macOS 14+. Read operations use direct SQLite access to `~/Library/Messages/chat.db` in read-only mode -- no database writes. The watch command uses filesystem events on the database file for low-latency notification of new messages (configurable debounce, default 250ms). Send operations use AppleScript via the Messages.app scripting bridge, keeping the implementation within Apple's sanctioned automation APIs. Phone numbers are normalized to E.164 format using the region flag (default US) for reliable matching.

The reusable Swift core lives in `Sources/IMsgCore` and can be consumed as a library target by other Swift applications.

## Integration with OS Agent

imsg enables an OS agent to participate in iMessage conversations. The `watch` command with JSON output provides a real-time event stream that an agent can consume for monitoring notifications, responding to messages, or triggering workflows based on incoming texts. The `send` command enables proactive outreach. Combined with gogcli (email) and wacli (WhatsApp), imsg completes the multi-channel communication layer of an OS agent, covering the three primary messaging platforms on Apple devices.

## Source

- Repository: https://github.com/steipete/imsg
- Stars: 614
- Language: Swift
