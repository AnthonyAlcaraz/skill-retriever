---
name: gogcli
description: Google Suite CLI for Gmail, Calendar, Chat, Drive, Contacts, Tasks, Sheets, Docs, Slides, and more
version: "1.0"
author: steipete
tags:
  - google
  - gmail
  - calendar
  - drive
  - contacts
  - tasks
  - sheets
  - cli
  - workspace
tools:
  - gog gmail
  - gog calendar
  - gog chat
  - gog drive
  - gog contacts
  - gog tasks
  - gog sheets
  - gog docs
  - gog slides
  - gog classroom
  - gog groups
  - gog keep
  - gog auth
---

# gogcli - Google Suite in Your Terminal

## Overview

gogcli (command: `gog`) is a fast, script-friendly CLI for the entire Google Workspace ecosystem. It covers Gmail (search, send, labels, filters, drafts, delegation, vacation, watch/Pub/Sub), Calendar (events, conflicts, free/busy, team calendars, focus/OOO events, recurrence), Chat (spaces, messages, DMs), Classroom (courses, roster, coursework, submissions), Drive (list, search, upload, download, permissions, shared drives), Contacts (search, create, directory), Tasks (full CRUD with repeat schedules), Sheets (read, write, format), Docs/Slides (export, create, copy), People, Groups, and Keep (Workspace only).

All output is JSON-first for scripting and automation. It supports multiple Google accounts simultaneously with aliases, command allowlists for sandboxed/agent runs, secure credential storage via OS keyring, auto-refreshing tokens, least-privilege auth scopes, and Workspace service account domain-wide delegation.

## Installation

```bash
# Homebrew
brew install steipete/tap/gogcli

# Build from source
git clone https://github.com/steipete/gogcli.git
cd gogcli && make
./bin/gog --help
```

## Key Commands / Usage

```bash
# Authentication
gog auth credentials ~/Downloads/client_secret_....json
gog auth add you@gmail.com

# Gmail
gog gmail search "from:boss subject:urgent" --json
gog gmail send --to user@example.com --subject "Update" --body "Status report"
gog gmail send --to user@example.com --track  # with open tracking

# Calendar
gog calendar list --days 7 --json
gog calendar create --title "Meeting" --start "2026-02-15T10:00:00" --duration 1h
gog calendar freebusy --email team@company.com

# Drive
gog drive list --json
gog drive search "quarterly report" --type spreadsheet
gog drive upload ./report.pdf --folder "Reports"
gog drive download <file-id> --output ~/Downloads/

# Tasks
gog tasks list --json
gog tasks add "Review PR" --due 2026-02-15
gog tasks done <task-id>

# Sheets
gog sheets read <spreadsheet-id> --range "A1:D10" --json
gog sheets write <spreadsheet-id> --range "A1" --values '[["Name","Score"],["Alice","95"]]'

# Chat
gog chat spaces list
gog chat send --space <space-id> --message "Deployment complete"

# Multiple accounts
gog --account work@company.com gmail search "project update"
```

## Architecture

Built in Go for fast startup and cross-platform compatibility. Uses Google's official API client libraries with OAuth2 for personal accounts and service account domain-wide delegation for Workspace. Credentials are stored in the OS keyring or an encrypted on-disk keyring. Token refresh is automatic. The command allowlist feature restricts which top-level commands are available, enabling safe agent usage without exposing the full API surface.

JSON output mode is the default for machine consumption, with human-readable formatting available for interactive use. Calendar adds day-of-week fields in JSON mode for easier scheduling logic.

## Integration with OS Agent

gogcli is the Google Workspace bridge for an OS agent. An agent can read and send emails, check and create calendar events, search and manage Drive files, update spreadsheets, and manage tasks -- all through a consistent CLI interface with JSON output. The command allowlist feature is particularly valuable: configure it to restrict an agent to read-only Gmail + Calendar + Drive operations, preventing accidental email sends or file deletions. Combined with imsg (iMessage) and wacli (WhatsApp), gogcli completes the communication toolkit an OS agent needs.

## Source

- Repository: https://github.com/steipete/gogcli
- Stars: 1634
- Language: Go
