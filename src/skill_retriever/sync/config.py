"""Configuration for auto-sync functionality."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class SyncConfig(BaseModel):
    """Configuration for repository sync."""

    # Webhook settings
    webhook_port: int = Field(default=9847, description="Port for webhook server")
    webhook_secret: str | None = Field(default=None, description="GitHub webhook secret for HMAC validation")

    # Polling settings
    poll_interval_seconds: int = Field(default=3600, description="Polling interval in seconds (default: 1 hour)")
    poll_enabled: bool = Field(default=False, description="Enable polling for repos without webhooks")

    # Storage
    registry_path: Path = Field(
        default_factory=lambda: Path.home() / ".skill-retriever" / "repo-registry.json",
        description="Path to repository registry file",
    )

    # Ingestion settings
    incremental: bool = Field(default=True, description="Use incremental ingestion (skip unchanged files)")


# Default configuration
SYNC_CONFIG = SyncConfig()
