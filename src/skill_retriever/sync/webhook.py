"""Webhook server for receiving GitHub push events."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import sys
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from aiohttp import web

if TYPE_CHECKING:
    from skill_retriever.sync.config import SyncConfig

logger = logging.getLogger(__name__)

# Configure logging to stderr (CRITICAL: never print to stdout for MCP compatibility)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class WebhookServer:
    """HTTP server for receiving GitHub webhook push events.

    Validates webhook signatures (if secret configured) and triggers
    re-ingestion callbacks when push events are received.
    """

    def __init__(
        self,
        config: SyncConfig,
        on_push: Callable[[str, str, str | None], Coroutine[Any, Any, None]] | None = None,
    ) -> None:
        """Initialize webhook server.

        Args:
            config: Sync configuration with webhook settings.
            on_push: Async callback for push events (owner, name, commit_sha).
        """
        self._config = config
        self._on_push = on_push
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None

    def _verify_signature(self, payload: bytes, signature: str) -> bool:
        """Verify GitHub webhook HMAC signature."""
        if not self._config.webhook_secret:
            # No secret configured, skip verification
            return True

        if not signature.startswith("sha256="):
            logger.warning("Invalid signature format: %s", signature)
            return False

        expected_sig = signature[7:]  # Remove "sha256=" prefix
        computed = hmac.new(
            self._config.webhook_secret.encode(),
            payload,
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(computed, expected_sig)

    async def _handle_webhook(self, request: web.Request) -> web.Response:
        """Handle incoming webhook request."""
        # Check event type
        event_type = request.headers.get("X-GitHub-Event", "")
        if event_type != "push":
            logger.debug("Ignoring non-push event: %s", event_type)
            return web.Response(text="OK", status=200)

        # Read payload
        try:
            payload = await request.read()
        except Exception:
            logger.exception("Failed to read webhook payload")
            return web.Response(text="Bad Request", status=400)

        # Verify signature
        signature = request.headers.get("X-Hub-Signature-256", "")
        if self._config.webhook_secret and not self._verify_signature(payload, signature):
            logger.warning("Webhook signature verification failed")
            return web.Response(text="Forbidden", status=403)

        # Parse payload
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            logger.exception("Failed to parse webhook JSON")
            return web.Response(text="Bad Request", status=400)

        # Extract repo info
        repo_data = data.get("repository", {})
        owner = repo_data.get("owner", {}).get("login") or repo_data.get("owner", {}).get("name")
        name = repo_data.get("name")
        commit_sha = data.get("after")  # SHA of the latest commit after push

        if not owner or not name:
            logger.warning("Missing repo info in webhook payload")
            return web.Response(text="Bad Request", status=400)

        logger.info("Received push event for %s/%s (commit: %s)", owner, name, commit_sha)

        # Trigger callback
        if self._on_push:
            try:
                await self._on_push(owner, name, commit_sha)
            except Exception:
                logger.exception("Push callback failed for %s/%s", owner, name)
                # Don't fail the webhook response
                return web.Response(text="OK (callback failed)", status=200)

        return web.Response(text="OK", status=200)

    async def _handle_health(self, _request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.Response(text="OK", status=200)

    async def start(self) -> None:
        """Start the webhook server."""
        self._app = web.Application()
        self._app.router.add_post("/webhook", self._handle_webhook)
        self._app.router.add_get("/health", self._handle_health)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()

        self._site = web.TCPSite(self._runner, "0.0.0.0", self._config.webhook_port)
        await self._site.start()

        logger.info("Webhook server started on port %d", self._config.webhook_port)

    async def stop(self) -> None:
        """Stop the webhook server."""
        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()

        logger.info("Webhook server stopped")

    async def run_forever(self) -> None:
        """Start server and run until cancelled."""
        await self.start()
        try:
            while True:
                await asyncio.sleep(3600)  # Sleep indefinitely
        except asyncio.CancelledError:
            await self.stop()
