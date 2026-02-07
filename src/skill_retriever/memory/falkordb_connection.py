"""FalkorDB connection wrapper with retry logic and health checking."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FalkorDBConfig:
    """Configuration for FalkorDB connection."""

    host: str = "localhost"
    port: int = 6379
    graph_name: str = "skill_retriever"
    max_retries: int = 3
    retry_base_delay: float = 0.5


class FalkorDBConnection:
    """Manages a connection to FalkorDB with retry logic.

    Uses the ``falkordb`` Python sync client. All graph operations
    go through ``execute_read`` / ``execute_write`` which handle
    retries with exponential backoff.
    """

    def __init__(self, config: FalkorDBConfig | None = None) -> None:
        self._config = config or FalkorDBConfig()
        self._db: Any = None
        self._graph: Any = None

    def connect(self) -> None:
        """Establish connection to FalkorDB with exponential backoff."""
        import falkordb  # type: ignore[import-untyped]

        last_exc: Exception | None = None
        for attempt in range(self._config.max_retries):
            try:
                self._db = falkordb.FalkorDB(
                    host=self._config.host,
                    port=self._config.port,
                )
                self._graph = self._db.select_graph(self._config.graph_name)
                # Verify connectivity with a simple query
                self._graph.query("RETURN 1")
                logger.info(
                    "Connected to FalkorDB at %s:%d graph=%s",
                    self._config.host,
                    self._config.port,
                    self._config.graph_name,
                )
                return
            except Exception as exc:
                last_exc = exc
                delay = self._config.retry_base_delay * (2**attempt)
                logger.warning(
                    "FalkorDB connection attempt %d/%d failed: %s (retry in %.1fs)",
                    attempt + 1,
                    self._config.max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)

        msg = f"Failed to connect to FalkorDB after {self._config.max_retries} attempts"
        raise ConnectionError(msg) from last_exc

    @property
    def graph(self) -> Any:
        """Return the FalkorDB graph handle, connecting if needed."""
        if self._graph is None:
            self.connect()
        return self._graph

    def execute_read(self, query: str, params: dict[str, Any] | None = None) -> Any:
        """Execute a read-only Cypher query with retry."""
        return self._execute(query, params, read_only=True)

    def execute_write(self, query: str, params: dict[str, Any] | None = None) -> Any:
        """Execute a write Cypher query with retry."""
        return self._execute(query, params, read_only=False)

    def _execute(
        self, query: str, params: dict[str, Any] | None, *, read_only: bool
    ) -> Any:
        """Execute a Cypher query with retry on transient failures."""
        last_exc: Exception | None = None
        for attempt in range(self._config.max_retries):
            try:
                if read_only:
                    return self.graph.ro_query(query, params or {})
                return self.graph.query(query, params or {})
            except Exception as exc:
                last_exc = exc
                if attempt < self._config.max_retries - 1:
                    delay = self._config.retry_base_delay * (2**attempt)
                    logger.warning(
                        "FalkorDB query attempt %d/%d failed: %s",
                        attempt + 1,
                        self._config.max_retries,
                        exc,
                    )
                    time.sleep(delay)
                    # Try reconnecting
                    self._graph = None

        msg = f"FalkorDB query failed after {self._config.max_retries} attempts"
        raise RuntimeError(msg) from last_exc

    def health_check(self) -> bool:
        """Check if the connection is alive."""
        try:
            self.graph.query("RETURN 1")
            return True
        except Exception:
            return False

    def close(self) -> None:
        """Close the FalkorDB connection."""
        self._graph = None
        self._db = None
        logger.info("FalkorDB connection closed")
