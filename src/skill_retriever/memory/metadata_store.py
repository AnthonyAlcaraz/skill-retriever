"""JSON-backed store for component metadata."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from skill_retriever.entities.components import ComponentMetadata


class MetadataStore:
    """JSON-backed store for component metadata.

    Provides simple CRUD operations for ComponentMetadata with
    JSON file persistence. Used by the installer to look up
    components by ID for installation.
    """

    def __init__(self, store_path: Path) -> None:
        """Initialize the metadata store.

        Args:
            store_path: Path to the JSON file for persistence.
        """
        self.store_path = store_path
        self._cache: dict[str, ComponentMetadata] = {}
        self._load()

    def _load(self) -> None:
        """Load metadata from JSON file if it exists."""
        if self.store_path.exists():
            from skill_retriever.entities.components import ComponentMetadata

            data = json.loads(self.store_path.read_text())
            for item in data:
                meta = ComponentMetadata.model_validate(item)
                self._cache[meta.id] = meta

    def save(self) -> None:
        """Persist all metadata to JSON file."""
        data = [m.model_dump(mode="json") for m in self._cache.values()]
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.store_path.write_text(json.dumps(data, indent=2, default=str))

    def get(self, component_id: str) -> ComponentMetadata | None:
        """Get metadata by component ID.

        Args:
            component_id: The component ID to look up.

        Returns:
            ComponentMetadata if found, None otherwise.
        """
        return self._cache.get(component_id)

    def add(self, metadata: ComponentMetadata) -> None:
        """Add or update a single component's metadata.

        Args:
            metadata: The ComponentMetadata to store.
        """
        self._cache[metadata.id] = metadata

    def add_many(self, components: list[ComponentMetadata]) -> None:
        """Add or update multiple components' metadata.

        Args:
            components: List of ComponentMetadata to store.
        """
        for comp in components:
            self._cache[comp.id] = comp

    def list_all(self) -> list[ComponentMetadata]:
        """Return all stored metadata.

        Returns:
            List of all ComponentMetadata in the store.
        """
        return list(self._cache.values())

    def __len__(self) -> int:
        """Return the number of stored components."""
        return len(self._cache)

    def __contains__(self, component_id: str) -> bool:
        """Check if a component ID exists in the store."""
        return component_id in self._cache
