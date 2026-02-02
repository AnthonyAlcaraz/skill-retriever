"""Domain models for knowledge graph relationships between components."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from skill_retriever.entities.components import ComponentType  # noqa: TC001


class EdgeType(StrEnum):
    """Relationship types between components in the knowledge graph."""

    DEPENDS_ON = "depends_on"
    ENHANCES = "enhances"
    CONFLICTS_WITH = "conflicts_with"
    BUNDLES_WITH = "bundles_with"
    SAME_CATEGORY = "same_category"


class GraphNode(BaseModel):
    """A node in the component knowledge graph.

    Maps 1:1 with a ComponentMetadata entry via shared ``id``.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    component_type: ComponentType
    label: str
    embedding_id: str = ""


class GraphEdge(BaseModel):
    """A directed edge in the component knowledge graph."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    metadata: dict[str, str] = Field(default_factory=dict)
