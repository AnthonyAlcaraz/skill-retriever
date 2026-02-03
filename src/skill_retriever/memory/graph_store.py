"""Graph store protocol and NetworkX implementation for component knowledge graph."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import networkx as nx

if TYPE_CHECKING:
    from skill_retriever.entities.graph import GraphEdge, GraphNode


@runtime_checkable
class GraphStore(Protocol):
    """Protocol for graph storage backends."""

    def add_node(self, node: GraphNode) -> None: ...

    def add_edge(self, edge: GraphEdge) -> None: ...

    def get_node(self, node_id: str) -> GraphNode | None: ...

    def get_neighbors(self, node_id: str) -> list[GraphNode]: ...

    def get_edges(self, node_id: str) -> list[GraphEdge]: ...

    def personalized_pagerank(
        self,
        seed_ids: list[str],
        alpha: float = 0.85,
        top_k: int = 10,
    ) -> list[tuple[str, float]]: ...

    def node_count(self) -> int: ...

    def edge_count(self) -> int: ...

    def save(self, path: str) -> None: ...

    def load(self, path: str) -> None: ...


class NetworkXGraphStore:
    """NetworkX-backed directed graph store with Personalized PageRank support."""

    def __init__(self) -> None:
        self._graph: nx.DiGraph[str] = nx.DiGraph()

    def add_node(self, node: GraphNode) -> None:
        """Add a node to the graph, storing attributes as plain dicts."""
        self._graph.add_node(
            node.id,
            component_type=str(node.component_type),
            label=node.label,
            embedding_id=node.embedding_id,
        )

    def add_edge(self, edge: GraphEdge) -> None:
        """Add a directed edge to the graph, storing attributes as plain dicts."""
        attrs: dict[str, str | float] = {
            "edge_type": str(edge.edge_type),
            "weight": edge.weight,
        }
        for k, v in edge.metadata.items():
            attrs[k] = v
        self._graph.add_edge(edge.source_id, edge.target_id, **attrs)

    def get_node(self, node_id: str) -> GraphNode | None:
        """Reconstruct a GraphNode from stored NX node data, or None if not found."""
        if node_id not in self._graph:
            return None
        from skill_retriever.entities.components import ComponentType
        from skill_retriever.entities.graph import GraphNode as _GraphNode

        data: dict[str, Any] = dict(self._graph.nodes[node_id])
        return _GraphNode(
            id=node_id,
            component_type=ComponentType(data["component_type"]),
            label=data["label"],
            embedding_id=data.get("embedding_id", ""),
        )

    def get_neighbors(self, node_id: str) -> list[GraphNode]:
        """Return all in-neighbors and out-neighbors as GraphNode objects."""
        if node_id not in self._graph:
            return []
        successor_ids: list[str] = list(self._graph.successors(node_id))
        predecessor_ids: list[str] = list(self._graph.predecessors(node_id))
        neighbor_ids = sorted(set(successor_ids) | set(predecessor_ids))
        nodes: list[GraphNode] = []
        for nid in neighbor_ids:
            node = self.get_node(nid)
            if node is not None:
                nodes.append(node)
        return nodes

    def get_edges(self, node_id: str) -> list[GraphEdge]:
        """Return all edges where node_id is source or target."""
        if node_id not in self._graph:
            return []
        from skill_retriever.entities.graph import EdgeType
        from skill_retriever.entities.graph import GraphEdge as _GraphEdge

        edges: list[GraphEdge] = []
        # Outgoing edges
        out_edges: Any = self._graph.out_edges(node_id, data=True)
        for _, target, data in out_edges:
            target_id: str = str(target)
            meta = {
                k: v for k, v in data.items() if k not in ("edge_type", "weight")
            }
            edges.append(
                _GraphEdge(
                    source_id=node_id,
                    target_id=target_id,
                    edge_type=EdgeType(data["edge_type"]),
                    weight=float(data.get("weight", 1.0)),
                    metadata=meta,
                )
            )
        # Incoming edges
        in_edges: Any = self._graph.in_edges(node_id, data=True)
        for source, _, data in in_edges:
            source_id: str = str(source)
            meta = {
                k: v for k, v in data.items() if k not in ("edge_type", "weight")
            }
            edges.append(
                _GraphEdge(
                    source_id=source_id,
                    target_id=node_id,
                    edge_type=EdgeType(data["edge_type"]),
                    weight=float(data.get("weight", 1.0)),
                    metadata=meta,
                )
            )
        return edges

    def personalized_pagerank(
        self,
        seed_ids: list[str],
        alpha: float = 0.85,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Run Personalized PageRank from seed nodes, returning top_k ranked results."""
        if self._graph.number_of_nodes() == 0 or not seed_ids:
            return []

        # Build personalization vector: all nodes get 0, seeds get equal share
        all_nodes: list[str] = list(self._graph.nodes)
        personalization: dict[str, float] = {n: 0.0 for n in all_nodes}
        valid_seeds = [s for s in seed_ids if s in self._graph]
        if not valid_seeds:
            return []
        seed_weight = 1.0 / len(valid_seeds)
        for sid in valid_seeds:
            personalization[sid] = seed_weight

        # High alpha values need more iterations to converge
        max_iter = 200 if alpha > 0.9 else 100
        scores: dict[str, float] = nx.pagerank(
            self._graph,
            alpha=alpha,
            personalization=personalization,
            weight="weight",
            max_iter=max_iter,
        )

        # Exclude seed nodes from results
        seed_set = set(valid_seeds)
        ranked = [
            (nid, score)
            for nid, score in scores.items()
            if nid not in seed_set
        ]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def node_count(self) -> int:
        """Return the number of nodes in the graph."""
        return int(self._graph.number_of_nodes())

    def edge_count(self) -> int:
        """Return the number of edges in the graph."""
        return int(self._graph.number_of_edges())

    def save(self, path: str) -> None:
        """Serialize graph to JSON file using node-link format."""
        data: dict[str, Any] = nx.node_link_data(self._graph)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        """Deserialize graph from JSON file using node-link format."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self._graph = nx.node_link_graph(data, directed=True)  # pyright: ignore[reportUnknownMemberType]
