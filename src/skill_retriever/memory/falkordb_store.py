"""Hybrid FalkorDB + NetworkX graph store.

FalkorDB is the persistent storage layer (Cypher CRUD, survives restarts).
NetworkX is the in-memory mirror for PPR, ``all_simple_paths``, ``descendants``.

Write-through: every ``add_node``/``add_edge`` writes to both FalkorDB and NetworkX.
Startup: sync full graph from FalkorDB into NetworkX mirror.
Fallback: if FalkorDB is down, load from ``graph.json`` (current behavior).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import networkx as nx

if TYPE_CHECKING:
    from skill_retriever.entities.graph import GraphEdge, GraphNode
    from skill_retriever.memory.falkordb_connection import FalkorDBConnection

logger = logging.getLogger(__name__)

# Batch size for UNWIND operations during ingestion
_BATCH_SIZE = 200

# Mapping from EdgeType string to FalkorDB relationship type
_EDGE_TYPE_TO_REL: dict[str, str] = {
    "depends_on": "DEPENDS_ON",
    "enhances": "ENHANCES",
    "conflicts_with": "CONFLICTS_WITH",
    "bundles_with": "BUNDLES_WITH",
    "same_category": "SAME_CATEGORY",
}

_REL_TO_EDGE_TYPE: dict[str, str] = {v: k for k, v in _EDGE_TYPE_TO_REL.items()}


class FalkorDBGraphStore:
    """Hybrid graph store: FalkorDB for persistence, NetworkX for algorithms.

    Implements the full ``GraphStore`` Protocol. When FalkorDB is unavailable,
    falls back to JSON-based persistence (same as ``NetworkXGraphStore``).
    """

    def __init__(
        self,
        connection: FalkorDBConnection | None = None,
        fallback_path: str = "",
    ) -> None:
        self._conn = connection
        self._fallback_path = fallback_path
        self._graph: nx.DiGraph[str] = nx.DiGraph()
        self._online = connection is not None

    # ------------------------------------------------------------------
    # Sync: FalkorDB -> NetworkX
    # ------------------------------------------------------------------

    def sync_from_falkordb(self) -> None:
        """Bulk-load all nodes and edges from FalkorDB into the NetworkX mirror."""
        if self._conn is None:
            logger.warning("No FalkorDB connection, skipping sync")
            return

        try:
            # Load nodes
            result = self._conn.execute_read(
                "MATCH (c:Component) "
                "RETURN c.id AS id, c.component_type AS ct, "
                "c.label AS label, c.embedding_id AS eid"
            )
            node_count = 0
            for row in result.result_set:
                node_id = str(row[0])
                self._graph.add_node(
                    node_id,
                    component_type=str(row[1]) if row[1] else "",
                    label=str(row[2]) if row[2] else "",
                    embedding_id=str(row[3]) if row[3] else "",
                )
                node_count += 1

            # Load edges
            result = self._conn.execute_read(
                "MATCH (s:Component)-[r]->(t:Component) "
                "RETURN s.id AS sid, t.id AS tid, type(r) AS etype, r.weight AS w"
            )
            edge_count = 0
            for row in result.result_set:
                source_id = str(row[0])
                target_id = str(row[1])
                rel_type = str(row[2])
                weight = float(row[3]) if row[3] is not None else 1.0
                edge_type = _REL_TO_EDGE_TYPE.get(rel_type, rel_type.lower())
                self._graph.add_edge(
                    source_id,
                    target_id,
                    edge_type=edge_type,
                    weight=weight,
                )
                edge_count += 1

            self._online = True
            logger.info(
                "Synced from FalkorDB: %d nodes, %d edges", node_count, edge_count
            )
        except Exception:
            logger.exception("Failed to sync from FalkorDB, falling back to offline mode")
            self._online = False

    # ------------------------------------------------------------------
    # Write-through operations
    # ------------------------------------------------------------------

    def add_node(self, node: GraphNode) -> None:
        """Add a node to both FalkorDB and NetworkX."""
        # NetworkX
        self._graph.add_node(
            node.id,
            component_type=str(node.component_type),
            label=node.label,
            embedding_id=node.embedding_id,
        )
        # FalkorDB
        if self._conn is not None and self._online:
            try:
                self._conn.execute_write(
                    "MERGE (c:Component {id: $id}) "
                    "SET c.component_type = $ct, c.label = $label, c.embedding_id = $eid",
                    {
                        "id": node.id,
                        "ct": str(node.component_type),
                        "label": node.label,
                        "eid": node.embedding_id,
                    },
                )
            except Exception:
                logger.exception("FalkorDB write failed for node %s", node.id)

    def add_edge(self, edge: GraphEdge) -> None:
        """Add an edge to both FalkorDB and NetworkX."""
        # NetworkX
        attrs: dict[str, str | float] = {
            "edge_type": str(edge.edge_type),
            "weight": edge.weight,
        }
        for k, v in edge.metadata.items():
            attrs[k] = v
        self._graph.add_edge(edge.source_id, edge.target_id, **attrs)

        # FalkorDB
        if self._conn is not None and self._online:
            rel_type = _EDGE_TYPE_TO_REL.get(
                str(edge.edge_type), str(edge.edge_type).upper()
            )
            try:
                self._conn.execute_write(
                    f"MATCH (s:Component {{id: $sid}}), (t:Component {{id: $tid}}) "
                    f"MERGE (s)-[r:{rel_type}]->(t) SET r.weight = $w",
                    {
                        "sid": edge.source_id,
                        "tid": edge.target_id,
                        "w": edge.weight,
                    },
                )
            except Exception:
                logger.exception(
                    "FalkorDB write failed for edge %s->%s",
                    edge.source_id,
                    edge.target_id,
                )

    # ------------------------------------------------------------------
    # Batch operations (for ingestion/migration)
    # ------------------------------------------------------------------

    def add_nodes_batch(self, nodes: list[GraphNode]) -> int:
        """Batch-insert nodes using UNWIND. Returns count of nodes written."""
        # NetworkX
        for node in nodes:
            self._graph.add_node(
                node.id,
                component_type=str(node.component_type),
                label=node.label,
                embedding_id=node.embedding_id,
            )

        if self._conn is None or not self._online:
            return len(nodes)

        written = 0
        for i in range(0, len(nodes), _BATCH_SIZE):
            batch = nodes[i : i + _BATCH_SIZE]
            batch_data = [
                {
                    "id": n.id,
                    "ct": str(n.component_type),
                    "label": n.label,
                    "eid": n.embedding_id,
                }
                for n in batch
            ]
            try:
                self._conn.execute_write(
                    "UNWIND $batch AS n "
                    "MERGE (c:Component {id: n.id}) "
                    "SET c.component_type = n.ct, c.label = n.label, "
                    "c.embedding_id = n.eid",
                    {"batch": batch_data},
                )
                written += len(batch)
            except Exception:
                logger.exception("FalkorDB batch node insert failed at offset %d", i)

        return written

    def add_edges_batch(self, edges: list[GraphEdge]) -> int:
        """Batch-insert edges grouped by type. Returns count of edges written."""
        # NetworkX
        for edge in edges:
            attrs: dict[str, str | float] = {
                "edge_type": str(edge.edge_type),
                "weight": edge.weight,
            }
            for k, v in edge.metadata.items():
                attrs[k] = v
            self._graph.add_edge(edge.source_id, edge.target_id, **attrs)

        if self._conn is None or not self._online:
            return len(edges)

        # Group edges by type for MERGE queries
        by_type: dict[str, list[GraphEdge]] = {}
        for edge in edges:
            rel = _EDGE_TYPE_TO_REL.get(
                str(edge.edge_type), str(edge.edge_type).upper()
            )
            by_type.setdefault(rel, []).append(edge)

        written = 0
        for rel_type, typed_edges in by_type.items():
            for i in range(0, len(typed_edges), _BATCH_SIZE):
                batch = typed_edges[i : i + _BATCH_SIZE]
                batch_data = [
                    {"sid": e.source_id, "tid": e.target_id, "w": e.weight}
                    for e in batch
                ]
                try:
                    self._conn.execute_write(
                        f"UNWIND $batch AS e "
                        f"MATCH (s:Component {{id: e.sid}}), (t:Component {{id: e.tid}}) "
                        f"MERGE (s)-[r:{rel_type}]->(t) SET r.weight = e.w",
                        {"batch": batch_data},
                    )
                    written += len(batch)
                except Exception:
                    logger.exception(
                        "FalkorDB batch edge insert failed for %s at offset %d",
                        rel_type,
                        i,
                    )

        return written

    # ------------------------------------------------------------------
    # Read operations (from NetworkX mirror)
    # ------------------------------------------------------------------

    def get_node(self, node_id: str) -> GraphNode | None:
        """Reconstruct a GraphNode from NetworkX mirror data."""
        if node_id not in self._graph:
            return None
        from skill_retriever.entities.components import ComponentType
        from skill_retriever.entities.graph import GraphNode as _GraphNode

        data: dict[str, Any] = dict(self._graph.nodes[node_id])
        if "component_type" not in data or "label" not in data:
            return None

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
        """Run PPR on the NetworkX mirror."""
        if self._graph.number_of_nodes() == 0 or not seed_ids:
            return []

        all_nodes: list[str] = list(self._graph.nodes)
        personalization: dict[str, float] = {n: 0.0 for n in all_nodes}
        valid_seeds = [s for s in seed_ids if s in self._graph]
        if not valid_seeds:
            return []
        seed_weight = 1.0 / len(valid_seeds)
        for sid in valid_seeds:
            personalization[sid] = seed_weight

        max_iter = 200 if alpha > 0.9 else 100
        scores: dict[str, float] = nx.pagerank(
            self._graph,
            alpha=alpha,
            personalization=personalization,
            weight="weight",
            max_iter=max_iter,
        )

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

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save graph to JSON (backup). FalkorDB data is already persisted."""
        data: dict[str, Any] = nx.node_link_data(self._graph)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        """Load graph: try FalkorDB sync first, fall back to JSON."""
        if self._conn is not None:
            try:
                self.sync_from_falkordb()
                if self._graph.number_of_nodes() > 0:
                    logger.info("Loaded graph from FalkorDB: %d nodes", self.node_count())
                    return
            except Exception:
                logger.exception("FalkorDB sync failed, falling back to JSON")

        # Fallback: load from JSON
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self._graph = nx.node_link_graph(data, directed=True)  # pyright: ignore[reportUnknownMemberType]
            logger.info("Loaded graph from JSON: %d nodes", self.node_count())
        except FileNotFoundError:
            logger.info("No graph file at %s, starting with empty graph", path)

    # ------------------------------------------------------------------
    # Protocol extensions (nx_graph, get_label_index, get_depends_on_subgraph)
    # ------------------------------------------------------------------

    @property
    def nx_graph(self) -> nx.DiGraph[str]:
        """Return the internal NetworkX graph for algorithms requiring direct access."""
        return self._graph

    def get_label_index(self) -> dict[str, list[str]]:
        """Build inverted index from lowercase labels to node IDs."""
        label_to_ids: dict[str, list[str]] = {}
        for node_id in self._graph.nodes:
            data: dict[str, Any] = dict(self._graph.nodes[node_id])
            label = str(data.get("label", "")).lower()
            if label:
                if label not in label_to_ids:
                    label_to_ids[label] = []
                label_to_ids[label].append(str(node_id))
        return label_to_ids

    def get_depends_on_subgraph(self) -> nx.DiGraph[str]:
        """Return a subgraph containing only DEPENDS_ON edges."""
        from skill_retriever.entities.graph import EdgeType

        depends_on_edges = [
            (u, v)
            for u, v, data in self._graph.edges(data=True)
            if data.get("edge_type") == str(EdgeType.DEPENDS_ON)
        ]
        return nx.DiGraph(depends_on_edges)

    def get_all_node_ids(self) -> set[str]:
        """Return the set of all node IDs in the graph."""
        return set(self._graph.nodes)

    # ------------------------------------------------------------------
    # Index creation (for FalkorDB)
    # ------------------------------------------------------------------

    def ensure_indexes(self) -> None:
        """Create FalkorDB indexes if they don't exist."""
        if self._conn is None or not self._online:
            return
        try:
            self._conn.execute_write(
                "CREATE INDEX FOR (c:Component) ON (c.id)"
            )
        except Exception:
            pass  # Index may already exist
        try:
            self._conn.execute_write(
                "CREATE INDEX FOR (c:Component) ON (c.component_type)"
            )
        except Exception:
            pass  # Index may already exist
        logger.info("FalkorDB indexes ensured")
