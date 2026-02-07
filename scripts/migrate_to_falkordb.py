#!/usr/bin/env python3
"""One-time migration: load graph.json into FalkorDB.

Usage:
    uv run python scripts/migrate_to_falkordb.py [--host HOST] [--port PORT]

Reads ~/.skill-retriever/data/graph.json and bulk-inserts all nodes and edges
into FalkorDB graph ``skill_retriever``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Batch size for UNWIND operations
BATCH_SIZE = 200

# Edge type string -> FalkorDB relationship type
EDGE_TYPE_TO_REL: dict[str, str] = {
    "depends_on": "DEPENDS_ON",
    "enhances": "ENHANCES",
    "conflicts_with": "CONFLICTS_WITH",
    "bundles_with": "BUNDLES_WITH",
    "same_category": "SAME_CATEGORY",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate graph.json to FalkorDB")
    parser.add_argument("--host", default="localhost", help="FalkorDB host")
    parser.add_argument("--port", type=int, default=6379, help="FalkorDB port")
    parser.add_argument(
        "--graph-json",
        default=str(Path.home() / ".skill-retriever" / "data" / "graph.json"),
        help="Path to graph.json",
    )
    parser.add_argument("--graph-name", default="skill_retriever", help="FalkorDB graph name")
    args = parser.parse_args()

    # Load graph.json
    graph_path = Path(args.graph_json)
    if not graph_path.exists():
        print(f"ERROR: {graph_path} not found")
        sys.exit(1)

    print(f"Loading {graph_path}...")
    with open(graph_path, encoding="utf-8") as f:
        data = json.load(f)

    nodes = data.get("nodes", [])
    links = data.get("links", data.get("edges", []))
    print(f"  Found {len(nodes)} nodes, {len(links)} edges")

    # Connect to FalkorDB
    try:
        import falkordb  # type: ignore[import-untyped]
    except ImportError:
        print("ERROR: falkordb package not installed. Run: uv pip install falkordb")
        sys.exit(1)

    print(f"Connecting to FalkorDB at {args.host}:{args.port}...")
    db = falkordb.FalkorDB(host=args.host, port=args.port)
    graph = db.select_graph(args.graph_name)

    # Verify connection
    try:
        graph.query("RETURN 1")
    except Exception as e:
        print(f"ERROR: Cannot connect to FalkorDB: {e}")
        sys.exit(1)

    print(f"Connected to graph '{args.graph_name}'")

    # Create indexes
    print("Creating indexes...")
    for prop in ("id", "component_type"):
        try:
            graph.query(f"CREATE INDEX FOR (c:Component) ON (c.{prop})")
            print(f"  Created index on Component.{prop}")
        except Exception:
            print(f"  Index on Component.{prop} already exists")

    # Batch insert nodes
    start = time.perf_counter()
    print(f"Inserting {len(nodes)} nodes (batch size {BATCH_SIZE})...")
    node_count = 0
    for i in range(0, len(nodes), BATCH_SIZE):
        batch = nodes[i : i + BATCH_SIZE]
        batch_data = []
        for n in batch:
            node_id = str(n.get("id", ""))
            if not node_id:
                continue
            batch_data.append({
                "id": node_id,
                "ct": str(n.get("component_type", "")),
                "label": str(n.get("label", "")),
                "eid": str(n.get("embedding_id", "")),
            })

        if batch_data:
            graph.query(
                "UNWIND $batch AS n "
                "MERGE (c:Component {id: n.id}) "
                "SET c.component_type = n.ct, c.label = n.label, c.embedding_id = n.eid",
                {"batch": batch_data},
            )
            node_count += len(batch_data)
            print(f"  {node_count}/{len(nodes)} nodes inserted", end="\r")

    print(f"  {node_count} nodes inserted                ")

    # Batch insert edges grouped by type
    print(f"Inserting {len(links)} edges...")
    by_type: dict[str, list[dict[str, str | float]]] = {}
    for link in links:
        edge_type_str = str(link.get("edge_type", ""))
        rel_type = EDGE_TYPE_TO_REL.get(edge_type_str, edge_type_str.upper())
        if not rel_type:
            continue
        by_type.setdefault(rel_type, []).append({
            "sid": str(link.get("source", "")),
            "tid": str(link.get("target", "")),
            "w": float(link.get("weight", 1.0)),
        })

    edge_count = 0
    for rel_type, typed_edges in by_type.items():
        for i in range(0, len(typed_edges), BATCH_SIZE):
            batch = typed_edges[i : i + BATCH_SIZE]
            graph.query(
                f"UNWIND $batch AS e "
                f"MATCH (s:Component {{id: e.sid}}), (t:Component {{id: e.tid}}) "
                f"MERGE (s)-[r:{rel_type}]->(t) SET r.weight = e.w",
                {"batch": batch},
            )
            edge_count += len(batch)
        print(f"  {rel_type}: {len(typed_edges)} edges")

    elapsed = time.perf_counter() - start
    print(f"\nMigration complete in {elapsed:.1f}s")

    # Verify counts
    result = graph.query("MATCH (n:Component) RETURN count(n)")
    db_nodes = result.result_set[0][0]
    result = graph.query("MATCH ()-[r]->() RETURN count(r)")
    db_edges = result.result_set[0][0]

    print(f"\nVerification:")
    print(f"  JSON:     {len(nodes)} nodes, {len(links)} edges")
    print(f"  FalkorDB: {db_nodes} nodes, {db_edges} edges")

    if db_nodes == len(nodes):
        print("  Nodes: MATCH")
    else:
        print(f"  Nodes: MISMATCH (diff: {db_nodes - len(nodes)})")

    # Edge count may differ slightly due to missing source/target nodes
    if db_edges == len(links):
        print("  Edges: MATCH")
    elif db_edges < len(links):
        print(f"  Edges: {len(links) - db_edges} edges skipped (missing source/target nodes)")
    else:
        print(f"  Edges: MISMATCH (diff: {db_edges - len(links)})")


if __name__ == "__main__":
    main()
