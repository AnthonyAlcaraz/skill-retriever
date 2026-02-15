"""
Batch-index all local skills in ~/repos/skill-retriever/skills/ into the skill-retriever data store.
Adds each to metadata.json, FAISS vectors, and graph.json.

Usage:
    cd ~/repos/skill-retriever && uv run python scripts/index_all_local_skills.py
"""

import json
import sys
import os
import re
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import faiss
from fastembed import TextEmbedding

# --- Config ---
SKILLS_DIR = Path.home() / "repos" / "skill-retriever" / "skills"
DATA_DIR = Path.home() / ".skill-retriever" / "data"
SOURCE_REPO = "AnthonyAlcaraz/os-agent-skills"
SKIP_ALREADY_INDEXED = True


def parse_frontmatter(text):
    """Extract YAML frontmatter from markdown."""
    lines = text.strip().split("\n")
    if not lines or lines[0].strip() != "---":
        return {}, text
    end = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end == -1:
        return {}, text
    fm = {}
    current_key = None
    current_list = None
    for line in lines[1:end]:
        stripped = line.strip()
        if stripped.startswith("- ") and current_key:
            if current_list is None:
                current_list = []
            current_list.append(stripped[2:].strip().strip('"').strip("'"))
            fm[current_key] = current_list
        elif ":" in stripped:
            if current_list is not None:
                current_list = None
            key, val = stripped.split(":", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            current_key = key
            if val:
                fm[key] = val
                current_list = None
            else:
                current_list = []
                fm[key] = current_list
        else:
            current_list = None
    body = "\n".join(lines[end + 1:]).strip()
    return fm, body


def build_tags(fm, skill_name):
    """Extract tags from frontmatter, handling list or string."""
    raw = fm.get("tags", [])
    if isinstance(raw, str):
        return [t.strip() for t in raw.split(",")]
    if isinstance(raw, list):
        return [str(t) for t in raw]
    return [skill_name]


def build_tools(fm):
    """Extract tools from frontmatter."""
    raw = fm.get("tools", [])
    if isinstance(raw, str):
        return [t.strip() for t in raw.split(",")]
    if isinstance(raw, list):
        return [str(t) for t in raw]
    return []


def main():
    # Discover all skills
    skill_dirs = sorted([
        d for d in SKILLS_DIR.iterdir()
        if d.is_dir() and (d / "SKILL.md").exists()
    ])

    if not skill_dirs:
        print("No skills found!")
        sys.exit(1)

    print(f"Found {len(skill_dirs)} skills to index")

    # Load existing stores
    meta_path = DATA_DIR / "metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    existing_ids = {m["id"] for m in metadata}
    print(f"Existing metadata entries: {len(metadata)}")

    index_path = DATA_DIR / "vectors" / "faiss_index.bin"
    mapping_path = DATA_DIR / "vectors" / "id_mapping.json"
    index = faiss.read_index(str(index_path))
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    id_to_int = mapping["id_to_int"]
    int_to_id = mapping["int_to_id"]

    graph_path = DATA_DIR / "graph.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    graph_ids = {n["id"] for n in graph.get("nodes", [])}

    # Initialize embedding model once
    print("Loading embedding model BAAI/bge-small-en-v1.5...")
    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    added = 0
    skipped = 0

    for skill_dir in skill_dirs:
        skill_name = skill_dir.name
        skill_path = skill_dir / "SKILL.md"
        component_id = f"{SOURCE_REPO}/skill/{skill_name}"

        if SKIP_ALREADY_INDEXED and component_id in existing_ids:
            print(f"  SKIP {skill_name} (already indexed)")
            skipped += 1
            continue

        print(f"\n  Indexing: {skill_name}")

        raw = skill_path.read_text(encoding="utf-8")
        fm, body = parse_frontmatter(raw)

        tags = build_tags(fm, skill_name)
        tools = build_tools(fm)

        # Build metadata entry
        entry = {
            "id": component_id,
            "name": fm.get("name", skill_name),
            "component_type": "skill",
            "description": fm.get("description", f"{skill_name} skill"),
            "tags": tags,
            "author": fm.get("author", ""),
            "version": fm.get("version", "1.0"),
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "commit_count": 1,
            "commit_frequency_30d": 0.0,
            "raw_content": raw,
            "parameters": {},
            "dependencies": [],
            "tools": tools,
            "source_repo": SOURCE_REPO,
            "source_path": f"skills/{skill_name}/SKILL.md",
            "category": "os-agent" if any(t in tags for t in [
                "computer-use", "os-agent", "gui-automation",
                "accessibility", "desktop-agent", "browser-automation"
            ]) else "automation",
            "install_url": None,
            "security_risk_level": "safe",
            "security_risk_score": 0.0,
            "security_findings_count": 0,
            "has_scripts": False
        }

        # Remove old entry if exists, add new
        metadata = [m for m in metadata if m["id"] != component_id]
        metadata.append(entry)

        # Generate embedding
        embeddings = list(model.embed([raw]))
        vector = np.array(embeddings[0], dtype=np.float32).reshape(1, -1)

        # Add to FAISS
        if component_id not in id_to_int:
            next_int = max(id_to_int.values()) + 1 if id_to_int else 0
            ids = np.array([next_int], dtype=np.int64)
            index.add_with_ids(vector, ids)
            id_to_int[component_id] = next_int
            int_to_id[str(next_int)] = component_id

        # Add to graph
        if component_id not in graph_ids:
            graph["nodes"].append({
                "id": component_id,
                "label": fm.get("name", skill_name),
                "component_type": "skill"
            })
            graph_ids.add(component_id)

        added += 1
        print(f"    OK ({len(raw)} chars, {len(tags)} tags)")

    # Save all stores
    print(f"\nSaving stores...")
    meta_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print(f"  Metadata: {len(metadata)} components")

    faiss.write_index(index, str(index_path))
    mapping["id_to_int"] = id_to_int
    mapping["int_to_id"] = int_to_id
    mapping_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    print(f"  FAISS: {index.ntotal} vectors")

    graph_path.write_text(json.dumps(graph, indent=2, default=str), encoding="utf-8")
    print(f"  Graph: {len(graph['nodes'])} nodes")

    print(f"\nDone! Added {added}, skipped {skipped}")


if __name__ == "__main__":
    main()
