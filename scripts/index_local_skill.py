"""
Index a local skill into the skill-retriever data store.
Adds to metadata.json, FAISS vectors, and id_mapping.

Usage:
    uv run python scripts/index_local_skill.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import faiss
from fastembed import TextEmbedding

# --- Config ---
SKILL_DIR = Path.home() / "repos" / "skill-retriever" / "skills" / "science-plots"
DATA_DIR = Path.home() / ".skill-retriever" / "data"
COMPONENT_ID = "AnthonyAlcaraz/claude-skills/skill/science-plots"

def parse_frontmatter(text):
    """Extract YAML frontmatter from markdown."""
    lines = text.strip().split("\n")
    if lines[0].strip() != "---":
        return {}, text
    end = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end == -1:
        return {}, text
    fm = {}
    for line in lines[1:end]:
        if ":" in line:
            key, val = line.split(":", 1)
            fm[key.strip()] = val.strip().strip('"').strip("'")
    body = "\n".join(lines[end + 1:]).strip()
    return fm, body

def main():
    # 1. Read skill content
    skill_path = SKILL_DIR / "SKILL.md"
    if not skill_path.exists():
        print(f"ERROR: {skill_path} not found")
        sys.exit(1)

    raw = skill_path.read_text(encoding="utf-8")
    fm, body = parse_frontmatter(raw)

    # 2. Build metadata entry
    entry = {
        "id": COMPONENT_ID,
        "name": "science-plots",
        "component_type": "skill",
        "description": fm.get("description", "Create publication-quality scientific figures using SciencePlots + matplotlib"),
        "tags": ["matplotlib", "scientific-plotting", "ieee", "nature", "latex", "data-visualization"],
        "author": fm.get("author", "AnthonyAlcaraz"),
        "version": fm.get("version", "1.0"),
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "commit_count": 1,
        "commit_frequency_30d": 0.0,
        "raw_content": raw,
        "parameters": {},
        "dependencies": [],
        "tools": [],
        "source_repo": "AnthonyAlcaraz/claude-skills",
        "source_path": "skills/science-plots/SKILL.md",
        "category": "data-visualization",
        "install_url": None,
        "security_risk_level": "safe",
        "security_risk_score": 0.0,
        "security_findings_count": 0,
        "has_scripts": True
    }

    # 3. Add to metadata.json
    meta_path = DATA_DIR / "metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    # Remove existing entry if present (idempotent)
    metadata = [m for m in metadata if m["id"] != COMPONENT_ID]
    metadata.append(entry)
    meta_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print(f"Metadata: added ({len(metadata)} total components)")

    # 4. Generate embedding
    print("Generating embedding with BAAI/bge-small-en-v1.5...")
    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    embeddings = list(model.embed([raw]))
    vector = np.array(embeddings[0], dtype=np.float32).reshape(1, -1)
    print(f"Embedding shape: {vector.shape}")

    # 5. Add to FAISS index
    index_path = DATA_DIR / "vectors" / "faiss_index.bin"
    mapping_path = DATA_DIR / "vectors" / "id_mapping.json"

    index = faiss.read_index(str(index_path))
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))

    id_to_int = mapping["id_to_int"]
    int_to_id = mapping["int_to_id"]

    # Determine next integer ID
    if id_to_int:
        next_int = max(id_to_int.values()) + 1
    else:
        next_int = 0

    # Remove existing if present (rebuild needed for true removal, but we can just skip re-adding)
    if COMPONENT_ID in id_to_int:
        print(f"Already in FAISS at index {id_to_int[COMPONENT_ID]}, updating mapping only")
    else:
        # IndexIDMap requires add_with_ids
        ids = np.array([next_int], dtype=np.int64)
        index.add_with_ids(vector, ids)
        id_to_int[COMPONENT_ID] = next_int
        int_to_id[str(next_int)] = COMPONENT_ID
        print(f"FAISS: added at index {next_int} (total vectors: {index.ntotal})")

    faiss.write_index(index, str(index_path))
    mapping_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    print("Vectors saved")

    # 6. Add to graph.json
    graph_path = DATA_DIR / "graph.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))

    # Add node if not present
    existing_ids = {n["id"] for n in graph.get("nodes", [])}
    if COMPONENT_ID not in existing_ids:
        graph["nodes"].append({
            "id": COMPONENT_ID,
            "label": "science-plots",
            "component_type": "skill"
        })
        print(f"Graph: added node ({len(graph['nodes'])} total)")
    else:
        print("Graph: node already exists")

    graph_path.write_text(json.dumps(graph, indent=2, default=str), encoding="utf-8")

    print(f"\nDone! '{COMPONENT_ID}' indexed into skill-retriever.")

if __name__ == "__main__":
    main()
