# Phase 03: Memory Layer - Research

**Researched:** 2026-02-03
**Domain:** Graph storage (NetworkX), vector similarity (FAISS), component memory tracking, PPR computation
**Confidence:** HIGH

## Summary

This phase builds three memory subsystems: (1) a graph store wrapping NetworkX DiGraph with PPR via scipy/scikit-network, behind a Protocol-based abstraction layer, (2) a vector store wrapping FAISS IndexFlatIP with FastEmbed embedding generation, and (3) a component memory tracker for recommendation counts and co-selection patterns.

The existing entity models (GraphNode, GraphEdge, ComponentMetadata) from Phase 2 are consumed directly. The graph store receives nodes/edges from the ingestion pipeline and provides PPR-ranked retrieval. The vector store indexes component descriptions/content as 384-dim embeddings and returns top-k similar components. Component memory tracks how often components are recommended, selected, and which components are selected together.

All three libraries (NetworkX 3.6.1, faiss-cpu 1.13.2, scikit-network 0.33.5) have confirmed Python 3.13 support with pre-built wheels. For a 1300-node graph, NetworkX's built-in `pagerank()` with the `personalization` parameter is sufficient for PPR -- no need for external PPR libraries. FAISS IndexFlatIP (exact search) is the correct index type for 1300 vectors at 384 dimensions; approximate indexes (IVF, HNSW) only pay off above ~50k vectors. FastEmbed's `TextEmbedding.embed()` returns a generator of numpy arrays, supporting batch processing with configurable batch_size and parallelism.

**Primary recommendation:** Use NetworkX's built-in `nx.pagerank()` for PPR (avoid adding scikit-network or fast-pagerank as dependencies for this scale), FAISS IndexIDMap wrapping IndexFlatIP for vector search with string-to-int ID mapping, and a simple Pydantic model with JSON persistence for component memory.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| networkx | >=3.6.1 | Directed graph storage, PPR computation | Pure Python, Python 3.13 support, built-in `pagerank()` with personalization, `node_link_data` for JSON persistence |
| faiss-cpu | >=1.13.2 | Vector similarity search (cosine via normalized inner product) | Pre-built wheels for Python 3.13/Windows, `IndexFlatIP` exact search optimal for ~1300 vectors |
| fastembed | >=0.7.4 | Embedding generation (BAAI/bge-small-en-v1.5) | Already pinned in pyproject.toml, ONNX Runtime backend, no GPU required |
| scipy | (transitive) | Sparse matrix support for NetworkX internals | Pulled in by networkx/faiss-cpu, used internally for PPR power iteration |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | (transitive) | Array operations for FAISS vectors | Vector normalization, array construction for FAISS add/search |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| NetworkX `pagerank()` | scikit-network `PageRank` | scikit-network operates on raw scipy sparse matrices (faster for repeated calls), but adds a dependency for marginal gain at 1300 nodes. Reserve for optimization if PPR latency exceeds 200ms. |
| NetworkX `pagerank()` | fast-pagerank | Unmaintained (v0.0.4, no updates in 12+ months), Python 3.13 compatibility unknown. Avoid. |
| FAISS IndexFlatIP | FAISS IndexIVFFlat | IVF requires training step and only benefits above ~50k vectors. Flat search is exact and fast at 1300 vectors. |
| JSON persistence | pickle | Pickle is faster to serialize but not human-readable, not cross-language, and has security concerns. JSON via `node_link_data` is safer and debuggable. |

**Installation:**
```bash
uv add networkx faiss-cpu
```

Note: `fastembed` is already in pyproject.toml. `scipy` and `numpy` come as transitive dependencies.

## Architecture Patterns

### Recommended Project Structure
```
src/skill_retriever/
├── memory/
│   ├── __init__.py          # Public API exports
│   ├── graph_store.py       # Protocol + NetworkX implementation
│   ├── vector_store.py      # FAISS index + FastEmbed wrapper
│   └── component_memory.py  # Usage tracking, co-selection
```

### Pattern 1: Protocol-Based Graph Store Abstraction
**What:** Define a `GraphStore` Protocol that the NetworkX implementation satisfies. Future FalkorDB migration swaps the implementation without changing callers.
**When to use:** Always -- this is a locked requirement from prior decisions.
**Example:**
```python
from typing import Protocol, runtime_checkable

from skill_retriever.entities.graph import GraphEdge, GraphNode


@runtime_checkable
class GraphStore(Protocol):
    """Abstract interface for graph storage backends."""

    def add_node(self, node: GraphNode) -> None: ...
    def add_edge(self, edge: GraphEdge) -> None: ...
    def get_node(self, node_id: str) -> GraphNode | None: ...
    def get_neighbors(self, node_id: str) -> list[GraphNode]: ...
    def get_edges(self, node_id: str) -> list[GraphEdge]: ...
    def personalized_pagerank(
        self, seed_ids: list[str], alpha: float = 0.85, top_k: int = 10
    ) -> list[tuple[str, float]]: ...
    def node_count(self) -> int: ...
    def edge_count(self) -> int: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
```

### Pattern 2: FAISS with String ID Mapping
**What:** FAISS only supports int64 IDs. Maintain a bidirectional string-to-int mapping alongside the FAISS index.
**When to use:** Always -- component IDs are strings like `owner/repo/type/name`.
**Example:**
```python
import faiss
import numpy as np


class VectorStore:
    def __init__(self, dimensions: int = 384) -> None:
        base_index = faiss.IndexFlatIP(dimensions)
        self._index = faiss.IndexIDMap(base_index)
        self._id_to_int: dict[str, int] = {}
        self._int_to_id: dict[int, str] = {}
        self._next_int_id: int = 0

    def add(self, component_id: str, embedding: np.ndarray) -> None:
        int_id = self._next_int_id
        self._next_int_id += 1
        self._id_to_int[component_id] = int_id
        self._int_to_id[int_id] = component_id
        # Normalize for cosine similarity via inner product
        vec = embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(vec)
        self._index.add_with_ids(vec, np.array([int_id], dtype=np.int64))

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        vec = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(vec)
        distances, indices = self._index.search(vec, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:  # FAISS returns -1 for missing results
                results.append((self._int_to_id[int(idx)], float(dist)))
        return results
```

### Pattern 3: NetworkX PPR with Personalization Dict
**What:** Use NetworkX's built-in `nx.pagerank()` with the `personalization` parameter for PPR from seed nodes.
**When to use:** When ranking components by relevance to a seed component.
**Example:**
```python
import networkx as nx


def personalized_pagerank(
    graph: nx.DiGraph,
    seed_ids: list[str],
    alpha: float = 0.85,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Compute PPR from seed nodes, return top-k ranked node IDs."""
    if not seed_ids or graph.number_of_nodes() == 0:
        return []
    # Build personalization vector: uniform over seed nodes
    personalization = {node: 0.0 for node in graph.nodes()}
    weight = 1.0 / len(seed_ids)
    for seed in seed_ids:
        if seed in personalization:
            personalization[seed] = weight
    scores = nx.pagerank(graph, alpha=alpha, personalization=personalization, weight="weight")
    # Sort by score descending, exclude seed nodes
    ranked = sorted(
        ((node_id, score) for node_id, score in scores.items() if node_id not in seed_ids),
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked[:top_k]
```

### Pattern 4: JSON Persistence for NetworkX Graph
**What:** Use `nx.node_link_data()` / `nx.node_link_graph()` for human-readable JSON serialization.
**When to use:** Saving/loading the graph store to disk.
**Example:**
```python
import json
from pathlib import Path

import networkx as nx


def save_graph(graph: nx.DiGraph, path: Path) -> None:
    data = nx.node_link_data(graph)
    path.write_text(json.dumps(data, indent=2, default=str))


def load_graph(path: Path) -> nx.DiGraph:
    data = json.loads(path.read_text())
    return nx.node_link_graph(data, directed=True)
```

### Pattern 5: FastEmbed Batch Embedding
**What:** Use `TextEmbedding.embed()` generator for efficient batch embedding of component descriptions.
**When to use:** Initial indexing and re-indexing of components.
**Example:**
```python
from fastembed import TextEmbedding
import numpy as np


def generate_embeddings(
    texts: list[str],
    model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 256,
) -> list[np.ndarray]:
    """Generate embeddings for a list of texts."""
    model = TextEmbedding(model_name=model_name)
    # embed() returns a generator of numpy arrays
    embeddings = list(model.embed(texts, batch_size=batch_size))
    return embeddings
```

### Anti-Patterns to Avoid
- **Converting NetworkX graph to scipy sparse matrix for PPR at this scale:** The overhead of conversion negates any performance gain at 1300 nodes. Use `nx.pagerank()` directly.
- **Using FAISS IVF/HNSW indexes for small collections:** Training overhead and approximate results are not worth it below ~50k vectors. Use IndexFlatIP for exact search.
- **Storing embeddings in the NetworkX graph:** Keep graph store and vector store as separate subsystems. The `embedding_id` field on GraphNode links them.
- **Using pickle for graph persistence:** Not human-readable, not debuggable, security concerns. Use `node_link_data` JSON.
- **Creating FastEmbed model instance per embedding call:** Model loading is expensive. Create once and reuse across batch operations.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| PageRank / PPR | Custom power iteration | `nx.pagerank(personalization=...)` | Handles dangling nodes, convergence, edge weights correctly |
| Vector similarity | Custom cosine distance loop | `faiss.IndexFlatIP` with L2 normalization | SIMD-optimized, handles batch search, persistence built in |
| Embedding generation | Raw ONNX model loading | `FastEmbed TextEmbedding` | Handles tokenization, batching, model caching, prefix handling |
| Graph serialization | Custom JSON schema | `nx.node_link_data()` / `nx.node_link_graph()` | Proven round-trip, handles all node/edge attributes |
| L2 normalization | `vec / np.linalg.norm(vec)` | `faiss.normalize_L2(vec)` | In-place, handles batches, avoids zero-division |

**Key insight:** At 1300 nodes / 384 dimensions, all standard library implementations are fast enough. The abstraction layer matters more than the performance of individual operations. Optimize for swappability, not raw speed.

## Common Pitfalls

### Pitfall 1: FAISS Inner Product Without Normalization
**What goes wrong:** `IndexFlatIP` computes raw inner product, not cosine similarity. Un-normalized vectors give meaningless similarity scores.
**Why it happens:** FAISS does not auto-normalize. Developers assume IP equals cosine.
**How to avoid:** Always call `faiss.normalize_L2(vectors)` before `add` and `search` operations. This converts IP to cosine similarity.
**Warning signs:** Similarity scores outside [0, 1] range, or scores that don't correlate with semantic similarity.

### Pitfall 2: FAISS Returns -1 for Missing Results
**What goes wrong:** When fewer than `k` vectors exist, FAISS pads results with index=-1 and distance=0.
**Why it happens:** FAISS always returns exactly `k` results per query.
**How to avoid:** Filter results where `index == -1` before returning to callers.
**Warning signs:** KeyError when looking up -1 in ID mapping.

### Pitfall 3: NetworkX Personalization Vector Must Cover All Nodes
**What goes wrong:** If `personalization` dict doesn't contain all nodes, NetworkX raises an error or produces unexpected results.
**Why it happens:** The personalization vector defines restart probabilities for ALL nodes; missing nodes get implicitly treated differently depending on version.
**How to avoid:** Initialize personalization dict with 0.0 for all nodes, then set non-zero values for seeds only.
**Warning signs:** `NetworkXError` about personalization vector.

### Pitfall 4: FastEmbed Query vs Passage Prefixes
**What goes wrong:** BGE models expect different prefixes for queries vs documents. Embedding queries without "query: " prefix degrades retrieval quality.
**Why it happens:** BGE models were trained with asymmetric prefixes.
**How to avoid:** FastEmbed handles this automatically when using `embed()` for documents and `query_embed()` for queries (if available), but verify the model's expected prefix behavior. For `bge-small-en-v1.5`, short queries benefit from "Represent this sentence: " prefix.
**Warning signs:** Retrieval quality significantly worse than expected despite correct embeddings.

### Pitfall 5: Saving FAISS Index Without ID Mapping
**What goes wrong:** `faiss.write_index()` saves the index but not the string-to-int ID mapping. Loading the index later loses the ability to map results back to component IDs.
**Why it happens:** FAISS only persists vector data and integer IDs. The external mapping is the developer's responsibility.
**How to avoid:** Always save the ID mapping (as JSON) alongside the FAISS index file. Load both together.
**Warning signs:** After restart, search returns integer IDs that can't be resolved.

### Pitfall 6: NetworkX `node_link_graph` Directed Flag
**What goes wrong:** `node_link_graph()` defaults to creating an undirected graph even if the serialized data says `directed: True`.
**Why it happens:** Historical API behavior. The `directed` parameter in `node_link_graph()` may not read from the data dict automatically in all versions.
**How to avoid:** Always pass `directed=True` explicitly when loading: `nx.node_link_graph(data, directed=True)`.
**Warning signs:** Graph has double the expected edges (each directed edge becomes two undirected edges).

## Code Examples

### Complete Graph Store Implementation Skeleton
```python
# Source: NetworkX 3.6.1 docs + project entity models
import json
from pathlib import Path

import networkx as nx

from skill_retriever.entities.graph import EdgeType, GraphEdge, GraphNode


class NetworkXGraphStore:
    """NetworkX-backed graph store implementing GraphStore Protocol."""

    def __init__(self) -> None:
        self._graph = nx.DiGraph()

    def add_node(self, node: GraphNode) -> None:
        self._graph.add_node(
            node.id,
            component_type=node.component_type.value,
            label=node.label,
            embedding_id=node.embedding_id,
        )

    def add_edge(self, edge: GraphEdge) -> None:
        self._graph.add_edge(
            edge.source_id,
            edge.target_id,
            edge_type=edge.edge_type.value,
            weight=edge.weight,
            **edge.metadata,
        )

    def get_node(self, node_id: str) -> GraphNode | None:
        if node_id not in self._graph:
            return None
        data = self._graph.nodes[node_id]
        return GraphNode(
            id=node_id,
            component_type=data["component_type"],
            label=data["label"],
            embedding_id=data.get("embedding_id", ""),
        )

    def personalized_pagerank(
        self,
        seed_ids: list[str],
        alpha: float = 0.85,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        if not seed_ids or self._graph.number_of_nodes() == 0:
            return []
        personalization = {n: 0.0 for n in self._graph.nodes()}
        weight = 1.0 / len(seed_ids)
        for sid in seed_ids:
            if sid in personalization:
                personalization[sid] = weight
        scores = nx.pagerank(
            self._graph, alpha=alpha, personalization=personalization, weight="weight"
        )
        ranked = sorted(
            ((nid, sc) for nid, sc in scores.items() if nid not in seed_ids),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]

    def save(self, path: str) -> None:
        data = nx.node_link_data(self._graph)
        Path(path).write_text(json.dumps(data, indent=2, default=str))

    def load(self, path: str) -> None:
        data = json.loads(Path(path).read_text())
        self._graph = nx.node_link_graph(data, directed=True)

    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    def edge_count(self) -> int:
        return self._graph.number_of_edges()
```

### Complete Vector Store Persistence
```python
# Source: FAISS wiki + FastEmbed docs
import json
from pathlib import Path

import faiss
import numpy as np

from skill_retriever.config import EMBEDDING_CONFIG


class FAISSVectorStore:
    """FAISS-backed vector store with string ID mapping."""

    def __init__(self, dimensions: int | None = None) -> None:
        dim = dimensions or EMBEDDING_CONFIG.dimensions
        base = faiss.IndexFlatIP(dim)
        self._index = faiss.IndexIDMap(base)
        self._id_to_int: dict[str, int] = {}
        self._int_to_id: dict[int, str] = {}
        self._next_id: int = 0

    def add(self, component_id: str, embedding: np.ndarray) -> None:
        int_id = self._next_id
        self._next_id += 1
        self._id_to_int[component_id] = int_id
        self._int_to_id[int_id] = component_id
        vec = embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(vec)
        self._index.add_with_ids(vec, np.array([int_id], dtype=np.int64))

    def add_batch(self, ids: list[str], embeddings: np.ndarray) -> None:
        """Add multiple vectors at once."""
        int_ids = []
        for cid in ids:
            int_id = self._next_id
            self._next_id += 1
            self._id_to_int[cid] = int_id
            self._int_to_id[int_id] = cid
            int_ids.append(int_id)
        vecs = embeddings.astype(np.float32)
        faiss.normalize_L2(vecs)
        self._index.add_with_ids(vecs, np.array(int_ids, dtype=np.int64))

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        vec = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(vec)
        distances, indices = self._index.search(vec, top_k)
        return [
            (self._int_to_id[int(idx)], float(dist))
            for dist, idx in zip(distances[0], indices[0])
            if idx != -1
        ]

    def save(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / "vectors.faiss"))
        mapping = {
            "id_to_int": self._id_to_int,
            "int_to_id": {str(k): v for k, v in self._int_to_id.items()},
            "next_id": self._next_id,
        }
        (path / "id_mapping.json").write_text(json.dumps(mapping))

    def load(self, directory: str) -> None:
        path = Path(directory)
        self._index = faiss.read_index(str(path / "vectors.faiss"))
        mapping = json.loads((path / "id_mapping.json").read_text())
        self._id_to_int = mapping["id_to_int"]
        self._int_to_id = {int(k): v for k, v in mapping["int_to_id"].items()}
        self._next_id = mapping["next_id"]

    @property
    def count(self) -> int:
        return self._index.ntotal
```

### Component Memory Model
```python
# Source: DeepAgent-style tracking pattern
from datetime import datetime

from pydantic import BaseModel, Field


class ComponentUsageStats(BaseModel):
    """Tracks recommendation and selection counts for a single component."""

    component_id: str
    recommendation_count: int = 0
    selection_count: int = 0
    last_recommended: datetime | None = None
    last_selected: datetime | None = None


class CoSelectionEntry(BaseModel):
    """Tracks how often two components are selected together."""

    component_a: str
    component_b: str  # Always sorted: a < b
    count: int = 0


class ComponentMemory(BaseModel):
    """Persistent memory of component usage patterns."""

    usage_stats: dict[str, ComponentUsageStats] = Field(default_factory=dict)
    co_selections: dict[str, CoSelectionEntry] = Field(default_factory=dict)

    def record_recommendation(self, component_id: str) -> None:
        stats = self.usage_stats.setdefault(
            component_id, ComponentUsageStats(component_id=component_id)
        )
        stats.recommendation_count += 1
        stats.last_recommended = datetime.now()

    def record_selection(self, component_ids: list[str]) -> None:
        now = datetime.now()
        for cid in component_ids:
            stats = self.usage_stats.setdefault(
                cid, ComponentUsageStats(component_id=cid)
            )
            stats.selection_count += 1
            stats.last_selected = now
        # Track co-selections (all pairs)
        sorted_ids = sorted(component_ids)
        for i, a in enumerate(sorted_ids):
            for b in sorted_ids[i + 1:]:
                key = f"{a}|{b}"
                entry = self.co_selections.setdefault(
                    key, CoSelectionEntry(component_a=a, component_b=b)
                )
                entry.count += 1

    def get_co_selected(self, component_id: str, top_k: int = 5) -> list[tuple[str, int]]:
        """Return components most frequently co-selected with the given one."""
        pairs = []
        for entry in self.co_selections.values():
            if entry.component_a == component_id:
                pairs.append((entry.component_b, entry.count))
            elif entry.component_b == component_id:
                pairs.append((entry.component_a, entry.count))
        return sorted(pairs, key=lambda x: x[1], reverse=True)[:top_k]
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| NetworkX `write_gpickle` | `nx.node_link_data` + JSON | NX 3.0 (Jan 2023) | `write_gpickle` removed; use `json.dumps(nx.node_link_data(G))` |
| NetworkX `pagerank_scipy` | `nx.pagerank` (unified) | NX 3.0 | Separate scipy/numpy variants merged into single `pagerank()` |
| faiss 1.7.x | faiss-cpu 1.13.2 | Dec 2025 | Python 3.13 wheels, improved Windows support |
| FastEmbed <0.5 | FastEmbed 0.7.4+ | 2024 | Stable API, ONNX Runtime backend, batch_size parameter |

**Deprecated/outdated:**
- `nx.write_gpickle` / `nx.read_gpickle`: Removed in NetworkX 3.0. Use `pickle.dump(G, f)` directly or (preferred) `node_link_data` + JSON.
- `nx.pagerank_scipy` / `nx.pagerank_numpy`: Merged into unified `nx.pagerank()` in NetworkX 3.0.
- `fast-pagerank` package: Unmaintained (v0.0.4), do not use.

## Open Questions

1. **PPR alpha tuning validation pairs**
   - What we know: 20-30 query-component pairs needed (from prior decisions). These should be built during Phase 2 ingestion.
   - What's unclear: Whether Phase 2 actually produced these pairs, or if they need to be created as part of Phase 3.
   - Recommendation: Check Phase 2 summaries for validation data. If missing, create a small hand-curated set as a Phase 3 task.

2. **FastEmbed query prefix behavior for bge-small-en-v1.5**
   - What we know: BGE models benefit from "query: " or "Represent this sentence: " prefix for retrieval queries.
   - What's unclear: Whether FastEmbed's `embed()` method automatically adds the correct prefix, or if this must be done manually.
   - Recommendation: Test during implementation. If FastEmbed does not auto-prefix, add "Represent this sentence: " to query texts before embedding.

3. **Graph store `load()` reconstruction of Pydantic models**
   - What we know: `node_link_data` serializes node/edge attributes as plain dicts. On `load()`, these need to be reconstructed as `GraphNode`/`GraphEdge` Pydantic models.
   - What's unclear: Best pattern for lazy vs eager reconstruction.
   - Recommendation: Keep the internal representation as NetworkX native (dicts). Reconstruct Pydantic models on-demand in `get_node()` / `get_edges()` methods, not during load.

4. **ComponentMemory persistence format**
   - What we know: Pydantic v2 supports `model_dump_json()` / `model_validate_json()` for round-trip serialization.
   - What's unclear: Whether a single JSON file is sufficient or if growth warrants SQLite.
   - Recommendation: Start with single JSON file. At ~1300 components with co-selection pairs, the file will be well under 1MB. Migrate to SQLite only if needed.

## Sources

### Primary (HIGH confidence)
- [NetworkX 3.6.1 pagerank docs](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html) - PPR API, personalization parameter
- [NetworkX 3.6.1 node_link_data docs](https://networkx.org/documentation/stable/reference/readwrite/generated/networkx.readwrite.json_graph.node_link_data.html) - JSON serialization API
- [FAISS indexes wiki](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes) - IndexFlatIP, IndexIDMap usage
- [faiss-cpu 1.13.2 PyPI](https://pypi.org/project/faiss-cpu/) - Python 3.13 wheel availability confirmed
- [FastEmbed GitHub](https://github.com/qdrant/fastembed) - Batch embed API, model configuration
- [scikit-network 0.33.5 PyPI](https://pypi.org/project/scikit-network/) - Python 3.13 support confirmed (reserved as fallback)

### Secondary (MEDIUM confidence)
- [scikit-network PageRank tutorial](https://scikit-network.readthedocs.io/en/latest/tutorials/ranking/pagerank.html) - PPR with weights parameter, solver options
- [BAAI/bge-small-en-v1.5 HuggingFace](https://huggingface.co/BAAI/bge-small-en-v1.5) - Model dimensions, prefix behavior
- [NetworkX PyPI](https://pypi.org/project/networkx/) - Version 3.6.1, Python 3.13 classifier

### Tertiary (LOW confidence)
- [fast-pagerank GitHub](https://github.com/asajadi/fast-pagerank) - Performance claims vs NetworkX (unmaintained, not recommended)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries verified on PyPI with Python 3.13 wheels, APIs confirmed via official docs
- Architecture: HIGH - Protocol pattern is standard Python, NetworkX/FAISS APIs verified, entity models already defined
- Pitfalls: HIGH - Well-documented issues (FAISS normalization, ID mapping, NX personalization vector)
- PPR performance at 1300 nodes: MEDIUM - No specific benchmark found, but NetworkX handles graphs orders of magnitude larger; 200ms target is very achievable

**Research date:** 2026-02-03
**Valid until:** 2026-03-03 (stable libraries, slow-moving domain)
