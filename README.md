# Neo4j Long-Term Memory MCP Server

An MCP server that provides persistent cross-session memory and a knowledge graph (GraphRAG) backed by Neo4j.

## Features

### Memory Store
- Store, search, and recall text memories with semantic similarity via vector indexes
- Tagging, category filtering, importance ranking
- Deduplication by embedding similarity

### Hybrid Retrieval
- Vector + BM25 (Lucene) full-text search fused via Reciprocal Rank Fusion
- Optional cross-encoder reranking (opt-in, ~90MB model)
- Catches exact-term matches (function names, IDs, versions) that pure semantic search blurs

### Knowledge Graph
- Typed entity nodes: Person, Organization, Technology, Concept, Event, Location, Metric
- Typed relationships: CAUSES, ENABLES, PREVENTS, REQUIRES, USES, PRODUCES, COMPETES_WITH, etc.
- Claim tracking with confidence scoring and status (supported/disputed/unverified/retracted)
- Temporal claim validity (`valid_from` / `valid_until`) for point-in-time queries
- Automatic contradiction detection when new claims conflict with existing supported ones
- Document management with provenance tracking
- Community detection and summarization
- Contradiction detection between relationships
- Hierarchy support (IS_A edges)
- Path finding and common neighbor discovery
- Confidence decay and graph pruning
- Session-level diffs

### Mental Models (Cached Answers)
- Precomputed answers to recurring questions, scoped per project
- Linked to the entities they depend on via `ABOUT` edges
- Deterministic staleness detection — manual flag, entity touched, adjacent edge touched, or max-age fallback
- Agent-driven refresh loop (server stays LLM-free)

### Observability
- Structured merge-audit log lines (`[MERGE] kind=... score=... …`) for every entity/claim/memory dedup
- Fail-soft design — graph layer no-ops cleanly when Neo4j or embeddings are unavailable

## Prerequisites

- Python 3.10+
- Neo4j 5.11+ (required for vector index support)

## Configuration

| Env Variable | Default | Description |
|---|---|---|
| `HOST_PORT` | `4398` | MCP server port |
| `HOST_ADDRESS` | `0.0.0.0` | MCP server bind address |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `research_pass` | Neo4j password |
| `NEO4J_DATABASE` | `neo4j` | Neo4j database name |
| `EMBEDDING_DIMENSIONS` | `384` | Vector embedding dimensions |
| `DEFAULT_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model name |
| `EMBEDDINGS_ENABLED` | `true` | Set `false` to disable embedding generation |
| `MEMORY_RRF_FUSION` | `true` | Fuse vector + BM25 search via RRF (set `false` for vector-only) |
| `MEMORY_RETRIEVAL_OVERFETCH` | `4` | Over-fetch multiplier for fused retrieval before rerank/trim |
| `MEMORY_RERANK` | `false` | Enable cross-encoder reranking (downloads ~90MB on first run) |
| `DEFAULT_CROSS_ENCODER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model name |
| `MAX_EMBEDDING_WORKERS` | `4` | Thread pool size for embedding + rerank |
| `HF_TOKEN` | _(unset)_ | Hugging Face token for gated model downloads |

## MCP Tools

All public methods from `AsyncLongTermMemory` and `KnowledgeGraph` are exposed as MCP tools (46 total):

### Memory
| Tool | Description |
|---|---|
| `memory_store` | Store a new memory (returns `merged: True` on near-duplicate) |
| `memory_get` | Direct-lookup by ID (bypasses semantic search) |
| `memory_update` | Update fields of an existing memory in-place |
| `memory_delete` | Delete a memory by ID |
| `memory_find_similar` | Find semantically similar memories |
| `memory_recall` | Recall memories with filters |
| `memory_list_categories` | List every distinct category with counts (table of contents) |
| `memory_recall_project` | All memories in the `project:<name>` scope |
| `memory_stats` | Get memory + graph element counts |
| `knowledge_check` | One-shot "do I already know about this?" probe across all layers |

### Entities
| Tool | Description |
|---|---|
| `graph_upsert_entity` | Insert or update a typed entity |
| `graph_delete_entity` | Delete an entity and all its relationships |
| `graph_find_entities` | Semantic entity search |

### Relationships
| Tool | Description |
|---|---|
| `graph_store_relationship` | Store a typed relationship |
| `graph_bulk_ingest` | Bulk create entities + relationships + hierarchies in one call |
| `graph_get_relationships` | Multi-hop relationship traversal |
| `graph_store_hierarchy` | Create IS_A edge |
| `graph_store_contradiction` | Flag conflicting facts (by rel-IDs *or* entity names) |
| `graph_find_contradictions` | Find contradiction edges |

### Claims
| Tool | Description |
|---|---|
| `graph_store_claim` | Store a claim with entity links |
| `graph_find_claims` | Semantic claim search |
| `graph_update_claim_status` | Change claim status |

### Documents
| Tool | Description |
|---|---|
| `graph_store_document` | Store/update a document node |
| `graph_find_documents` | Semantic document search |
| `graph_link_document_to_entity` | Link document to entity |
| `graph_get_provenance` | Get source documents for entities |

### Mental Models (Cached Answers)
| Tool | Description |
|---|---|
| `graph_set_mental_model` | Store/refresh a cached answer (optionally link to entities via ABOUT edges) |
| `graph_find_mental_models` | Semantic search over cached answers (does not bump counters) |
| `graph_get_mental_model` | Direct-ID lookup (does not bump counters) |
| `graph_list_mental_models` | List cached answers in a scope, newest first |
| `graph_touch_mental_model` | Record a cache-hit (bumps `last_accessed` + `access_count`) |
| `graph_delete_mental_model` | Delete a cached answer |
| `graph_mark_mental_model_stale` | Manually flag a cached answer as stale |
| `graph_find_stale_mental_models` | Find cached answers needing refresh, with per-row `reasons` |

### Communities
| Tool | Description |
|---|---|
| `graph_get_communities` | Get community summaries |

### Recency & Temporal
| Tool | Description |
|---|---|
| `graph_recent_entities` | Recently confirmed entities |
| `graph_recent_relationships` | Recently created/confirmed edges |
| `graph_entity_history` | Chronological changelog for a single entity |
| `graph_changed_between` | Entities / relationships / claims changed in a date window |
| `graph_claims_as_of` | Claims that were semantically active on a given date (time-travel) |
| `graph_session_diff` | Changes from a specific session |

### Graph Analysis
| Tool | Description |
|---|---|
| `graph_find_paths` | Shortest path between entities |
| `graph_find_common_neighbors` | Entities connected to N named entities |
| `graph_recall_context` | Build structured context from graph |

### Maintenance
| Tool | Description |
|---|---|
| `graph_decay_confidence` | Exponentially decay edge confidence |
| `graph_prune` | Remove stale/low-confidence elements |

## Running

```bash
# Install dependencies
pip install -r requirements.txt

# Start the MCP server
python server.py
```

Local dev outside Docker requires `PYTHONPATH` to include `./tools`
(matching how the Dockerfile sets it):

```bash
make dev
# or
PYTHONPATH=.:./tools python server.py
```

## Project scoping convention

When storing knowledge about a specific project, use
`category="project:<name>"` on `memory_store`. This unlocks:

- **`memory_recall_project("<name>")`** — pulls every memory in that scope.
- **`memory_list_categories()`** — surfaces every `project:*` scope as a
  browsable table of contents.
- **`knowledge_check("<name>")`** — reports matching project scopes so
  you can tell at a glance whether the agent already knows about the
  project before starting work.

Recommended cold-start flow at the beginning of a session:

1. `knowledge_check(topic)` — cheap single-call probe.
2. If `known=True`: `graph_recall_context(topic)` for a formatted summary
   that now folds in flat memories automatically.
3. If `known=False`: proceed fresh, then persist findings with
   `memory_store(..., category="project:<name>")` and/or
   `graph_bulk_ingest(...)` at the end.

## Project Structure

```
├── server.py                     # MCP server entry point with tool wrappers
├── tools/
│   ├── long_term_memory.py       # Async Neo4j-backed memory + knowledge graph
│   └── embeddings.py             # SentenceTransformer + cross-encoder wrappers
├── requirements.txt
├── Makefile
├── AGENTS.md                     # Operator guide for agents using this server
└── README.md
```
