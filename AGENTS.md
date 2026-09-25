# AGENTS.md

Read this file first when picking up work in this repo.

## What this is

A **long-term memory MCP server** for AI agents. It exposes 35+ MCP tools
that let an agent persist and recall knowledge across sessions using two
complementary layers on top of Neo4j:

1. **Flat memory layer** — `Memory` nodes with content, category, tags,
   and vector embeddings. Good for raw notes and quick semantic recall.
2. **Knowledge graph layer** — typed entities (Person, Organization,
   Technology, ...) with typed relationships (WORKS_FOR, USES, ...),
   IS_A hierarchies, communities, claims, and source documents.

Both layers are searchable by embedding-based vector similarity. Tools
mix and match freely — e.g. `knowledge_check` runs one probe across
every layer at once.

## Repo layout

```
mcp-memory-server/
├── server.py                    # MCP tool wrappers (all @mcp.tool() decorators)
├── tools/
│   ├── long_term_memory.py      # AsyncLongTermMemory + KnowledgeGraph classes
│   └── embeddings.py            # sentence-transformers wrapper
├── docker-compose.yml           # mcp-server + neo4j services
├── Dockerfile                   # python:3.12-slim, bakes in embedding model
├── Makefile                     # dev / run / stop / restart / clean targets
├── requirements.txt
└── data/                        # Neo4j volume (git-ignored)
```

## Architecture

- **`server.py`** is thin. It instantiates one `AsyncLongTermMemory` at
  import and every `@mcp.tool()` function just forwards to a method on
  `ltm` or `ltm.graph`. Add a new tool by adding a wrapper here and the
  underlying method in `tools/long_term_memory.py`.
- **`tools/long_term_memory.py`** owns all Cypher and business logic.
  Two classes:
  - `AsyncLongTermMemory` — flat `Memory` node CRUD, vector recall,
    dedup, `knowledge_check`, `list_categories`, `recall_project`.
  - `KnowledgeGraph` — typed entities, relationships, hierarchies,
    claims, documents, contradictions, community detection, prune /
    decay, bulk ingest, formatted context recall.
  - `AsyncLongTermMemory` exposes `.graph` as its `KnowledgeGraph`
    instance so callers can bounce between layers cheaply.
- **`tools/embeddings.py`** wraps sentence-transformers (`all-MiniLM-L6-v2`,
  384-dim). Exports `embed_texts()`, `EMBEDDINGS_ENABLED`, `EmbeddingError`.
  The model is baked into the Docker image so cold starts don't hit HF.
- **Neo4j 5.26** with APOC. Vector indexes exist per entity type + on
  `Memory.embedding`, `Claim.embedding`, `Document.embedding`. Union
  queries across the entity vector indexes are built by
  `KnowledgeGraph._build_union_vector_search`.

## Running

Preferred flow is Docker Compose — Neo4j + the MCP server come up together:

```bash
make run       # docker-compose up -d --build
make restart   # down + up --build (use this after code changes)
make stop      # docker-compose down
```

For iterating on `long_term_memory.py` outside Docker:

```bash
make dev       # PYTHONPATH=.:./tools python server.py
```

The `PYTHONPATH` matters because `long_term_memory.py` uses a bare
`from embeddings import ...`. The Dockerfile sets
`PYTHONPATH=/app:/app/tools` for the same reason.

Neo4j credentials come from `.env` (`NEO4J_PASSWORD=...`, default
`research_pass`). The MCP server listens on port `4398`; Neo4j's browser
is on `7474` and Bolt is on `7687`.

## Conventions

### Project scoping

Store project-specific memories with `category="project:<name>"`. This
unlocks:

- `memory_recall_project("<name>")` — every memory in that scope.
- `memory_list_categories()` — every distinct category (browse the
  `project:*` prefix to see what the agent knows about).
- `knowledge_check("<name>")` — reports matching project scopes so you
  can tell at a glance whether prior work exists.

### Return-value shape

Write methods return `{"success": bool, ...}`. Near-duplicate detection
on `memory_store` / `graph_upsert_entity` / `graph_store_relationship`
returns `success: True, merged: True` (**not** `success: False`) so
agents don't treat legitimate dedup as an error. Real failures still
return `success: False` with a `message`.

### Session IDs

`session_id` is optional everywhere. When supplied it's stored on
entities (as JSON-serialized `source_sessions`) and relationships, and
enables `graph_session_diff("<id>")` to retrieve everything a session
contributed. Because `source_sessions` is a JSON *string* (not a Neo4j
list), matching against it must wrap the needle in quotes — see
`session_diff` for the pattern.

## Recommended flows for agents

### Cold start of a session

1. `knowledge_check(topic)` — cheap one-shot probe across memories,
   entities, claims, and documents. Returns a verdict, top score,
   per-layer counts, matching `project:*` categories, and a summary.
2. If `known=True`, follow up with `graph_recall_context(topic)` for a
   formatted paragraph. It now folds in flat `Memory` nodes by default
   (`include_memories=True`), so cold graphs still surface something.
3. If `known=False`, start fresh.

### Persisting new findings

- Small note or fact → `memory_store(content, category="project:<name>",
  importance=..., tags=...)`.
- Structured knowledge (people, orgs, tech, relationships) →
  `graph_bulk_ingest(entities=[...], relationships=[...],
  hierarchies=[...])` in one call.
- Source URLs → `graph_store_document(url, title, ...)` then
  `graph_link_document_to_entity(...)`.
- Conflicting facts → `graph_store_contradiction(entity_a=..., entity_b=...,
  explanation=...)` — the entity-name form is fine when you don't have
  relationship IDs.

### Maintenance

- `graph_decay_confidence(half_life_days=30)` on a cadence to age out
  unused edges.
- `graph_prune(dry_run=True, sample_size=5)` to see what would be
  deleted (returns example items per category); flip `dry_run=False` to
  commit.

## When editing the server

- **Adding a tool:** define the async method on `AsyncLongTermMemory`
  or `KnowledgeGraph`, then add a thin `@mcp.tool()` wrapper in
  `server.py`. Keep the wrapper's docstring in sync — it becomes the
  tool description shown to callers.
- **Cypher changes:** run `python3 -m py_compile` on the file to catch
  quoting issues (f-string interpolation into Cypher parameter maps
  bites in this codebase). Actual query correctness requires a running
  Neo4j; use `make run` and hit tools through an MCP client.
- **Adding a new entity type:** append to the `ENTITY_TYPES` tuple in
  `long_term_memory.py`. The vector index bootstrap in
  `_ensure_schema()` picks it up automatically.
- **Adding a new relationship type:** append to `_FACTUAL_REL_TYPES` if
  the edge should count as a "fact" (participates in decay, prune,
  contradiction traversal). Non-factual edges (IS_A, MEMBER_OF,
  SOURCED_FROM, CONTRADICTS, ASSERTS) are handled separately.

## Common gotchas

- **Tool changes need a rebuild.** `make restart` — a plain restart
  won't reload `server.py` because it's baked into the image.
- **Embedding failures are non-fatal.** If HF fails and
  `EMBEDDINGS_ENABLED=False`, tools still work but semantic search
  degrades to substring / no-op paths. Check `EMBEDDINGS_ENABLED` if a
  recall query returns nothing unexpectedly.
- **Neo4j vector indexes are async to build.** After first startup,
  give the DB a few seconds before hammering it with queries. Look for
  `[KnowledgeGraph] Vector index ready` in logs.
- **`source_sessions` is a JSON string, not a list.** Cypher `IN`
  doesn't work on it. Use `CONTAINS '"<id>"'` (with the JSON quotes).
- **`memory_get` is a direct-lookup bypass.** Use it when you have the
  UUID; use `memory_recall` / `memory_find_similar` for search.

## Tool inventory (35 tools)

**Memory layer:** `memory_store`, `memory_get`, `memory_update`,
`memory_delete`, `memory_find_similar`, `memory_recall`,
`memory_list_categories`, `memory_recall_project`, `memory_stats`,
`knowledge_check`.

**Entities:** `graph_upsert_entity`, `graph_delete_entity`,
`graph_find_entities`, `graph_recent_entities`.

**Relationships:** `graph_store_relationship`, `graph_get_relationships`,
`graph_store_hierarchy`, `graph_recent_relationships`,
`graph_bulk_ingest`.

**Contradictions:** `graph_store_contradiction`,
`graph_find_contradictions`.

**Claims:** `graph_store_claim`, `graph_find_claims`,
`graph_update_claim_status`.

**Documents / provenance:** `graph_store_document`,
`graph_find_documents`, `graph_link_document_to_entity`,
`graph_get_provenance`.

**Graph analytics:** `graph_get_communities`, `graph_find_paths`,
`graph_find_common_neighbors`, `graph_recall_context`,
`graph_session_diff`.

**Maintenance:** `graph_decay_confidence`, `graph_prune`.

Full parameter docs live in each tool's docstring in `server.py`. See
`README.md` for a shorter user-facing summary.
