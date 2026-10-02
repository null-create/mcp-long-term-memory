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
  commit. Orphan entities are pruned in two tiers: `is_placeholder=true`
  entities (typo-fallout from `store_relationship` /
  `store_contradiction` / `store_claim` auto-create) are swept
  unconditionally; real orphans are only swept once their most recent
  timestamp is older than `max_age_days`. Setting `max_age_days <= 0`
  keeps real orphans forever. The per-tier breakdown is returned on
  `result["entity_breakdown"]`.

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
- **Full-text indexes are built alongside vector indexes.** Every
  Memory / entity / Claim / Document gets both a vector index and a
  `*_fulltext_idx` Lucene index. `recall`, `graph_find_entities`,
  `graph_find_claims`, and `graph_find_documents` fuse the two legs via
  Reciprocal Rank Fusion (k=60) — vector catches paraphrases, full-text
  catches exact terms (function names, version numbers, IDs). The fused
  path is on by default; set `MEMORY_RRF_FUSION=false` to A/B against
  the old vector-only behavior.
- **Cross-encoder rerank is opt-in.** Set `MEMORY_RERANK=true` to run
  the top-N fused candidates through
  `cross-encoder/ms-marco-MiniLM-L-6-v2` before returning. Costs an
  extra ~90MB on first load but materially improves ranking quality for
  ambiguous queries.
- **`source_sessions` is a JSON string, not a list.** Cypher `IN`
  doesn't work on it. Use `CONTAINS '"<id>"'` (with the JSON quotes).
- **`memory_get` is a direct-lookup bypass.** Use it when you have the
  UUID; use `memory_recall` / `memory_find_similar` for search.
- **Auto-created placeholder entities carry `is_placeholder: true`.**
  When `graph_store_relationship`, `graph_store_contradiction` (by
  name), or `graph_store_claim` is called with an entity name that
  doesn't exist yet, that endpoint is auto-created as a `Concept` with
  this flag so the edge is never silently dropped and you can tell "the
  agent reasoned about this" from "the agent typo'd this." The flag is
  cleared automatically the next time any explicit `graph_upsert_entity`
  touches the same name.
- **Every merge decision is logged.** `AsyncLongTermMemory.store`,
  `graph_upsert_entity`, and `graph_store_claim` dedup paths emit a
  structured `[MERGE] kind=... target_id=... score=... new="..."
  existing="..."` log line so you can reconstruct exactly which two
  candidates were merged, at what cosine score, and when. Grep logs for
  `[MERGE]` to audit.
- **Store-claim dedup now returns `success: True, merged: True`.**
  Previously it returned `success: False` on a near-duplicate; the new
  shape matches `memory_store` and `graph_upsert_entity` so agents
  don't treat legitimate dedup as an error.
- **`graph_store_claim` auto-flags contradictions.** When the new claim
  overlaps on entities with existing `supported` claims and the two
  disagree on negation polarity (cosine ≥ 0.78), a CONTRADICTS edge is
  auto-created and the pre-existing claim is demoted to `disputed`. The
  response payload carries a `contradictions: [...]` list when any
  were raised. Note: the claim dedup path has a polarity gate so that
  "X works" and "X doesn't work" — which embed at ~0.98 cosine, well
  above `_DEDUP_THRESHOLD=0.95` — are *not* deduped as a pair, giving
  the contradiction scan a chance to see them.
- **CONTRADICTS edges have two provenance shapes.** Pass-3 of
  `graph_prune` only sweeps relationship-backed ones (edges with
  non-null `rel_id_a` + `rel_id_b`, created via
  `graph_store_contradiction(rel_id_a=..., rel_id_b=...)`). Claim-backed
  edges (auto-detected or created via the `entity_a`/`entity_b` path)
  are immune — their backing facts are Claim nodes, cleaned up by
  Pass 4 or by cascading entity deletion. Earlier revisions pruned
  claim-backed edges every run; the current scoping keeps auto-detected
  contradictions sticky.
- **`valid_from` is not `created_at`.** `created_at` is the write time;
  `valid_from` is when the assertion is *semantically* active. If on
  2026-09-15 you discover the API rate limit *has been* 100 req/min
  since 2026-01-01, store the claim with `valid_from=2026-01-01` so
  `graph_claims_as_of("2026-06-01")` returns it. When omitted,
  `valid_from` defaults to the write time — matching pre-temporal
  behavior. `valid_until` is auto-populated by
  `graph_update_claim_status` on transitions to `retracted` /
  `disputed` (and cleared on transitions back to active), so the same
  `claims_as_of` query answers "what did we believe about X on date Y"
  correctly across supersessions.
- **`classify_relation` normalizes `_` and `-` to spaces.** The
  relation-type classifier (used by `graph_store_relationship` when
  `relationship_label` isn't supplied) accepts both English phrases
  (`"depends on"`, `"authored by"`) and SCREAMING_SNAKE_CASE
  (`"DEPENDS_ON"`, `"AUTHORED_BY"`) — they resolve to the same label.
  Keep passing whichever form reads naturally in context; the
  classifier is case-insensitive and underscore/hyphen agnostic. The
  controlled vocabulary is fixed (`_FACTUAL_REL_TYPES` — CAUSES,
  ENABLES, PREVENTS, REQUIRES, PART_OF, USES, PRODUCES, COMPETES_WITH,
  AFFILIATED_WITH, AUTHORED_BY, FUNDED_BY, PRECEDED_BY, OCCURRED_AT,
  ASSERTS, SUPPORTS, REFUTES, MENTIONS, RELATES_TO); anything not
  matching a pattern falls back to `RELATES_TO`. If you want a
  specific label and are unsure the phrase will route correctly, pass
  `relationship_label` explicitly to bypass classification.
- **MentalModels dedup by scope.** `graph_set_mental_model(question,
  answer, scope)` stores a canonical question + its cached answer. The
  dedup probe only looks within the same `scope` — "architecture?" in
  `project:A` and `project:B` are two independent entries. Within a
  scope, semantic dedup fires at the standard cosine
  `_DEDUP_THRESHOLD=0.95`, which is strict — paraphrases like `"What's"`
  vs `"What is"` collapse, but rephrases like `"How is X architected?"`
  vs `"What is X's architecture?"` typically embed around 0.69 and
  stay separate. If that's not what you want, delete the near-duplicate
  explicitly. **`find_mental_models` does NOT bump counters** — call
  `graph_touch_mental_model(id)` after you actually consume a cached
  answer so refresh triggers can see usage. `graph_get_mental_model`
  also doesn't bump counters (used by refresh triggers for observational
  reads).
- **MentalModel refresh is agent-driven.** The server never summarizes
  on its own. Lifecycle: (1) link cached answers to the entities they
  depend on via `graph_set_mental_model(..., entity_names=[...])`,
  which creates `(mm)-[:ABOUT]->(Entity)` edges; (2) call
  `graph_find_stale_mental_models(scope?, max_age_days?)` to discover
  stale entries — the scanner surfaces MMs when `stale=true`
  (`manual_flag`), when any ABOUT entity's own timestamps bumped after
  `last_refreshed` (`entity_touched: <name>`), when an adjacent
  non-ABOUT relationship was created/confirmed after `last_refreshed`
  (`adjacent_edge_touched: <name>`), or (fallback, only when the MM
  has no ABOUT edges) when `last_refreshed` exceeds `max_age_days`
  (`max_age_exceeded`); (3) for each stale MM, pull fresh context with
  `graph_recall_context(mm["question"])`, resummarize **client-side**,
  then write back via `graph_set_mental_model(..., entity_names=mm["about_entities"])`
  — the dedup path merges it in place and auto-clears the stale flag.
  Use `graph_mark_mental_model_stale(id)` when you know out-of-band
  that an answer is wrong but haven't produced a replacement yet.
  `entity_names` follows null-is-preserve on refresh: `None` keeps
  existing ABOUT edges, `[]` clears them, a list replaces them.
  Unresolved names are returned in `unresolved_entities` without
  failing the call.

## Tool inventory (46 tools)

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

**Mental models (cached answers):** `graph_set_mental_model`,
`graph_find_mental_models`, `graph_get_mental_model`,
`graph_list_mental_models`, `graph_delete_mental_model`,
`graph_touch_mental_model`, `graph_mark_mental_model_stale`,
`graph_find_stale_mental_models`.

**Graph analytics:** `graph_get_communities`, `graph_find_paths`,
`graph_find_common_neighbors`, `graph_recall_context`,
`graph_session_diff`.

**Temporal:** `graph_claims_as_of`, `graph_changed_between`,
`graph_entity_history`.

**Maintenance:** `graph_decay_confidence`, `graph_prune`.

Full parameter docs live in each tool's docstring in `server.py`. See
`README.md` for a shorter user-facing summary.
