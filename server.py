"""
Long term memory server using MCP and Neo4j.
This server provides a set of tools for storing, retrieving, and managing long-term memories, entities, relationships, claims, and documents in a Neo4j graph database.
It is designed to be used as a backend for AI agents that require persistent memory and knowledge graph capabilities.
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from mcp.server import MCPServer

load_dotenv()

from tools.long_term_memory import AsyncLongTermMemory

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Change as needed
HOST_PORT = int(os.getenv("HOST_PORT", "4398"))
HOST_ADDRESS = os.getenv("HOST_ADDRESS", "0.0.0.0")

# Neo4j configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "research_pass")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "384"))

# Initialize LongTermMemory (async_init is called in the lifespan)
ltm = AsyncLongTermMemory(
    neo4j_uri=NEO4J_URI,
    neo4j_user=NEO4J_USER,
    neo4j_password=NEO4J_PASSWORD,
    neo4j_database=NEO4J_DATABASE,
    embedding_dimensions=EMBEDDING_DIMENSIONS,
)


@asynccontextmanager
async def lifespan(app: MCPServer):
    """Startup: initialize Neo4j driver and schema.  Shutdown: close driver."""
    await ltm.async_init()
    try:
        yield
    finally:
        await ltm.close()


# Initialize the MCP server (transport params are passed to run())
mcp = MCPServer(name="Long term memory", lifespan=lifespan)

# -------------------------------------------------------------------
# Memory tools
# -------------------------------------------------------------------


@mcp.tool()
async def memory_delete(memory_id: str) -> Dict[str, Any]:
    """Delete a Memory node by its ID. Returns success status and deleted count."""
    return await ltm.delete(memory_id=memory_id)


@mcp.tool()
async def memory_get(memory_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a single Memory by its ID.

    Returns the memory dict (id, content, category, importance, created_at,
    last_accessed, access_count, tags, metadata) or ``None`` if not found.

    Use this to hydrate a Memory from a Document whose URL is
    ``memory://<memory_id>`` — parse the ID out of the URL, then call this.
    Semantic search (``memory_recall``/``memory_find_similar``) is unnecessary
    when the ID is already known.
    """
    return await ltm.get(memory_id=memory_id)


@mcp.tool()
async def memory_store(
    content: str,
    category: str = "general",
    importance: int = 5,
    tags: Optional[List[str]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Store a memory. Returns success status and memory_id."""
    return await ltm.store(
        content=content,
        category=category,
        importance=importance,
        tags=tags,
        extra_metadata=extra_metadata,
    )


@mcp.tool()
async def memory_find_similar(
    query: str,
    limit: int = 5,
    min_similarity: float = 0.7,
) -> List[Dict[str, Any]]:
    """Find memories semantically similar to the query."""
    return await ltm.find_similar(
        query=query, limit=limit, min_similarity=min_similarity
    )


@mcp.tool()
async def memory_recall(
    query: Optional[str] = None,
    category: Optional[str] = None,
    min_importance: Optional[int] = None,
    limit: int = 10,
    similarity_threshold: float = 0.0,
) -> List[Dict[str, Any]]:
    """Recall memories via semantic search, optionally filtered by category or importance."""
    return await ltm.recall(
        query=query,
        category=category,
        min_importance=min_importance,
        limit=limit,
        similarity_threshold=similarity_threshold,
    )


@mcp.tool()
async def memory_stats() -> Dict[str, int]:
    """Return counts of all graph elements (memories, entities, relationships, documents, claims, etc.)."""
    return await ltm.graph.stats()


@mcp.tool()
async def memory_update(
    memory_id: str,
    content: Optional[str] = None,
    category: Optional[str] = None,
    importance: Optional[int] = None,
    tags: Optional[List[str]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Update fields on an existing Memory in-place.

    Any parameter left as ``None`` is preserved. When ``content`` changes,
    the embedding is regenerated so semantic search reflects the new text.
    Preserves the memory ``id``, ``created_at``, and ``access_count`` —
    use this instead of delete + re-store to refine a note without losing
    its provenance.
    """
    return await ltm.update(
        memory_id=memory_id,
        content=content,
        category=category,
        importance=importance,
        tags=tags,
        extra_metadata=extra_metadata,
    )


@mcp.tool()
async def memory_list_categories() -> List[Dict[str, Any]]:
    """List every distinct memory category with counts and freshness.

    Returns a list of ``{category, count, latest, max_importance}`` dicts
    sorted by count descending. Use this as a "table of contents" for
    memory — especially to browse ``project:*`` categories to see which
    projects the agent already has knowledge about.
    """
    return await ltm.list_categories()


@mcp.tool()
async def memory_recall_project(
    project: str,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Return every Memory whose category matches ``project:<name>``.

    Formalizes the ``category="project:<name>"`` convention: pass the bare
    project name (with or without the ``project:`` prefix) and get back
    every memory in that scope, ordered by importance then recency.
    """
    return await ltm.recall_project(project=project, limit=limit)


@mcp.tool()
async def knowledge_check(
    topic: str,
    entity_limit: int = 5,
    memory_limit: int = 5,
    min_similarity: float = 0.4,
) -> Dict[str, Any]:
    """One-shot "do I already know about this?" probe.

    Fans a single query across memories, entities, claims, and documents
    concurrently, then returns a verdict:

    - ``known`` (bool): any signal cleared ``min_similarity``.
    - ``richness`` (str): none / low / medium / high.
    - ``top_score`` (float): best cosine similarity across all layers.
    - ``counts`` (dict): per-layer hit counts.
    - ``project_categories`` (list[str]): matching ``project:*`` scopes.
    - ``summary`` (str): a short human-readable one-liner.
    - ``entities`` / ``memories``: the top matches.

    Ideal as the very first call at the start of a session to decide
    whether to dive into the graph or start from scratch.
    """
    return await ltm.knowledge_check(
        topic=topic,
        entity_limit=entity_limit,
        memory_limit=memory_limit,
        min_similarity=min_similarity,
    )


# -------------------------------------------------------------------
# Entity tools
# -------------------------------------------------------------------


@mcp.tool()
async def graph_upsert_entity(
    name: str,
    entity_type: str = "concept",
    description: str = "",
    session_id: str = "",
    properties: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Insert or update a typed entity node. Deduplicates by semantic name similarity."""
    return await ltm.graph.upsert_entity(
        name=name,
        entity_type=entity_type,
        description=description,
        session_id=session_id,
        properties=properties,
    )


@mcp.tool()
async def graph_delete_entity(entity_id: str) -> Dict[str, Any]:
    """Delete an entity node and all its relationships by its ID. Returns success status and deleted count."""
    return await ltm.graph.delete_entity(entity_id=entity_id)


@mcp.tool()
async def graph_find_entities(
    query: str,
    limit: int = 5,
    include_hierarchy: bool = False,
    node_types: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Find entities semantically similar to the query. Optionally filter by node types."""
    return await ltm.graph.find_entities(
        query=query,
        limit=limit,
        include_hierarchy=include_hierarchy,
        node_types=node_types,
    )


# -------------------------------------------------------------------
# Relationship tools
# -------------------------------------------------------------------


@mcp.tool()
async def graph_store_relationship(
    source: str,
    target: str,
    relation: str,
    evidence: str = "",
    confidence: float = 0.8,
    session_id: str = "",
    step_id: int = 0,
    relationship_label: Optional[str] = None,
) -> Dict[str, Any]:
    """Store a directed typed relationship between two entities."""
    return await ltm.graph.store_relationship(
        source=source,
        target=target,
        relation=relation,
        evidence=evidence,
        confidence=confidence,
        session_id=session_id,
        step_id=step_id,
        relationship_label=relationship_label,
    )


@mcp.tool()
async def graph_get_relationships(
    entity_ids: Optional[List[str]] = None,
    entity_names: Optional[List[str]] = None,
    max_hops: int = 1,
) -> List[Dict[str, Any]]:
    """Get relationships involving the given entities, with optional multi-hop traversal."""
    return await ltm.graph.get_relationships(
        entity_ids=entity_ids,
        entity_names=entity_names,
        max_hops=max_hops,
    )


@mcp.tool()
async def graph_bulk_ingest(
    entities: Optional[List[Dict[str, Any]]] = None,
    relationships: Optional[List[Dict[str, Any]]] = None,
    hierarchies: Optional[List[Dict[str, Any]]] = None,
    session_id: str = "",
) -> Dict[str, Any]:
    """Ingest many entities, relationships, and IS_A hierarchies in one call.

    Item shapes:
      - ``entities``: dicts accepted by ``graph_upsert_entity`` — must have
        ``name``; optional ``entity_type``, ``description``, ``session_id``,
        ``properties``.
      - ``relationships``: dicts accepted by ``graph_store_relationship`` —
        must have ``source``, ``target``, ``relation``; optional
        ``evidence``, ``confidence``, ``session_id``, ``step_id``,
        ``relationship_label``.
      - ``hierarchies``: dicts with ``child_name`` and ``parent_name``.

    Returns a summary with per-kind ``created`` / ``merged`` / ``failed``
    counts and a compact ``errors`` list identifying any items that
    couldn't be stored. Vastly cheaper than issuing one MCP call per item
    when populating knowledge about a new project.
    """
    return await ltm.graph.bulk_ingest(
        entities=entities,
        relationships=relationships,
        hierarchies=hierarchies,
        session_id=session_id,
    )


@mcp.tool()
async def graph_store_hierarchy(
    child_name: str,
    parent_name: str,
) -> Dict[str, Any]:
    """Create an IS_A edge from child to parent (idempotent). Both entities must exist."""
    return await ltm.graph.store_hierarchy(
        child_name=child_name, parent_name=parent_name
    )


@mcp.tool()
async def graph_store_contradiction(
    rel_id_a: Optional[str] = None,
    rel_id_b: Optional[str] = None,
    explanation: str = "",
    session_id: str = "",
    entity_a: Optional[str] = None,
    entity_b: Optional[str] = None,
) -> Dict[str, Any]:
    """Record that two facts contradict each other.

    Two calling conventions:

    - **By relationship ID** — pass ``rel_id_a`` and ``rel_id_b``. Creates
      a CONTRADICTS edge between the two relationships' source entities.
    - **By entity name** — pass ``entity_a`` and ``entity_b``. Creates a
      CONTRADICTS edge directly between the two named entities (auto-
      creating either as a Concept if missing). Use this when you have a
      natural-language description of the conflict but no graph handles
      for the underlying facts.

    Exactly one convention must be fully specified.
    """
    return await ltm.graph.store_contradiction(
        rel_id_a=rel_id_a,
        rel_id_b=rel_id_b,
        explanation=explanation,
        session_id=session_id,
        entity_a=entity_a,
        entity_b=entity_b,
    )


@mcp.tool()
async def graph_find_contradictions(
    entity_names: Optional[List[str]] = None,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Return CONTRADICTS edges involving the given entities (or all if None)."""
    return await ltm.graph.find_contradictions(entity_names=entity_names, limit=limit)


# -------------------------------------------------------------------
# Claim tools
# -------------------------------------------------------------------


@mcp.tool()
async def graph_store_claim(
    text: str,
    confidence: float = 0.5,
    status: str = "unverified",
    source_session: str = "",
    step_id: int = 0,
    entity_names: Optional[List[str]] = None,
    document_id: Optional[str] = None,
    valid_from: Optional[str] = None,
) -> Dict[str, Any]:
    """Store a Claim node and link it to entities via ASSERTS edges.

    ``valid_from`` is the ISO-8601 timestamp when the claim's assertion
    is semantically active, distinct from the write time (``created_at``).
    Defaults to the write time. ``valid_until`` is auto-set by
    ``graph_update_claim_status`` on transitions to ``retracted`` or
    ``disputed`` so ``graph_claims_as_of`` can time-travel.
    """
    return await ltm.graph.store_claim(
        text=text,
        confidence=confidence,
        status=status,
        source_session=source_session,
        step_id=step_id,
        entity_names=entity_names,
        document_id=document_id,
        valid_from=valid_from,
    )


@mcp.tool()
async def graph_find_claims(
    query: str = "",
    limit: int = 10,
    entity_name: Optional[str] = None,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Find claims by semantic search, optionally filtered by entity or status."""
    return await ltm.graph.find_claims(
        query=query,
        limit=limit,
        entity_name=entity_name,
        status=status,
    )


@mcp.tool()
async def graph_update_claim_status(
    claim_id: str,
    new_status: str,
) -> Dict[str, Any]:
    """Update a claim's status (supported, disputed, unverified, retracted)."""
    return await ltm.graph.update_claim_status(claim_id=claim_id, new_status=new_status)


# -------------------------------------------------------------------
# Document tools
# -------------------------------------------------------------------


@mcp.tool()
async def graph_store_document(
    url: str,
    title: str = "",
    content_summary: str = "",
    doc_type: str = "article",
    credibility_score: float = 0.5,
    session_id: str = "",
) -> Dict[str, Any]:
    """Store or update a Document node (unique by URL)."""
    return await ltm.graph.store_document(
        url=url,
        title=title,
        content_summary=content_summary,
        doc_type=doc_type,
        credibility_score=credibility_score,
        session_id=session_id,
    )


@mcp.tool()
async def graph_find_documents(
    query: str = "",
    limit: int = 10,
    offset: int = 0,
    doc_type: Optional[str] = None,
    min_credibility: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Find documents by semantic search, optionally filtered.

    ``offset`` skips the first N results — useful for paging past the top
    hits when browsing many memory-backed documents.
    """
    return await ltm.graph.find_documents(
        query=query,
        limit=limit,
        offset=offset,
        doc_type=doc_type,
        min_credibility=min_credibility,
    )


@mcp.tool()
async def graph_link_document_to_entity(
    document_url: str,
    entity_name: str,
    relationship_label: str = "SOURCED_FROM",
) -> Dict[str, Any]:
    """Create a typed edge from an entity to a Document node.

    Stored direction is ``(entity)-[:LABEL]->(document)``.

    ``relationship_label`` must be one of:
    ``SOURCED_FROM`` (default; the label ``graph_get_provenance`` traverses),
    ``MENTIONS``, ``SUPPORTS``, ``REFUTES``, ``AUTHORED_BY``. Any other value
    fails with ``success=False`` — labels are not silently coerced.
    """
    return await ltm.graph.link_document_to_entity(
        document_url=document_url,
        entity_name=entity_name,
        relationship_label=relationship_label,
    )


@mcp.tool()
async def graph_get_provenance(
    entity_names: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Return Document nodes linked to the given entities via SOURCED_FROM."""
    return await ltm.graph.get_provenance(entity_names=entity_names)


# -------------------------------------------------------------------
# MentalModel tools (cached-answer / standing-question nodes)
# -------------------------------------------------------------------


@mcp.tool()
async def graph_set_mental_model(
    question: str,
    answer: str,
    scope: str = "global",
    source_session: str = "",
    entity_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Insert or refresh a MentalModel (cached-answer) node.

    A MentalModel stores a canonical ``question`` + its cached ``answer``
    for a given ``scope`` (recommend passing the bare project name, matching
    the ``category = "project:<name>"`` convention). Re-calling this with a
    semantically similar question *within the same scope* refreshes the
    existing node in place (updates ``answer``, bumps ``last_refreshed``,
    clears ``stale``) rather than creating a duplicate. Dedup threshold is
    the same cosine ``_DEDUP_THRESHOLD`` used by claims / memories.

    ``entity_names`` controls the ``(mm)-[:ABOUT]->(Entity)`` edges that
    power ``graph_find_stale_mental_models``' deterministic staleness
    detection:

    - ``None`` (default) — preserve existing ABOUT edges on refresh.
    - ``[]`` — explicitly clear all ABOUT edges.
    - ``["Name1", "Name2"]`` — replace ABOUT edges to match the list.
      Names that don't resolve to an existing ``:Entity`` are returned
      in ``unresolved_entities`` so the caller can see what was dropped
      without the whole call failing.
    """
    return await ltm.graph.set_mental_model(
        question=question,
        answer=answer,
        scope=scope,
        source_session=source_session,
        entity_names=entity_names,
    )


@mcp.tool()
async def graph_find_mental_models(
    query: str = "",
    scope: Optional[str] = None,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Semantic search for MentalModels, optionally scoped.

    Uses the same fused Lucene + vector + RRF retrieval as
    ``graph_find_claims`` / ``graph_find_documents``. When ``query`` is
    empty, returns the most recently refreshed MentalModels in the given
    scope. Does NOT bump usage counters — call ``graph_touch_mental_model``
    after you actually consume the cached answer.
    """
    return await ltm.graph.find_mental_models(
        query=query, scope=scope, limit=limit
    )


@mcp.tool()
async def graph_get_mental_model(
    mental_model_id: str,
) -> Optional[Dict[str, Any]]:
    """Direct-ID lookup for a MentalModel. Does NOT bump counters.

    Use this when Item 4's refresh trigger wants to inspect staleness
    without recording a "hit" against the cached answer. For
    answer-consumption lookups, prefer ``graph_find_mental_models``
    followed by ``graph_touch_mental_model``.
    """
    return await ltm.graph.get_mental_model(mental_model_id=mental_model_id)


@mcp.tool()
async def graph_list_mental_models(
    scope: Optional[str] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """List MentalModels (optionally scoped), ordered by last_refreshed DESC."""
    return await ltm.graph.list_mental_models(scope=scope, limit=limit)


@mcp.tool()
async def graph_delete_mental_model(
    mental_model_id: str,
) -> Dict[str, Any]:
    """Delete a MentalModel by id. Returns success status and deleted count."""
    return await ltm.graph.delete_mental_model(mental_model_id=mental_model_id)


@mcp.tool()
async def graph_touch_mental_model(
    mental_model_id: str,
) -> Dict[str, Any]:
    """Record a cache-hit on a MentalModel.

    Bumps ``last_accessed`` and increments ``access_count``. Separated
    from ``graph_find_mental_models`` / ``graph_get_mental_model`` so
    observational reads (e.g. staleness checks by Item 4's refresh
    trigger) do not poison usage statistics. Call this when an agent
    actually *uses* the cached answer in a response.
    """
    return await ltm.graph.touch_mental_model(mental_model_id=mental_model_id)


@mcp.tool()
async def graph_mark_mental_model_stale(
    mental_model_id: str,
) -> Dict[str, Any]:
    """Flag a MentalModel as stale so refresh scanners pick it up.

    Sets ``mm.stale = true``. On the next
    ``graph_find_stale_mental_models`` scan the flagged MM surfaces with
    reason ``manual_flag``. The flag is cleared automatically on the
    next ``graph_set_mental_model`` call for that question + scope
    (fresh insert or merge). Use when you have out-of-band knowledge
    that the cached answer is wrong but haven't yet produced a fresh
    one.
    """
    return await ltm.graph.mark_mental_model_stale(
        mental_model_id=mental_model_id,
    )


@mcp.tool()
async def graph_find_stale_mental_models(
    scope: Optional[str] = None,
    max_age_days: Optional[int] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Return MentalModels that look stale, with per-MM ``reasons``.

    Deterministic hybrid staleness detection. A MM qualifies as stale
    if ANY of:

    - ``mm.stale == true`` → reason ``manual_flag``.
    - An ``:ABOUT`` entity's own ``last_confirmed/last_seen/first_seen``
      is newer than ``mm.last_refreshed`` → ``entity_touched: <name>``.
    - An ``:ABOUT`` entity has an adjacent non-``:ABOUT`` relationship
      whose ``last_confirmed/created_at`` is newer than
      ``mm.last_refreshed`` → ``adjacent_edge_touched: <name>``.
    - ``max_age_days`` is set AND the MM has no ABOUT edges AND
      ``mm.last_refreshed`` is older than that many days →
      ``max_age_exceeded``.

    ``max_age_days`` is a **fallback only**: a MM with ABOUT edges
    trusts the entity signal and ignores age. Pass ``max_age_days=0``
    for an aggressive "any un-ABOUT model older than today" sweep.

    Results are sorted by ``last_refreshed ASC`` (oldest first) and
    each row includes ``about_entities`` so the agent can replay the
    links when it writes the refreshed answer back via
    ``graph_set_mental_model(..., entity_names=about_entities)``.
    """
    return await ltm.graph.find_stale_mental_models(
        scope=scope,
        max_age_days=max_age_days,
        limit=limit,
    )


# -------------------------------------------------------------------
# Community tools
# -------------------------------------------------------------------


@mcp.tool()
async def graph_get_communities(
    entity_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Retrieve community summaries relevant to the given entities."""
    return await ltm.graph.get_communities(entity_ids=entity_ids)


# -------------------------------------------------------------------
# Recency tools
# -------------------------------------------------------------------


@mcp.tool()
async def graph_recent_entities(
    since_date: Optional[str] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Return entities added or confirmed since the given ISO-8601 date."""
    return await ltm.graph.recent_entities(since_date=since_date, limit=limit)


@mcp.tool()
async def graph_recent_relationships(
    since_date: Optional[str] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Return factual edges created or confirmed since the given ISO-8601 date."""
    return await ltm.graph.recent_relationships(since_date=since_date, limit=limit)


@mcp.tool()
async def graph_entity_history(
    entity_name: str,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Return a chronological changelog of events touching ``entity_name``.

    Each event has ``kind`` (``entity_created``, ``claim_asserted``, or
    ``relationship``), a ``timestamp``, and kind-specific payload.
    Sorted ASC — oldest event first — so it reads top-to-bottom like a
    history log. Empty timestamps sort last.
    """
    return await ltm.graph.entity_history(entity_name=entity_name, limit=limit)


@mcp.tool()
async def graph_changed_between(
    start_date: str,
    end_date: str,
    kinds: Optional[List[str]] = None,
    limit: int = 50,
) -> Dict[str, List[Dict[str, Any]]]:
    """Return entities / relationships / claims that changed in ``[start, end]``.

    ``kinds`` is an optional subset of ``{"entities", "relationships",
    "claims"}`` — defaults to all three. Each bucket is capped at
    ``limit``. Fail-soft: a Cypher error on any single bucket yields
    an empty list for that bucket rather than failing the whole call.
    """
    return await ltm.graph.changed_between(
        start_date=start_date, end_date=end_date, kinds=kinds, limit=limit
    )


@mcp.tool()
async def graph_claims_as_of(
    as_of_date: str,
    entity_name: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Return claims that were semantically active on ``as_of_date``.

    A claim is active at date D when ``coalesce(valid_from, created_at)
    <= D`` AND ``(valid_until IS NULL OR valid_until > D)``. Supports
    time-travel queries like "what did we believe about X on 2026-09-15?"
    """
    return await ltm.graph.claims_as_of(
        as_of_date=as_of_date, entity_name=entity_name, limit=limit
    )


@mcp.tool()
async def graph_session_diff(
    session_id: str,
) -> Dict[str, List[Dict[str, Any]]]:
    """Return entities and relationships created during a specific session."""
    return await ltm.graph.session_diff(session_id=session_id)


# -------------------------------------------------------------------
# Graph path / neighbor tools
# -------------------------------------------------------------------


@mcp.tool()
async def graph_find_paths(
    source_name: str,
    target_name: str,
    max_depth: int = 4,
) -> List[Dict[str, Any]]:
    """Find shortest paths between two named entities."""
    return await ltm.graph.find_paths(
        source_name=source_name, target_name=target_name, max_depth=max_depth
    )


@mcp.tool()
async def graph_find_common_neighbors(
    entity_names: List[str],
    min_shared: int = 2,
) -> List[Dict[str, Any]]:
    """Return entities connected to at least min_shared of the named entities."""
    return await ltm.graph.find_common_neighbors(
        entity_names=entity_names, min_shared=min_shared
    )


# -------------------------------------------------------------------
# Graph-aware recall (the main context-building tool)
# -------------------------------------------------------------------


@mcp.tool()
async def graph_recall_context(
    query: str,
    entity_limit: int = 5,
    max_hops: int = 2,
    min_confidence: float = 0.0,
    include_contradictions: bool = False,
    include_provenance: bool = False,
    include_claims: bool = True,
    include_documents: bool = False,
    include_memories: bool = True,
    memory_limit: int = 5,
    node_types: Optional[List[str]] = None,
) -> str:
    """Build a structured text context from the knowledge graph for a query.

    Searches typed entities, traverses relationships, pulls community
    summaries, and (when ``include_memories`` is True — the default) folds
    in relevant flat ``Memory`` nodes so the answer stays useful even
    when the graph layer is empty for this topic. Returns formatted text.
    """
    return await ltm.graph.recall_graph_context(
        query=query,
        entity_limit=entity_limit,
        max_hops=max_hops,
        min_confidence=min_confidence,
        include_contradictions=include_contradictions,
        include_provenance=include_provenance,
        include_claims=include_claims,
        include_documents=include_documents,
        include_memories=include_memories,
        memory_limit=memory_limit,
        node_types=node_types,
    )


# -------------------------------------------------------------------
# Maintenance tools
# -------------------------------------------------------------------


@mcp.tool()
async def graph_decay_confidence(
    half_life_days: int = 30,
) -> int:
    """Exponentially decay confidence of factual edges not confirmed recently."""
    return await ltm.graph.decay_confidence(half_life_days=half_life_days)


@mcp.tool()
async def graph_prune(
    min_confidence: float = 0.1,
    max_age_days: int = 180,
    dry_run: bool = True,
    sample_size: int = 5,
) -> Dict[str, Any]:
    """Remove stale, low-confidence graph elements. Set dry_run=False to actually delete.

    Orphaned entities are swept in two tiers so an ``upsert_entity`` now /
    wire-up-relationships-later workflow is not eaten by the next prune:

    * ``is_placeholder = True`` entities (auto-created by
      ``graph_store_relationship`` / ``graph_store_contradiction`` /
      ``graph_store_claim`` when an endpoint was missing) are swept
      unconditionally — they are typo-fallout if nothing followed up.
    * Real entities (``is_placeholder`` absent or False) only get swept
      when their most recent timestamp
      (``last_confirmed`` → ``last_seen`` → ``first_seen``) is older than
      ``max_age_days``. Set ``max_age_days <= 0`` to disable the age gate
      and keep real orphans forever.

    The result includes an ``entity_breakdown`` key with the per-tier
    counts, and entity samples carry ``is_placeholder`` + ``last_touched``
    so you can see which tier each one came from.

    On dry runs, up to ``sample_size`` example items per category are
    returned under a ``samples`` key so callers can eyeball what would be
    deleted before committing.
    """
    return await ltm.graph.prune(
        min_confidence=min_confidence,
        max_age_days=max_age_days,
        dry_run=dry_run,
        sample_size=sample_size,
    )


# Application entry point
if __name__ == "__main__":
    try:
        mcp.run(
            transport="streamable-http", host=HOST_ADDRESS, port=HOST_PORT
        )
    except KeyboardInterrupt:
        logging.info("Shutting down...")
