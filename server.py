"""
Long term memory server using FastMCP and Neo4j.
This server provides a set of tools for storing, retrieving, and managing long-term memories, entities, relationships, claims, and documents in a Neo4j graph database.
It is designed to be used as a backend for AI agents that require persistent memory and knowledge graph capabilities.
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from mcp.server import FastMCP

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
async def lifespan(app: FastMCP):
    """Startup: initialize Neo4j driver and schema.  Shutdown: close driver."""
    await ltm.async_init()
    try:
        yield
    finally:
        await ltm.close()


# Initialize the FastMCP server
mcp = FastMCP(
    name="Long term memory", host=HOST_ADDRESS, port=HOST_PORT, lifespan=lifespan
)

# -------------------------------------------------------------------
# Memory tools
# -------------------------------------------------------------------


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
    """Return counts of all graph elements (entities, relationships, documents, claims, etc.)."""
    return await ltm.graph.stats()


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
    rel_id_a: str,
    rel_id_b: str,
    explanation: str = "",
    session_id: str = "",
) -> Dict[str, Any]:
    """Record that two relationships contradict each other."""
    return await ltm.graph.store_contradiction(
        rel_id_a=rel_id_a,
        rel_id_b=rel_id_b,
        explanation=explanation,
        session_id=session_id,
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
) -> Dict[str, Any]:
    """Store a Claim node and link it to entities via ASSERTS edges."""
    return await ltm.graph.store_claim(
        text=text,
        confidence=confidence,
        status=status,
        source_session=source_session,
        step_id=step_id,
        entity_names=entity_names,
        document_id=document_id,
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
    doc_type: Optional[str] = None,
    min_credibility: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Find documents by semantic search, optionally filtered."""
    return await ltm.graph.find_documents(
        query=query,
        limit=limit,
        doc_type=doc_type,
        min_credibility=min_credibility,
    )


@mcp.tool()
async def graph_link_document_to_entity(
    document_url: str,
    entity_name: str,
    relationship_label: str = "MENTIONS",
) -> Dict[str, Any]:
    """Create a typed edge from a Document to an entity node."""
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
    node_types: Optional[List[str]] = None,
) -> str:
    """Build a structured text context from the knowledge graph for a query. Returns formatted text."""
    return await ltm.graph.recall_graph_context(
        query=query,
        entity_limit=entity_limit,
        max_hops=max_hops,
        min_confidence=min_confidence,
        include_contradictions=include_contradictions,
        include_provenance=include_provenance,
        include_claims=include_claims,
        include_documents=include_documents,
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
) -> Dict[str, int]:
    """Remove stale, low-confidence graph elements. Set dry_run=False to actually delete."""
    return await ltm.graph.prune(
        min_confidence=min_confidence,
        max_age_days=max_age_days,
        dry_run=dry_run,
    )


# Application entry point
if __name__ == "__main__":
    try:
        mcp.run(transport="streamable-http")
    except KeyboardInterrupt:
        logging.info("Shutting down...")
