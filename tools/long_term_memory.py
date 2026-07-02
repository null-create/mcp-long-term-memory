"""
long_term_memory.py

Persistent cross-session memory store backed by Neo4j, with an integrated
knowledge graph (GraphRAG) for capturing relationships between facts.

Replaces the standalone memory MCP server (mcp/memory/) by running the same
persistence logic in-process alongside the FastAPI backend.  Embedding is
handled by the shared embeddings.py module (sentence-transformers +
ThreadPoolExecutor), so there is exactly one model instance and one thread-pool
in the process.

Neo4j's native graph model stores entities and relationships as first-class
graph primitives, and its built-in vector indexes (since 5.11) provide cosine-
similarity ranking for embedding-based semantic search.

Architecture
------------
The store has two layers:

1. **Flat memory store** (``Memory`` nodes): raw text blobs with cosine
   similarity ranking via a Neo4j vector index.  Used for raw evidence
   persistence and simple semantic search.

2. **Knowledge graph** (``KnowledgeGraph``): Typed entity nodes, Claim nodes,
   Document nodes, and Community nodes with a rich set of typed relationship
   labels supporting structural recall, provenance tracking, contradiction
   detection, and hierarchical taxonomy.

Entity Node Types (fully separate labels)
------------------------------------------
  Person       — people with affiliation, role, expertise_areas
  Organization — companies/institutions with org_type, industry, headquarters
  Technology   — tools/frameworks with tech_category, version, license, maturity
  Concept      — abstract ideas with domain, abstraction_level
  Event        — temporal occurrences with start_date, end_date, event_type
  Location     — geographic entities with geo_type, parent_location
  Metric       — quantitative data with value, unit, measurement_date, trend

Additional Node Types
---------------------
  Memory    — raw evidence text blobs
  Claim     — individual assertions with confidence + status + source linkage
  Document  — research source documents (subsumes old Source node)
  Community — LLM-summarized entity clusters

Typed Relationship Labels (factual)
------------------------------------
  CAUSES, ENABLES, PREVENTS, REQUIRES, PART_OF, USES, PRODUCES,
  COMPETES_WITH, AFFILIATED_WITH, AUTHORED_BY, FUNDED_BY,
  PRECEDED_BY, OCCURRED_AT, ASSERTS, SUPPORTS, REFUTES, MENTIONS,
  RELATES_TO (catch-all fallback)

Structural Relationship Labels
-------------------------------
  IS_A         — hierarchical taxonomy (child → parent category)
  CONTRADICTS  — flags two entities as involved in conflicting claims
  SOURCED_FROM — entity → Document node (backward compat alias for old Source)
  MEMBER_OF    — entity → community cluster

Public API
----------
  async_init()                               → None  (call once after event loop starts)
  store(content, category, importance, tags) → Dict
  find_similar(query, limit, min_similarity) → List[Dict]
  recall(query, category, min_importance,    → List[Dict]
         limit, similarity_threshold)

Graph API (via .graph attribute)
---------------------------------
  graph.upsert_entity(name, entity_type, description, session_id, properties) → Dict
  graph.store_relationship(source, target, relation, evidence, ...,
                           relationship_label)                                → Dict
  graph.store_hierarchy(child_name, parent_name)                              → Dict
  graph.store_contradiction(rel_id_a, rel_id_b, explanation, session_id)      → Dict
  graph.store_source(url, title, credibility_score)                           → Dict  (deprecated → store_document)
  graph.link_to_source(entity_name, source_url)                               → Dict  (deprecated → link_document_to_entity)
  graph.store_document(url, title, content_summary, ...)                      → Dict
  graph.find_documents(query, limit, doc_type, min_credibility)               → List[Dict]
  graph.link_document_to_entity(document_id, entity_name, rel_label)          → Dict
  graph.store_claim(text, confidence, status, ...)                            → Dict
  graph.find_claims(query, limit, entity_name, status)                        → List[Dict]
  graph.update_claim_status(claim_id, new_status)                             → Dict
  graph.find_entities(query, limit, include_hierarchy, node_types)            → List[Dict]
  graph.find_contradictions(entity_names, limit)                              → List[Dict]
  graph.get_relationships(entity_ids, max_hops)                               → List[Dict]
  graph.get_communities(entity_ids)                                           → List[Dict]
  graph.get_provenance(entity_names)                                          → List[Dict]
  graph.recent_entities(since_date, limit)                                    → List[Dict]
  graph.recent_relationships(since_date, limit)                               → List[Dict]
  graph.session_diff(session_id)                                              → Dict
  graph.find_paths(source_name, target_name, max_depth)                       → List[Dict]
  graph.find_common_neighbors(entity_names, min_shared)                       → List[Dict]
  graph.decay_confidence(half_life_days)                                      → int
  graph.prune(min_confidence, max_age_days, dry_run)                          → Dict
  graph.update_communities(summarize_fn)                                      → int
  graph.recall_graph_context(query, entity_limit, max_hops, ...)              → str
  graph.stats()                                                                → Dict
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple

import neo4j
from neo4j import AsyncGraphDatabase

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Cosine similarity threshold above which a candidate is considered a
# near-duplicate of an already-stored memory.
_DEDUP_THRESHOLD: float = 0.95

# Cosine similarity threshold for entity deduplication — entities with names
# that embed above this are considered the same entity and merged.
_ENTITY_DEDUP_THRESHOLD: float = 0.92

# Minimum number of shared relationships for two entities to be grouped
# into the same community during community detection.
_COMMUNITY_MIN_SHARED_RELS: int = 2

# ---------------------------------------------------------------------------
# Entity type system
# ---------------------------------------------------------------------------

# Canonical entity type labels — each is a separate Neo4j node label.
ENTITY_TYPES: Tuple[str, ...] = (
    "Person",
    "Organization",
    "Technology",
    "Concept",
    "Event",
    "Location",
    "Metric",
)

# Maps the LLM's lowercase type string to the Neo4j node label.
_TYPE_TO_LABEL: Dict[str, str] = {
    "person": "Person",
    "organization": "Organization",
    "org": "Organization",
    "technology": "Technology",
    "tech": "Technology",
    "concept": "Concept",
    "event": "Event",
    "location": "Location",
    "metric": "Metric",
}

# Type specificity ranking — higher is more specific.  When cross-type dedup
# detects a near-duplicate under a different type, the more specific type wins.
_TYPE_SPECIFICITY: Dict[str, int] = {
    "Person": 7,
    "Organization": 6,
    "Technology": 5,
    "Event": 4,
    "Location": 3,
    "Metric": 2,
    "Concept": 1,
}

# Vector index names for each entity type.
_ENTITY_INDEX_NAMES: Dict[str, str] = {
    label: f"{label.lower()}_embedding_idx" for label in ENTITY_TYPES
}

# ---------------------------------------------------------------------------
# Typed relationship system
# ---------------------------------------------------------------------------

# Factual relationship labels (used for multi-hop traversal queries).
_FACTUAL_REL_TYPES: Tuple[str, ...] = (
    "CAUSES",
    "ENABLES",
    "PREVENTS",
    "REQUIRES",
    "PART_OF",
    "USES",
    "PRODUCES",
    "COMPETES_WITH",
    "AFFILIATED_WITH",
    "AUTHORED_BY",
    "FUNDED_BY",
    "PRECEDED_BY",
    "OCCURRED_AT",
    "ASSERTS",
    "SUPPORTS",
    "REFUTES",
    "MENTIONS",
    "RELATES_TO",
)

# Cypher fragment for matching any factual relationship type.
_FACTUAL_REL_CYPHER = "|".join(_FACTUAL_REL_TYPES)

# Keyword patterns for classifying free-form verb phrases into typed labels.
# Each tuple: (compiled regex, Neo4j relationship label).
# Order matters — first match wins.
_RELATION_CLASSIFIERS: List[Tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"\bcaus(es?|ed|ing)\b|\bresult(s|ed)? in\b|\bleads? to\b|\btrigger", re.I
        ),
        "CAUSES",
    ),
    (
        re.compile(r"\benabl(es?|ed|ing)\b|\bfacilit|\ballow(s|ed)?\b|\bempower", re.I),
        "ENABLES",
    ),
    (
        re.compile(r"\bpreven(ts?|ted|ting)\b|\binhib|\bblock(s|ed)?\b|\bhindr", re.I),
        "PREVENTS",
    ),
    (
        re.compile(
            r"\brequir(es?|ed|ing)\b|\bdepend(s|ed)? on\b|\bneeds?\b|\bnecessit", re.I
        ),
        "REQUIRES",
    ),
    (
        re.compile(
            r"\bpart of\b|\bcompon|\bincluded? in\b|\bsubset\b|\bcontain(s|ed)?\b", re.I
        ),
        "PART_OF",
    ),
    (
        re.compile(
            r"\buses?\b|\butiliz|\bemploy(s|ed)?\b|\bleverage|\bpowered by\b", re.I
        ),
        "USES",
    ),
    (
        re.compile(
            r"\bproduc(es?|ed|ing)\b|\bcreat(es?|ed|ing)\b|\bgenerat|\bbuil[dt]\b|\bdevelop",
            re.I,
        ),
        "PRODUCES",
    ),
    (
        re.compile(r"\bcompet(es?|ed|ing)\b|\brival|\balternativ|\bvs\.?\b", re.I),
        "COMPETES_WITH",
    ),
    (
        re.compile(
            r"\baffiliat|\bmember of\b|\bworks? (at|for)\b|\bemploy(ed|ee)?\b|\bjoined?\b",
            re.I,
        ),
        "AFFILIATED_WITH",
    ),
    (
        re.compile(
            r"\bauthor(ed|s)?\b|\bwrote\b|\bwritten by\b|\bpublish(ed)?\b|\bcreator\b",
            re.I,
        ),
        "AUTHORED_BY",
    ),
    (
        re.compile(
            r"\bfund(ed|s|ing)?\b|\bfinanc(ed|es|ing)\b|\binvest(ed|s)?\b|\bsponsor",
            re.I,
        ),
        "FUNDED_BY",
    ),
    (
        re.compile(
            r"\bpreced(ed|es|ing)?\b|\bbefore\b|\bprior to\b|\bfollow(s|ed)? by\b|\bsucce(ed|ss)",
            re.I,
        ),
        "PRECEDED_BY",
    ),
    (
        re.compile(
            r"\boccur(red|s)? (at|in)\b|\blocated (at|in)\b|\bheld (at|in)\b|\btakes? place\b",
            re.I,
        ),
        "OCCURRED_AT",
    ),
    (
        re.compile(
            r"\bassert(s|ed)?\b|\bclaim(s|ed)?\b|\bstat(es?|ed)\b|\bdeclar", re.I
        ),
        "ASSERTS",
    ),
    (
        re.compile(
            r"\bsupport(s|ed|ing)?\b|\bconfirm(s|ed)?\b|\bvalidat|\bcorroborat|\bevidence for\b",
            re.I,
        ),
        "SUPPORTS",
    ),
    (
        re.compile(
            r"\brefut(es?|ed|ing)\b|\bcontradict(s|ed)?\b|\bdisprov|\bchalleng(es?|ed)\b|\bundermin",
            re.I,
        ),
        "REFUTES",
    ),
    (
        re.compile(
            r"\bmention(s|ed|ing)?\b|\brefer(s|red)? to\b|\bcit(es?|ed)\b|\bnot(es?|ed)\b",
            re.I,
        ),
        "MENTIONS",
    ),
]

# Claim status values.
CLAIM_STATUSES: Tuple[str, ...] = (
    "supported",
    "disputed",
    "unverified",
    "retracted",
)


def classify_relation(verb_phrase: str) -> str:
    """Map a free-form verb phrase to a typed Neo4j relationship label.

    Uses keyword matching — no LLM call, deterministic, zero latency.
    Falls back to ``RELATES_TO`` when no pattern matches.
    """
    for pattern, label in _RELATION_CLASSIFIERS:
        if pattern.search(verb_phrase):
            return label
    return "RELATES_TO"


def resolve_entity_label(entity_type: str) -> str:
    """Map LLM entity type string to Neo4j node label.  Defaults to Concept."""
    return _TYPE_TO_LABEL.get(entity_type.lower().strip(), "Concept")


class AsyncLongTermMemory:
    """
    Async, in-process persistent memory store backed by Neo4j.

    Embeddings are generated by the shared ``embeddings.py`` module and stored
    as properties on Neo4j nodes.  Vector indexes provide cosine-similarity
    search over those embeddings.

    Usage
    -----
    ltm = AsyncLongTermMemory(neo4j_uri="bolt://localhost:7687")
    await ltm.async_init()   # call once, after the event loop is running

    await ltm.store("Key finding: X is true", category="research_finding", importance=7)
    memories = await ltm.find_similar("what do we know about X?", limit=5)
    """

    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "research_pass",
        neo4j_database: str = "neo4j",
        embedding_dimensions: int = 384,
    ) -> None:
        self._uri = neo4j_uri
        self._user = neo4j_user
        self._password = neo4j_password
        self._database = neo4j_database
        self._embedding_dimensions = embedding_dimensions
        self._driver: Optional[neo4j.AsyncDriver] = None
        # Set to False if async_init() fails (Neo4j unavailable); all
        # public methods become no-ops so the research pipeline is unaffected.
        self._available: bool = False
        # Knowledge graph layer — shares the same Neo4j driver.
        self.graph: KnowledgeGraph = KnowledgeGraph(self)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    async def async_init(self) -> None:
        """Initialize the Neo4j driver, constraints, and vector indexes.

        Must be called once after the asyncio event loop is running (e.g. from
        the FastAPI lifespan handler).  Idempotent — safe to call multiple times.

        Failures are logged as warnings rather than raised so that an unavailable
        Neo4j instance never blocks server startup.  When init fails, all public
        methods silently return empty results.
        """
        try:
            self._driver = AsyncGraphDatabase.driver(
                self._uri, auth=(self._user, self._password)
            )
            await self._driver.verify_connectivity()
            await self._create_schema()
            self._available = True
            logger.info(
                "[LongTermMemory] Ready — uri=%s  database=%s",
                self._uri,
                self._database,
            )
        except Exception as exc:
            logger.warning(
                "[LongTermMemory] Initialization failed — long-term memory will be "
                "disabled for this session.  Cause: %s",
                exc,
            )

    async def _create_schema(self) -> None:
        """Create uniqueness constraints and vector indexes (idempotent)."""
        assert self._driver is not None
        dims = self._embedding_dimensions
        async with self._driver.session(database=self._database) as session:
            # ── Memory node ───────────────────────────────────────────────
            await session.run(
                "CREATE CONSTRAINT memory_id IF NOT EXISTS "
                "FOR (m:Memory) REQUIRE m.id IS UNIQUE"
            )
            await session.run(
                "CREATE VECTOR INDEX memory_embedding_idx IF NOT EXISTS "
                "FOR (m:Memory) ON (m.embedding) "
                "OPTIONS {indexConfig: {`vector.dimensions`: $dims, "
                "`vector.similarity_function`: 'cosine'}}",
                dims=dims,
            )

            # ── Typed entity nodes ────────────────────────────────────────
            for label in ENTITY_TYPES:
                idx_name = _ENTITY_INDEX_NAMES[label]
                await session.run(
                    f"CREATE CONSTRAINT {label.lower()}_id IF NOT EXISTS "
                    f"FOR (e:{label}) REQUIRE e.id IS UNIQUE"
                )
                await session.run(
                    f"CREATE VECTOR INDEX {idx_name} IF NOT EXISTS "
                    f"FOR (e:{label}) ON (e.embedding) "
                    "OPTIONS {indexConfig: {`vector.dimensions`: $dims, "
                    "`vector.similarity_function`: 'cosine'}}",
                    dims=dims,
                )

            # ── Community node ────────────────────────────────────────────
            await session.run(
                "CREATE CONSTRAINT community_id IF NOT EXISTS "
                "FOR (c:Community) REQUIRE c.id IS UNIQUE"
            )
            await session.run(
                "CREATE VECTOR INDEX community_embedding_idx IF NOT EXISTS "
                "FOR (c:Community) ON (c.embedding) "
                "OPTIONS {indexConfig: {`vector.dimensions`: $dims, "
                "`vector.similarity_function`: 'cosine'}}",
                dims=dims,
            )

            # ── Claim node ────────────────────────────────────────────────
            await session.run(
                "CREATE CONSTRAINT claim_id IF NOT EXISTS "
                "FOR (cl:Claim) REQUIRE cl.id IS UNIQUE"
            )
            await session.run(
                "CREATE VECTOR INDEX claim_embedding_idx IF NOT EXISTS "
                "FOR (cl:Claim) ON (cl.embedding) "
                "OPTIONS {indexConfig: {`vector.dimensions`: $dims, "
                "`vector.similarity_function`: 'cosine'}}",
                dims=dims,
            )

            # ── Document node (subsumes old Source) ───────────────────────
            await session.run(
                "CREATE CONSTRAINT document_id IF NOT EXISTS "
                "FOR (d:Document) REQUIRE d.id IS UNIQUE"
            )
            await session.run(
                "CREATE CONSTRAINT document_url IF NOT EXISTS "
                "FOR (d:Document) REQUIRE d.url IS UNIQUE"
            )
            await session.run(
                "CREATE VECTOR INDEX document_embedding_idx IF NOT EXISTS "
                "FOR (d:Document) ON (d.embedding) "
                "OPTIONS {indexConfig: {`vector.dimensions`: $dims, "
                "`vector.similarity_function`: 'cosine'}}",
                dims=dims,
            )

    async def close(self) -> None:
        """Close the Neo4j driver.  Call during application shutdown."""
        if self._driver is not None:
            await self._driver.close()
            self._driver = None
            self._available = False

    # ------------------------------------------------------------------
    # Public async API (flat memory store — unchanged)
    # ------------------------------------------------------------------

    async def store(
        self,
        content: str,
        category: str = "general",
        importance: int = 5,
        tags: Optional[List[str]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Store a memory.  Returns ``{"success": False}`` if LTM is unavailable."""
        if not self._available:
            return {"success": False, "message": "LongTermMemory not available"}

        if not content or not content.strip():
            return {"success": False, "message": "Empty content"}

        from embeddings import embed_texts, EmbeddingError, EMBEDDINGS_ENABLED

        vector: Optional[List[float]] = None
        if EMBEDDINGS_ENABLED:
            try:
                vecs = await embed_texts([content])
                vector = vecs[0] if vecs else None
            except EmbeddingError as exc:
                logger.debug(
                    "[LongTermMemory] Embedding unavailable for store: %s", exc
                )
            except Exception as exc:
                logger.debug(
                    "[LongTermMemory] Unexpected embed error in store: %s", exc
                )

        assert self._driver is not None
        memory_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        tags_list = tags or []
        extra = extra_metadata or {}

        async with self._driver.session(database=self._database) as session:
            try:
                # Dedup check via vector similarity
                if vector is not None:
                    result = await session.run(
                        "CALL db.index.vector.queryNodes("
                        "  'memory_embedding_idx', 1, $vector"
                        ") YIELD node, score "
                        "WHERE score >= $threshold "
                        "RETURN node.id AS id, score",
                        vector=vector,
                        threshold=_DEDUP_THRESHOLD,
                    )
                    record = await result.single()
                    if record:
                        return {
                            "success": False,
                            "memory_id": record["id"],
                            "message": (
                                f"Near-duplicate already stored "
                                f"(similarity={record['score']:.3f})"
                            ),
                        }

                # Build the CREATE query
                props: Dict[str, Any] = {
                    "id": memory_id,
                    "content": content,
                    "category": category,
                    "importance": importance,
                    "created_at": now,
                    "last_accessed": now,
                    "access_count": 0,
                    "tags": json.dumps(tags_list),
                }
                for k, v in extra.items():
                    props[f"custom_{k}"] = (
                        json.dumps(v) if isinstance(v, (list, dict)) else v
                    )
                if vector is not None:
                    props["embedding"] = vector

                await session.run(
                    "CREATE (m:Memory) SET m = $props",
                    props=props,
                )
                return {"success": True, "memory_id": memory_id, "message": "Stored"}

            except Exception as exc:
                logger.error("[LongTermMemory] store failed: %s", exc)
                return {"success": False, "message": str(exc)}

    async def find_similar(
        self,
        query: str,
        limit: int = 5,
        min_similarity: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Find memories semantically similar to *query*. Returns [] if LTM unavailable."""
        if not self._available:
            return []
        return await self.recall(
            query=query,
            limit=limit,
            similarity_threshold=min_similarity,
        )

    async def recall(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        min_importance: Optional[int] = None,
        limit: int = 10,
        similarity_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Recall memories via semantic search.  Returns [] if LTM unavailable."""
        if not self._available:
            return []

        if not query:
            return await self._recall_no_query(category, min_importance, limit)

        from embeddings import (
            embed_query as _embed_q,
            EmbeddingError,
            EMBEDDINGS_ENABLED,
        )

        query_vec: Optional[List[float]] = None
        if EMBEDDINGS_ENABLED:
            try:
                query_vec = await _embed_q(query)
            except EmbeddingError as exc:
                logger.debug(
                    "[LongTermMemory] Embedding unavailable for recall: %s", exc
                )
            except Exception as exc:
                logger.debug(
                    "[LongTermMemory] Unexpected embed error in recall: %s", exc
                )

        if query_vec is None:
            return await self._recall_no_query(category, min_importance, limit)

        return await self._recall_with_vector(
            query_vec, category, min_importance, limit, similarity_threshold
        )

    # ------------------------------------------------------------------
    # Internal recall helpers
    # ------------------------------------------------------------------

    async def _recall_with_vector(
        self,
        query_vec: List[float],
        category: Optional[str],
        min_importance: Optional[int],
        limit: int,
        similarity_threshold: float,
    ) -> List[Dict[str, Any]]:
        assert self._driver is not None
        # Fetch more than needed to allow post-filtering
        fetch_limit = limit * 2

        where_parts: List[str] = []
        params: Dict[str, Any] = {
            "vector": query_vec,
            "fetch_limit": fetch_limit,
            "threshold": similarity_threshold,
        }
        if category:
            where_parts.append("node.category = $category")
            params["category"] = category
        if min_importance is not None:
            where_parts.append("node.importance >= $min_importance")
            params["min_importance"] = min_importance

        where_clause = ""
        if where_parts:
            where_clause = " AND " + " AND ".join(where_parts)

        cypher = (
            "CALL db.index.vector.queryNodes("
            "  'memory_embedding_idx', $fetch_limit, $vector"
            ") YIELD node, score "
            f"WHERE score >= $threshold{where_clause} "
            "RETURN node, score "
            "ORDER BY node.importance * score DESC "
            "LIMIT $limit"
        )
        params["limit"] = limit

        async with self._driver.session(database=self._database) as session:
            try:
                result = await session.run(cypher, **params)
                records = await result.data()
            except Exception as exc:
                logger.debug("[LongTermMemory] recall query failed: %s", exc)
                return []

        now = datetime.now().isoformat()
        memories: List[Dict[str, Any]] = []
        update_ids: List[str] = []

        for rec in records:
            node = rec["node"]
            score = rec["score"]
            memories.append(self._unpack(node, score))
            update_ids.append(node["id"])

        # Batch-update access stats (best-effort)
        if update_ids:
            try:
                async with self._driver.session(database=self._database) as session:
                    await session.run(
                        "UNWIND $ids AS mid "
                        "MATCH (m:Memory {id: mid}) "
                        "SET m.last_accessed = $now, "
                        "    m.access_count = m.access_count + 1",
                        ids=update_ids,
                        now=now,
                    )
            except Exception:
                pass

        return memories

    async def _recall_no_query(
        self,
        category: Optional[str],
        min_importance: Optional[int],
        limit: int,
    ) -> List[Dict[str, Any]]:
        assert self._driver is not None

        where_parts: List[str] = []
        params: Dict[str, Any] = {"limit": limit}
        if category:
            where_parts.append("m.category = $category")
            params["category"] = category
        if min_importance is not None:
            where_parts.append("m.importance >= $min_importance")
            params["min_importance"] = min_importance

        where_clause = " WHERE " + " AND ".join(where_parts) if where_parts else ""
        cypher = f"MATCH (m:Memory){where_clause} RETURN m LIMIT $limit"

        async with self._driver.session(database=self._database) as session:
            try:
                result = await session.run(cypher, **params)
                records = await result.data()
            except Exception as exc:
                logger.debug("[LongTermMemory] recall (no query) failed: %s", exc)
                return []

        return [self._unpack(rec["m"], 0.0) for rec in records]

    @staticmethod
    def _unpack(node: Dict[str, Any], score: float) -> Dict[str, Any]:
        tags = json.loads(node.get("tags", "[]"))
        custom: Dict[str, Any] = {}
        for k, v in node.items():
            if k.startswith("custom_"):
                try:
                    custom[k[7:]] = json.loads(v)
                except Exception:
                    custom[k[7:]] = v
        return {
            "id": node.get("id"),
            "content": node.get("content"),
            "category": node.get("category"),
            "importance": node.get("importance"),
            "created_at": node.get("created_at"),
            "last_accessed": node.get("last_accessed"),
            "access_count": node.get("access_count", 0),
            "tags": tags,
            "metadata": custom or None,
            "similarity": round(score, 4),
        }


# ---------------------------------------------------------------------------
# Knowledge Graph (GraphRAG)
# ---------------------------------------------------------------------------


class KnowledgeGraph:
    """
    Relationship-aware knowledge graph stored natively in Neo4j.

    Typed entity nodes (Person, Organization, Technology, Concept, Event,
    Location, Metric) replace the old generic Entity label.  Each type has
    its own vector index for semantic search and type-specific properties.

    Typed relationship labels (CAUSES, ENABLES, USES, etc.) replace the old
    catch-all RELATES_TO.  Free-form verb phrases are classified via keyword
    matching into typed labels; unmatched phrases fall back to RELATES_TO.

    Claim nodes store individual assertions linked to entities and documents.
    Document nodes subsume the old Source node with richer provenance.

    Community nodes are LLM-generated summaries of densely-connected entity
    clusters, linked via MEMBER_OF edges.

    All public methods are async.  The ``_ltm`` back-reference provides access
    to the ``_available`` flag and the Neo4j driver.
    """

    def __init__(self, ltm: AsyncLongTermMemory) -> None:
        self._ltm = ltm

    @property
    def _driver(self) -> Optional[neo4j.AsyncDriver]:
        return self._ltm._driver

    @property
    def _database(self) -> str:
        return self._ltm._database

    @property
    def available(self) -> bool:
        return self._ltm._available and self._driver is not None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _embed(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text, returning None on failure."""
        from embeddings import embed_texts, EmbeddingError, EMBEDDINGS_ENABLED

        if not EMBEDDINGS_ENABLED:
            return None
        try:
            vecs = await embed_texts([text])
            return vecs[0] if vecs else None
        except (EmbeddingError, Exception) as exc:
            logger.debug("[KnowledgeGraph] Embedding failed: %s", exc)
            return None

    async def _embed_query(self, text: str) -> Optional[List[float]]:
        """Generate query embedding, returning None on failure."""
        from embeddings import embed_query as _eq, EmbeddingError, EMBEDDINGS_ENABLED

        if not EMBEDDINGS_ENABLED:
            return None
        try:
            return await _eq(text)
        except (EmbeddingError, Exception) as exc:
            logger.debug("[KnowledgeGraph] Query embed failed: %s", exc)
            return None

    def _build_union_vector_search(
        self,
        index_names: List[str],
        limit: int,
        extra_return_fields: str = "",
    ) -> str:
        """Build a UNION ALL Cypher query across multiple vector indexes.

        Each sub-query fetches ``limit`` candidates from its index.  The outer
        query de-duplicates by node id, keeps the highest score per node, and
        returns the top ``limit`` results sorted by score descending.
        """
        subqueries: List[str] = []
        for idx_name in index_names:
            sq = (
                f"CALL db.index.vector.queryNodes('{idx_name}', $limit, $vector) "
                f"YIELD node, score "
                f"RETURN node, score{extra_return_fields}"
            )
            subqueries.append(sq)

        union = " UNION ALL ".join(subqueries)
        outer = (
            f"CALL {{ {union} }} "
            "WITH node, score "
            "ORDER BY score DESC "
            "WITH node, max(score) AS score "
            "ORDER BY score DESC "
            "LIMIT $limit"
        )
        return outer

    def _any_entity_match(self) -> str:
        """Cypher fragment matching any typed entity node by name."""
        parts = [
            f"MATCH (e:{label} {{name: $name}}) RETURN e" for label in ENTITY_TYPES
        ]
        return " UNION ".join(parts)

    # ------------------------------------------------------------------
    # Entity CRUD
    # ------------------------------------------------------------------

    async def upsert_entity(
        self,
        name: str,
        entity_type: str = "concept",
        description: str = "",
        session_id: str = "",
        properties: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Insert or update a typed entity node.

        Deduplicates by semantic name similarity across ALL entity type indexes.
        When a near-duplicate exists under a different type, the more specific
        type wins (specificity ranking: Person > Organization > Technology >
        Event > Location > Metric > Concept).
        """
        if not self.available or not name or not name.strip():
            return {"success": False, "message": "Graph not available or empty name"}

        label = resolve_entity_label(entity_type)
        doc_text = f"{name}: {description}" if description else name
        vector = await self._embed(doc_text)

        assert self._driver is not None
        now = datetime.now().isoformat()
        name_stripped = name.strip()
        desc_stripped = description.strip()
        extra_props = properties or {}

        async with self._driver.session(database=self._database) as session:
            # Dedup: check ALL entity type vector indexes for near-duplicate
            if vector is not None:
                try:
                    # UNION ALL across all entity type indexes
                    union_q = self._build_union_vector_search(
                        list(_ENTITY_INDEX_NAMES.values()), 1
                    )
                    result = await session.run(
                        f"{union_q} "
                        "WITH node, score "
                        "WHERE score >= $threshold "
                        "RETURN node, score LIMIT 1",
                        vector=vector,
                        limit=1,
                        threshold=_ENTITY_DEDUP_THRESHOLD,
                    )
                    record = await result.single()
                    if record:
                        existing = record["node"]
                        existing_id = existing["id"]
                        mention_count = existing.get("mention_count", 1) + 1
                        sessions = json.loads(existing.get("source_sessions", "[]"))
                        if session_id and session_id not in sessions:
                            sessions.append(session_id)
                        existing_desc = existing.get("description", "")
                        new_doc = (
                            f"{name_stripped}: {desc_stripped}"
                            if desc_stripped
                            else name_stripped
                        )
                        use_doc = (
                            new_doc
                            if len(new_doc) > len(existing_desc)
                            else existing_desc
                        )
                        update_props: Dict[str, Any] = {
                            "mention_count": mention_count,
                            "source_sessions": json.dumps(sessions),
                            "last_seen": now,
                            "description": use_doc,
                            "last_confirmed": now,
                            "confirmation_count": existing.get("confirmation_count", 1)
                            + 1,
                        }
                        if vector is not None:
                            update_props["embedding"] = vector
                        # Merge type-specific properties
                        for k, v in extra_props.items():
                            update_props[k] = v

                        # Determine if type upgrade is needed
                        existing_type = existing.get("entity_type", "concept")
                        existing_label = resolve_entity_label(existing_type)
                        new_specificity = _TYPE_SPECIFICITY.get(label, 1)
                        existing_specificity = _TYPE_SPECIFICITY.get(existing_label, 1)

                        if (
                            new_specificity > existing_specificity
                            and label != existing_label
                        ):
                            # Type upgrade: remove old label, add new one
                            update_props["entity_type"] = entity_type.lower().strip()
                            await session.run(
                                f"MATCH (e:{existing_label} {{id: $eid}}) "
                                f"REMOVE e:{existing_label} "
                                f"SET e:{label} "
                                "SET e += $props",
                                eid=existing_id,
                                props=update_props,
                            )
                        else:
                            # Same type or less specific — just update props
                            await session.run(
                                f"MATCH (e {{id: $eid}}) " "SET e += $props",
                                eid=existing_id,
                                props=update_props,
                            )
                        return {
                            "success": True,
                            "entity_id": existing_id,
                            "entity_type": label,
                            "merged": True,
                            "mention_count": mention_count,
                        }
                except Exception as exc:
                    logger.debug("[KnowledgeGraph] Entity dedup check failed: %s", exc)

            # New entity
            entity_id = str(uuid.uuid4())
            props: Dict[str, Any] = {
                "id": entity_id,
                "name": name_stripped,
                "entity_type": entity_type.lower().strip(),
                "description": (
                    f"{name_stripped}: {desc_stripped}"
                    if desc_stripped
                    else name_stripped
                ),
                "mention_count": 1,
                "first_seen": now,
                "last_seen": now,
                "last_confirmed": now,
                "confirmation_count": 1,
                "source_sessions": json.dumps([session_id] if session_id else []),
            }
            # Type-specific properties
            for k, v in extra_props.items():
                props[k] = v
            if vector is not None:
                props["embedding"] = vector

            try:
                await session.run(
                    f"CREATE (e:{label}) SET e = $props",
                    props=props,
                )
                return {
                    "success": True,
                    "entity_id": entity_id,
                    "entity_type": label,
                    "merged": False,
                }
            except Exception as exc:
                logger.error("[KnowledgeGraph] Entity store failed: %s", exc)
                return {"success": False, "message": str(exc)}

    # ------------------------------------------------------------------
    # Relationship CRUD
    # ------------------------------------------------------------------

    async def _entity_exists(self, name: str) -> bool:
        """Return True if an entity node with *name* already exists."""
        if not self.available or not name.strip():
            return False
        assert self._driver is not None
        all_labels = "|".join(ENTITY_TYPES)
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                f"MATCH (n:{all_labels} {{name: $name}}) RETURN n.id AS id LIMIT 1",
                name=name.strip(),
            )
            return (await result.single()) is not None

    async def store_relationship(
        self,
        source: str,
        target: str,
        relation: str,
        evidence: str = "",
        confidence: float = 0.8,
        session_id: str = "",
        step_id: int = 0,
        relationship_label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Store a directed typed relationship between two entities.

        If *relationship_label* is not provided, the free-form *relation*
        string is classified via keyword matching into a typed Neo4j label
        (CAUSES, ENABLES, USES, etc.) with RELATES_TO as fallback.

        Deduplicates by (source_name, target_name, relation_type, rel_label):
        if an identical edge already exists, it is updated rather than
        duplicated — confidence is raised to the higher of the two values,
        evidence is appended (if new), and confirmation_count is incremented.
        """
        if not self.available:
            return {"success": False, "message": "Graph not available"}
        if not source.strip() or not target.strip() or not relation.strip():
            return {
                "success": False,
                "message": "Source, target, and relation required",
            }

        assert self._driver is not None
        now = datetime.now().isoformat()
        src = source.strip()
        tgt = target.strip()
        rel = relation.strip()
        evid = evidence.strip()
        rel_label = relationship_label or classify_relation(rel)

        # Guarantee both endpoints exist before the MATCH ... MATCH ... CREATE
        # below. If either entity is missing, that pattern binds nothing, the
        # CREATE silently no-ops, and the edge is lost — yet the method would
        # still report success. Auto-create any missing endpoint as a generic
        # concept entity so relationships are never silently dropped.
        for endpoint in (src, tgt):
            if not await self._entity_exists(endpoint):
                await self.upsert_entity(
                    name=endpoint,
                    entity_type="concept",
                    session_id=session_id,
                )

        # Build Cypher that matches any typed entity node for source/target.
        # We can't know the label of source/target at this point, so we use
        # a label-agnostic match via property.
        all_labels = "|".join(ENTITY_TYPES)

        async with self._driver.session(database=self._database) as session:
            try:
                # ── Dedup check: exact match on source→target + relation_type + label ─
                dedup_q = (
                    f"MATCH (s:{all_labels} {{name: $source}})"
                    f"-[r:{rel_label}]->"
                    f"(t:{all_labels} {{name: $target}}) "
                    "WHERE r.relation_type = $relation "
                    "RETURN r.id AS rid, r.confidence AS conf, "
                    "       r.evidence AS evid, r.confirmation_count AS cnt"
                )
                result = await session.run(
                    dedup_q, source=src, target=tgt, relation=rel
                )
                existing = await result.single()
                if existing:
                    # Merge into existing edge
                    existing_evid = existing["evid"] or ""
                    merged_evid = (
                        f"{existing_evid}; {evid}".strip("; ")
                        if evid and evid not in existing_evid
                        else existing_evid
                    )
                    new_conf = max(existing["conf"] or 0.8, confidence)
                    new_count = (existing["cnt"] or 1) + 1
                    await session.run(
                        f"MATCH ()-[r:{rel_label} {{id: $rid}}]->() "
                        "SET r.confidence = $conf, "
                        "    r.confirmation_count = $cnt, "
                        "    r.evidence = $evid, "
                        "    r.last_confirmed = $now",
                        rid=existing["rid"],
                        conf=new_conf,
                        cnt=new_count,
                        evid=merged_evid,
                        now=now,
                    )
                    return {
                        "success": True,
                        "relationship_id": existing["rid"],
                        "relationship_label": rel_label,
                        "merged": True,
                    }

                # ── Create new edge ──────────────────────────────────────────
                rel_id = str(uuid.uuid4())
                await session.run(
                    f"MATCH (s:{all_labels} {{name: $source}}) "
                    f"MATCH (t:{all_labels} {{name: $target}}) "
                    f"CREATE (s)-[:{rel_label} {{"
                    "  id: $rel_id, relation_type: $relation, "
                    "  relationship_label: $rel_label, "
                    "  confidence: $confidence, evidence: $evidence, "
                    "  session_id: $session_id, step_id: $step_id, "
                    "  created_at: $now, last_confirmed: $now, "
                    "  confirmation_count: 1"
                    "}]->(t)",
                    source=src,
                    target=tgt,
                    rel_id=rel_id,
                    relation=rel,
                    rel_label=rel_label,
                    confidence=confidence,
                    evidence=evid,
                    session_id=session_id,
                    step_id=step_id,
                    now=now,
                )
                return {
                    "success": True,
                    "relationship_id": rel_id,
                    "relationship_label": rel_label,
                    "merged": False,
                }
            except Exception as exc:
                logger.error("[KnowledgeGraph] Relationship store failed: %s", exc)
                return {"success": False, "message": str(exc)}

    # ------------------------------------------------------------------
    # Entity search
    # ------------------------------------------------------------------

    async def find_entities(
        self,
        query: str,
        limit: int = 5,
        include_hierarchy: bool = False,
        node_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Find entities semantically similar to *query*.

        Searches across all entity type vector indexes (or a subset if
        *node_types* is specified) using UNION ALL, then merges and ranks
        by similarity score.

        If *include_hierarchy* is True, each result also carries an
        ``ancestors`` list of parent entity names (via IS_A edges, up to
        3 hops).
        """
        if not self.available:
            return []

        query_vec = await self._embed_query(query)
        if query_vec is None:
            return []

        # Determine which indexes to query
        if node_types:
            resolved = [resolve_entity_label(t) for t in node_types]
            index_names = [
                _ENTITY_INDEX_NAMES[lbl]
                for lbl in resolved
                if lbl in _ENTITY_INDEX_NAMES
            ]
        else:
            index_names = list(_ENTITY_INDEX_NAMES.values())

        if not index_names:
            return []

        assert self._driver is not None
        async with self._driver.session(database=self._database) as session:
            try:
                union_q = self._build_union_vector_search(index_names, limit)
                if include_hierarchy:
                    cypher = (
                        f"{union_q} "
                        "OPTIONAL MATCH (node)-[:IS_A*1..3]->(anc) "
                        "RETURN node, score, collect(DISTINCT anc.name) AS ancestors"
                    )
                else:
                    cypher = f"{union_q} " "RETURN node, score, [] AS ancestors"
                result = await session.run(cypher, vector=query_vec, limit=limit)
                records = await result.data()
            except Exception as exc:
                logger.debug("[KnowledgeGraph] Entity query failed: %s", exc)
                return []

        entities: List[Dict[str, Any]] = []
        for rec in records:
            node = rec["node"]
            entity_type_str = node.get("entity_type", "concept")
            entity_dict: Dict[str, Any] = {
                "id": node.get("id", ""),
                "name": node.get("name", ""),
                "entity_type": entity_type_str,
                "entity_label": resolve_entity_label(entity_type_str),
                "description": node.get("description", ""),
                "mention_count": node.get("mention_count", 1),
                "last_confirmed": node.get("last_confirmed", ""),
                "confirmation_count": node.get("confirmation_count", 1),
                "similarity": round(rec["score"], 4),
                "ancestors": rec.get("ancestors") or [],
            }
            # Include type-specific properties if present
            for key in (
                "affiliation",
                "role",
                "expertise_areas",
                "org_type",
                "industry",
                "headquarters",
                "tech_category",
                "version",
                "license",
                "maturity",
                "domain",
                "abstraction_level",
                "start_date",
                "end_date",
                "event_type",
                "location",
                "geo_type",
                "parent_location",
                "value",
                "unit",
                "measurement_date",
                "trend",
            ):
                val = node.get(key)
                if val is not None:
                    entity_dict[key] = val
            entities.append(entity_dict)
        return entities

    # ------------------------------------------------------------------
    # Relationship traversal
    # ------------------------------------------------------------------

    async def get_relationships(
        self,
        entity_ids: Optional[List[str]] = None,
        entity_names: Optional[List[str]] = None,
        max_hops: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Get relationships involving the given entities.

        Traverses all typed factual relationship labels (CAUSES, ENABLES,
        USES, ..., RELATES_TO) for multi-hop exploration.
        """
        if not self.available:
            return []

        assert self._driver is not None
        all_labels = "|".join(ENTITY_TYPES)

        # Resolve entity_ids → names
        names: Set[str] = set(entity_names or [])
        if entity_ids:
            async with self._driver.session(database=self._database) as session:
                try:
                    # Search across all typed entity labels
                    result = await session.run(
                        f"MATCH (e:{all_labels}) WHERE e.id IN $ids RETURN e.name AS name",
                        ids=entity_ids,
                    )
                    records = await result.data()
                    for rec in records:
                        if rec["name"]:
                            names.add(rec["name"])
                except Exception:
                    pass

        if not names:
            return []

        # Use variable-length Cypher path for multi-hop traversal across
        # ALL factual relationship types.
        async with self._driver.session(database=self._database) as session:
            try:
                result = await session.run(
                    f"MATCH (seed:{all_labels}) WHERE seed.name IN $names "
                    f"MATCH path = (seed)-[r:{_FACTUAL_REL_CYPHER}*1.."
                    + str(max_hops)
                    + f"]-(other:{all_labels}) "
                    "UNWIND relationships(path) AS rel "
                    "WITH DISTINCT rel, startNode(rel) AS src, endNode(rel) AS tgt "
                    "RETURN rel.id AS rel_id, "
                    "       rel.relation_type AS relation_type, "
                    "       rel.relationship_label AS relationship_label, "
                    "       type(rel) AS rel_neo4j_type, "
                    "       rel.confidence AS confidence, "
                    "       rel.evidence AS evidence, "
                    "       rel.session_id AS session_id, "
                    "       src.name AS source_entity, "
                    "       tgt.name AS target_entity",
                    names=list(names),
                )
                records = await result.data()
            except Exception as exc:
                logger.debug("[KnowledgeGraph] Relationship traversal failed: %s", exc)
                return []

        seen_ids: Set[str] = set()
        all_rels: List[Dict[str, Any]] = []
        for rec in records:
            rel_id = rec.get("rel_id") or ""
            if rel_id in seen_ids:
                continue
            seen_ids.add(rel_id)
            all_rels.append(
                {
                    "id": rel_id,
                    "source_entity": rec.get("source_entity", ""),
                    "target_entity": rec.get("target_entity", ""),
                    "relation_type": rec.get("relation_type", ""),
                    "relationship_label": rec.get("relationship_label")
                    or rec.get("rel_neo4j_type", "RELATES_TO"),
                    "confidence": rec.get("confidence", 0.0),
                    "evidence": rec.get("evidence", ""),
                    "session_id": rec.get("session_id", ""),
                }
            )
        return all_rels

    # ------------------------------------------------------------------
    # Community detection + summaries
    # ------------------------------------------------------------------

    async def get_communities(
        self, entity_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve community summaries relevant to the given entities."""
        if not self.available:
            return []

        assert self._driver is not None
        all_labels = "|".join(ENTITY_TYPES)

        async with self._driver.session(database=self._database) as session:
            try:
                if entity_ids:
                    result = await session.run(
                        f"MATCH (e:{all_labels})-[:MEMBER_OF]->(c:Community) "
                        "WHERE e.id IN $ids "
                        "RETURN DISTINCT c",
                        ids=entity_ids,
                    )
                else:
                    result = await session.run("MATCH (c:Community) RETURN c LIMIT 20")
                records = await result.data()
            except Exception as exc:
                logger.debug("[KnowledgeGraph] Community fetch failed: %s", exc)
                return []

        communities: List[Dict[str, Any]] = []
        for rec in records:
            c = rec["c"]
            entity_id_list = json.loads(c.get("entity_ids", "[]"))
            communities.append(
                {
                    "id": c.get("id", ""),
                    "topic": c.get("topic", ""),
                    "summary": c.get("summary", ""),
                    "entity_ids": entity_id_list,
                    "created_at": c.get("created_at", ""),
                }
            )
        return communities

    async def update_communities(
        self,
        summarize_fn: Callable[[str], Coroutine[Any, Any, str]],
        max_communities: int = 10,
    ) -> int:
        """
        Detect entity communities by relationship co-occurrence and generate
        LLM summaries for each cluster.
        """
        if not self.available:
            return 0

        clusters = await self._detect_clusters()
        if not clusters:
            return 0

        count = 0
        for cluster_entity_ids, cluster_names in clusters[:max_communities]:
            context = await self._build_cluster_context(
                cluster_entity_ids, cluster_names
            )
            if not context:
                continue

            prompt = (
                "Summarize the following cluster of related entities and their "
                "relationships into a concise paragraph. Focus on the key themes, "
                "connections, and implications:\n\n" + context
            )
            try:
                summary = await summarize_fn(prompt)
            except Exception as exc:
                logger.debug("[KnowledgeGraph] Community summarization failed: %s", exc)
                continue

            if not summary or not summary.strip():
                continue

            topic = ", ".join(sorted(cluster_names)[:3])

            community_vector = await self._embed(summary.strip())

            await self._store_community(
                cluster_entity_ids, topic, summary.strip(), community_vector
            )
            count += 1

        logger.debug("[KnowledgeGraph] Updated %d communities.", count)
        return count

    async def _detect_clusters(self) -> List[tuple]:
        """
        Simple community detection: group entities that share >= N relationships.

        Returns list of (entity_id_list, entity_name_set) tuples.
        """
        assert self._driver is not None
        all_labels = "|".join(ENTITY_TYPES)

        # Fetch adjacency from Neo4j — match all factual relationship types
        async with self._driver.session(database=self._database) as session:
            try:
                result = await session.run(
                    f"MATCH (s:{all_labels})-[:{_FACTUAL_REL_CYPHER}]-(t:{all_labels}) "
                    "RETURN s.name AS source, t.name AS target"
                )
                records = await result.data()
            except Exception:
                return []

        if not records:
            return []

        # Build adjacency by entity name
        adjacency: Dict[str, Set[str]] = defaultdict(set)
        for rec in records:
            src = rec["source"]
            tgt = rec["target"]
            if src and tgt:
                adjacency[src].add(tgt)
                adjacency[tgt].add(src)

        # Simple greedy clustering
        visited: Set[str] = set()
        clusters: List[Set[str]] = []

        for entity_name in adjacency:
            if entity_name in visited:
                continue
            cluster: Set[str] = {entity_name}
            queue = [entity_name]
            while queue:
                current = queue.pop(0)
                for neighbor in adjacency.get(current, set()):
                    if neighbor in cluster:
                        continue
                    shared = len(adjacency.get(neighbor, set()) & cluster)
                    if shared >= _COMMUNITY_MIN_SHARED_RELS or len(cluster) == 1:
                        cluster.add(neighbor)
                        queue.append(neighbor)
            if len(cluster) >= 2:
                visited |= cluster
                clusters.append(cluster)

        # Resolve names → entity_ids across all typed labels
        async with self._driver.session(database=self._database) as session:
            try:
                result = await session.run(
                    f"MATCH (e:{all_labels}) RETURN e.name AS name, e.id AS id"
                )
                records = await result.data()
            except Exception:
                return []

        name_to_id: Dict[str, str] = {}
        for rec in records:
            if rec["name"]:
                name_to_id[rec["name"]] = rec["id"]

        final: List[tuple] = []
        for cluster_names in clusters:
            ids = [name_to_id[n] for n in cluster_names if n in name_to_id]
            if len(ids) >= 2:
                final.append((ids, cluster_names))

        return final

    async def _build_cluster_context(
        self, entity_ids: List[str], entity_names: Set[str]
    ) -> str:
        """Build a text context for community summarization."""
        assert self._driver is not None
        all_labels = "|".join(ENTITY_TYPES)
        parts: List[str] = []

        async with self._driver.session(database=self._database) as session:
            # Entity descriptions
            try:
                result = await session.run(
                    f"MATCH (e:{all_labels}) WHERE e.id IN $ids "
                    "RETURN e.description AS desc, e.entity_type AS etype",
                    ids=entity_ids,
                )
                records = await result.data()
                for rec in records:
                    if rec["desc"]:
                        etype = rec.get("etype", "")
                        parts.append(f"Entity [{etype}]: {rec['desc']}")
            except Exception:
                pass

            # Relationships between cluster members
            try:
                result = await session.run(
                    f"MATCH (s:{all_labels})-[r:{_FACTUAL_REL_CYPHER}]->(t:{all_labels}) "
                    "WHERE s.name IN $names AND t.name IN $names "
                    "RETURN s.name AS src, type(r) AS rtype, "
                    "       r.relation_type AS rel, t.name AS tgt",
                    names=list(entity_names),
                )
                records = await result.data()
                for rec in records:
                    rtype = rec.get("rtype", "RELATES_TO")
                    rel_desc = rec.get("rel", rtype)
                    parts.append(
                        f"Relationship: {rec['src']} ─{rtype}→ {rec['tgt']} ({rel_desc})"
                    )
            except Exception:
                pass

        return "\n".join(parts)

    async def _store_community(
        self,
        entity_ids: List[str],
        topic: str,
        summary: str,
        vector: Optional[List[float]] = None,
    ) -> None:
        """Store or update a community summary."""
        assert self._driver is not None
        now = datetime.now().isoformat()
        all_labels = "|".join(ENTITY_TYPES)

        async with self._driver.session(database=self._database) as session:
            # Check if a community with overlapping entities already exists
            try:
                result = await session.run("MATCH (c:Community) RETURN c")
                records = await result.data()
                for rec in records:
                    c = rec["c"]
                    existing_ids = set(json.loads(c.get("entity_ids", "[]")))
                    overlap = len(existing_ids & set(entity_ids))
                    if overlap >= len(entity_ids) * 0.5:
                        # Update existing community
                        props: Dict[str, Any] = {
                            "entity_ids": json.dumps(entity_ids),
                            "topic": topic,
                            "summary": summary,
                            "updated_at": now,
                        }
                        if vector is not None:
                            props["embedding"] = vector
                        await session.run(
                            "MATCH (c:Community {id: $cid}) SET c += $props",
                            cid=c["id"],
                            props=props,
                        )
                        # Rebuild MEMBER_OF edges
                        await session.run(
                            "MATCH (c:Community {id: $cid})<-[r:MEMBER_OF]-() "
                            "DELETE r",
                            cid=c["id"],
                        )
                        await session.run(
                            "MATCH (c:Community {id: $cid}) "
                            "UNWIND $eids AS eid "
                            f"MATCH (e:{all_labels} {{id: eid}}) "
                            "MERGE (e)-[:MEMBER_OF]->(c)",
                            cid=c["id"],
                            eids=entity_ids,
                        )
                        return
            except Exception:
                pass

            # New community
            community_id = str(uuid.uuid4())
            try:
                props = {
                    "id": community_id,
                    "entity_ids": json.dumps(entity_ids),
                    "topic": topic,
                    "summary": summary,
                    "created_at": now,
                }
                if vector is not None:
                    props["embedding"] = vector
                await session.run(
                    "CREATE (c:Community) SET c = $props",
                    props=props,
                )
                # Create MEMBER_OF edges
                await session.run(
                    "MATCH (c:Community {id: $cid}) "
                    "UNWIND $eids AS eid "
                    f"MATCH (e:{all_labels} {{id: eid}}) "
                    "MERGE (e)-[:MEMBER_OF]->(c)",
                    cid=community_id,
                    eids=entity_ids,
                )
            except Exception as exc:
                logger.debug("[KnowledgeGraph] Community store failed: %s", exc)

    # ------------------------------------------------------------------
    # Claim CRUD
    # ------------------------------------------------------------------

    async def store_claim(
        self,
        text: str,
        confidence: float = 0.5,
        status: str = "unverified",
        source_session: str = "",
        step_id: int = 0,
        entity_names: Optional[List[str]] = None,
        document_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Store a Claim node and link it to entities via ASSERTS edges.

        Optionally link to a Document via a SUPPORTS edge.
        """
        if not self.available or not text or not text.strip():
            return {"success": False, "message": "Graph not available or empty text"}

        if status not in CLAIM_STATUSES:
            status = "unverified"

        vector = await self._embed(text.strip())

        assert self._driver is not None
        now = datetime.now().isoformat()
        claim_id = str(uuid.uuid4())
        all_labels = "|".join(ENTITY_TYPES)

        async with self._driver.session(database=self._database) as session:
            # Dedup by vector similarity
            if vector is not None:
                try:
                    result = await session.run(
                        "CALL db.index.vector.queryNodes("
                        "  'claim_embedding_idx', 1, $vector"
                        ") YIELD node, score "
                        "WHERE score >= $threshold "
                        "RETURN node.id AS id, score",
                        vector=vector,
                        threshold=_DEDUP_THRESHOLD,
                    )
                    record = await result.single()
                    if record:
                        return {
                            "success": False,
                            "claim_id": record["id"],
                            "message": (
                                f"Near-duplicate claim exists "
                                f"(similarity={record['score']:.3f})"
                            ),
                        }
                except Exception:
                    pass

            try:
                props: Dict[str, Any] = {
                    "id": claim_id,
                    "text": text.strip(),
                    "confidence": confidence,
                    "status": status,
                    "source_session": source_session,
                    "step_id": step_id,
                    "created_at": now,
                }
                if vector is not None:
                    props["embedding"] = vector

                await session.run(
                    "CREATE (cl:Claim) SET cl = $props",
                    props=props,
                )

                # Link to entities via ASSERTS
                if entity_names:
                    for ename in entity_names:
                        ename = ename.strip()
                        if not ename:
                            continue
                        await session.run(
                            f"MATCH (e:{all_labels} {{name: $name}}) "
                            "MATCH (cl:Claim {id: $cid}) "
                            "MERGE (cl)-[:ASSERTS]->(e)",
                            name=ename,
                            cid=claim_id,
                        )

                # Link to document via SUPPORTS
                if document_id:
                    await session.run(
                        "MATCH (d:Document {id: $did}) "
                        "MATCH (cl:Claim {id: $cid}) "
                        "MERGE (d)-[:SUPPORTS]->(cl)",
                        did=document_id,
                        cid=claim_id,
                    )

                return {"success": True, "claim_id": claim_id}
            except Exception as exc:
                logger.error("[KnowledgeGraph] Claim store failed: %s", exc)
                return {"success": False, "message": str(exc)}

    async def find_claims(
        self,
        query: str = "",
        limit: int = 10,
        entity_name: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Find claims by semantic search, optionally filtered by entity or status."""
        if not self.available:
            return []

        assert self._driver is not None
        all_labels = "|".join(ENTITY_TYPES)

        # If entity_name is provided, use graph traversal instead of vector search
        if entity_name:
            async with self._driver.session(database=self._database) as session:
                try:
                    status_clause = ""
                    params: Dict[str, Any] = {
                        "name": entity_name.strip(),
                        "limit": limit,
                    }
                    if status:
                        status_clause = " AND cl.status = $status"
                        params["status"] = status
                    result = await session.run(
                        f"MATCH (cl:Claim)-[:ASSERTS]->(e:{all_labels} {{name: $name}}) "
                        f"WHERE true{status_clause} "
                        "RETURN cl ORDER BY cl.confidence DESC LIMIT $limit",
                        **params,
                    )
                    records = await result.data()
                except Exception as exc:
                    logger.debug(
                        "[KnowledgeGraph] Claim search (entity) failed: %s", exc
                    )
                    return []

            return [self._unpack_claim(rec["cl"]) for rec in records]

        # Vector search
        if not query:
            # Return recent claims
            async with self._driver.session(database=self._database) as session:
                try:
                    status_clause = ""
                    params = {"limit": limit}
                    if status:
                        status_clause = " WHERE cl.status = $status"
                        params["status"] = status
                    result = await session.run(
                        f"MATCH (cl:Claim){status_clause} "
                        "RETURN cl ORDER BY cl.created_at DESC LIMIT $limit",
                        **params,
                    )
                    records = await result.data()
                except Exception:
                    return []
            return [self._unpack_claim(rec["cl"]) for rec in records]

        query_vec = await self._embed_query(query)
        if query_vec is None:
            return []

        async with self._driver.session(database=self._database) as session:
            try:
                status_clause = ""
                params = {"vector": query_vec, "limit": limit}
                if status:
                    status_clause = " AND node.status = $status"
                    params["status"] = status
                result = await session.run(
                    "CALL db.index.vector.queryNodes("
                    "  'claim_embedding_idx', $limit, $vector"
                    ") YIELD node, score "
                    f"WHERE true{status_clause} "
                    "RETURN node, score ORDER BY score DESC LIMIT $limit",
                    **params,
                )
                records = await result.data()
            except Exception as exc:
                logger.debug("[KnowledgeGraph] Claim vector search failed: %s", exc)
                return []

        return [
            {**self._unpack_claim(rec["node"]), "similarity": round(rec["score"], 4)}
            for rec in records
        ]

    async def update_claim_status(
        self,
        claim_id: str,
        new_status: str,
    ) -> Dict[str, Any]:
        """Update a claim's status."""
        if not self.available:
            return {"success": False, "message": "Graph not available"}
        if new_status not in CLAIM_STATUSES:
            return {"success": False, "message": f"Invalid status: {new_status}"}

        assert self._driver is not None
        async with self._driver.session(database=self._database) as session:
            try:
                result = await session.run(
                    "MATCH (cl:Claim {id: $cid}) "
                    "SET cl.status = $status "
                    "RETURN 'ok' AS result",
                    cid=claim_id,
                    status=new_status,
                )
                record = await result.single()
                if record:
                    return {"success": True}
                return {"success": False, "message": "Claim not found"}
            except Exception as exc:
                logger.debug("[KnowledgeGraph] Claim status update failed: %s", exc)
                return {"success": False, "message": str(exc)}

    @staticmethod
    def _unpack_claim(node: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": node.get("id", ""),
            "text": node.get("text", ""),
            "confidence": node.get("confidence", 0.0),
            "status": node.get("status", "unverified"),
            "source_session": node.get("source_session", ""),
            "step_id": node.get("step_id", 0),
            "created_at": node.get("created_at", ""),
        }

    # ------------------------------------------------------------------
    # Document CRUD (subsumes old Source node)
    # ------------------------------------------------------------------

    async def store_document(
        self,
        url: str,
        title: str = "",
        content_summary: str = "",
        doc_type: str = "article",
        credibility_score: float = 0.5,
        session_id: str = "",
    ) -> Dict[str, Any]:
        """Store or update a Document node (unique by URL).

        On conflict, title is updated only if the new value is non-empty;
        credibility_score is updated to the higher of the two values.
        """
        if not self.available:
            return {"success": False, "message": "Graph not available"}
        if not url or not url.strip().startswith("http"):
            return {"success": False, "message": "Invalid URL"}

        from urllib.parse import urlparse

        domain = urlparse(url.strip()).netloc or ""
        now = datetime.now().isoformat()
        doc_id = str(uuid.uuid4())

        # Embed the content summary for semantic search
        embed_text = (
            content_summary.strip() if content_summary.strip() else title.strip()
        )
        vector = await self._embed(embed_text) if embed_text else None

        assert self._driver is not None
        async with self._driver.session(database=self._database) as session:
            try:
                # Build ON CREATE / ON MATCH dynamically to handle embedding
                create_props = (
                    "s.id = $doc_id, s.title = $title, "
                    "s.content_summary = $summary, "
                    "s.doc_type = $doc_type, "
                    "s.credibility_score = $credibility_score, "
                    "s.domain = $domain, "
                    "s.session_id = $session_id, "
                    "s.first_seen = $now, s.retrieved_at = $now"
                )
                match_set = (
                    "s.title = CASE WHEN $title <> '' THEN $title ELSE s.title END, "
                    "s.credibility_score = CASE "
                    "  WHEN $credibility_score > s.credibility_score "
                    "  THEN $credibility_score ELSE s.credibility_score END, "
                    "s.content_summary = CASE "
                    "  WHEN $summary <> '' THEN $summary ELSE s.content_summary END"
                )
                if vector is not None:
                    create_props += ", s.embedding = $vector"
                    match_set += ", s.embedding = $vector"

                params: Dict[str, Any] = {
                    "url": url.strip(),
                    "doc_id": doc_id,
                    "title": title.strip(),
                    "summary": content_summary.strip(),
                    "doc_type": doc_type,
                    "credibility_score": credibility_score,
                    "domain": domain,
                    "session_id": session_id,
                    "now": now,
                }
                if vector is not None:
                    params["vector"] = vector

                await session.run(
                    f"MERGE (s:Document {{url: $url}}) "
                    f"ON CREATE SET {create_props} "
                    f"ON MATCH SET {match_set}",
                    **params,
                )
                return {"success": True, "url": url.strip(), "document_id": doc_id}
            except Exception as exc:
                logger.error("[KnowledgeGraph] Document store failed: %s", exc)
                return {"success": False, "message": str(exc)}

    async def find_documents(
        self,
        query: str = "",
        limit: int = 10,
        doc_type: Optional[str] = None,
        min_credibility: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Find documents by semantic search, optionally filtered."""
        if not self.available:
            return []

        assert self._driver is not None

        if not query:
            # Return recent documents
            where_parts: List[str] = []
            params: Dict[str, Any] = {"limit": limit}
            if doc_type:
                where_parts.append("d.doc_type = $doc_type")
                params["doc_type"] = doc_type
            if min_credibility is not None:
                where_parts.append("d.credibility_score >= $min_cred")
                params["min_cred"] = min_credibility
            where_clause = " WHERE " + " AND ".join(where_parts) if where_parts else ""

            async with self._driver.session(database=self._database) as session:
                try:
                    result = await session.run(
                        f"MATCH (d:Document){where_clause} "
                        "RETURN d ORDER BY d.retrieved_at DESC LIMIT $limit",
                        **params,
                    )
                    records = await result.data()
                except Exception:
                    return []
            return [self._unpack_document(rec["d"]) for rec in records]

        query_vec = await self._embed_query(query)
        if query_vec is None:
            return []

        filter_parts: List[str] = []
        params = {"vector": query_vec, "limit": limit}
        if doc_type:
            filter_parts.append("node.doc_type = $doc_type")
            params["doc_type"] = doc_type
        if min_credibility is not None:
            filter_parts.append("node.credibility_score >= $min_cred")
            params["min_cred"] = min_credibility
        filter_clause = (" AND " + " AND ".join(filter_parts)) if filter_parts else ""

        async with self._driver.session(database=self._database) as session:
            try:
                result = await session.run(
                    "CALL db.index.vector.queryNodes("
                    "  'document_embedding_idx', $limit, $vector"
                    ") YIELD node, score "
                    f"WHERE true{filter_clause} "
                    "RETURN node, score ORDER BY score DESC LIMIT $limit",
                    **params,
                )
                records = await result.data()
            except Exception as exc:
                logger.debug("[KnowledgeGraph] Document vector search failed: %s", exc)
                return []

        return [
            {**self._unpack_document(rec["node"]), "similarity": round(rec["score"], 4)}
            for rec in records
        ]

    async def link_document_to_entity(
        self,
        document_url: str,
        entity_name: str,
        relationship_label: str = "MENTIONS",
    ) -> Dict[str, Any]:
        """Create a typed edge from a Document to an entity node."""
        if not self.available:
            return {"success": False, "message": "Graph not available"}

        # Validate relationship label
        valid_labels = {
            "MENTIONS",
            "SUPPORTS",
            "REFUTES",
            "AUTHORED_BY",
            "SOURCED_FROM",
        }
        if relationship_label not in valid_labels:
            relationship_label = "MENTIONS"

        assert self._driver is not None
        all_labels = "|".join(ENTITY_TYPES)

        async with self._driver.session(database=self._database) as session:
            try:
                result = await session.run(
                    f"MATCH (e:{all_labels} {{name: $name}}) "
                    "MATCH (d:Document {url: $url}) "
                    f"MERGE (e)-[:{relationship_label}]->(d) "
                    "RETURN 'ok' AS result",
                    name=entity_name.strip(),
                    url=document_url.strip(),
                )
                record = await result.single()
                if record:
                    return {"success": True}
                return {"success": False, "message": "Entity or document not found"}
            except Exception as exc:
                logger.debug("[KnowledgeGraph] Document-entity link failed: %s", exc)
                return {"success": False, "message": str(exc)}

    @staticmethod
    def _unpack_document(node: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": node.get("id", ""),
            "url": node.get("url", ""),
            "title": node.get("title", ""),
            "content_summary": node.get("content_summary", ""),
            "doc_type": node.get("doc_type", ""),
            "credibility_score": node.get("credibility_score", 0.0),
            "domain": node.get("domain", ""),
            "session_id": node.get("session_id", ""),
            "first_seen": node.get("first_seen", ""),
            "retrieved_at": node.get("retrieved_at", ""),
        }

    # ------------------------------------------------------------------
    # Deprecated Source methods (thin wrappers for backward compat)
    # ------------------------------------------------------------------

    async def store_source(
        self,
        url: str,
        title: str = "",
        credibility_score: float = 0.5,
    ) -> Dict[str, Any]:
        """Deprecated — use store_document() instead."""
        logger.debug(
            "[KnowledgeGraph] store_source() is deprecated — use store_document()"
        )
        return await self.store_document(
            url=url, title=title, credibility_score=credibility_score
        )

    async def link_to_source(
        self,
        entity_name: str,
        source_url: str,
    ) -> Dict[str, Any]:
        """Deprecated — use link_document_to_entity() instead."""
        logger.debug(
            "[KnowledgeGraph] link_to_source() is deprecated — "
            "use link_document_to_entity()"
        )
        return await self.link_document_to_entity(
            document_url=source_url,
            entity_name=entity_name,
            relationship_label="SOURCED_FROM",
        )

    async def get_provenance(
        self,
        entity_names: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Return Document nodes linked to the given entities via SOURCED_FROM."""
        if not self.available:
            return []

        assert self._driver is not None
        all_labels = "|".join(ENTITY_TYPES)

        async with self._driver.session(database=self._database) as session:
            try:
                if entity_names:
                    result = await session.run(
                        f"MATCH (e:{all_labels})-[:SOURCED_FROM]->(s:Document) "
                        "WHERE e.name IN $names "
                        "RETURN e.name AS entity_name, s.url AS source_url, "
                        "       s.title AS source_title, "
                        "       s.credibility_score AS credibility_score, "
                        "       s.domain AS domain",
                        names=entity_names,
                    )
                else:
                    result = await session.run(
                        f"MATCH (e:{all_labels})-[:SOURCED_FROM]->(s:Document) "
                        "RETURN e.name AS entity_name, s.url AS source_url, "
                        "       s.title AS source_title, "
                        "       s.credibility_score AS credibility_score, "
                        "       s.domain AS domain "
                        "LIMIT 50"
                    )
                records = await result.data()
            except Exception as exc:
                logger.debug("[KnowledgeGraph] Provenance query failed: %s", exc)
                return []

        return [
            {
                "entity_name": rec.get("entity_name", ""),
                "source_url": rec.get("source_url", ""),
                "source_title": rec.get("source_title", ""),
                "credibility_score": rec.get("credibility_score", 0.0),
                "domain": rec.get("domain", ""),
            }
            for rec in records
        ]

    # ------------------------------------------------------------------
    # Graph-aware recall (the main query interface)
    # ------------------------------------------------------------------

    async def recall_graph_context(
        self,
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
        """
        Build a structured text context from the knowledge graph for a query.

        1. Vector-search entities relevant to the query
        2. Traverse 1–N hops of relationships from those entities
        3. Retrieve community summaries for the entity clusters
        4. Optionally append claims, documents, contradictions, provenance
        5. Return formatted text
        """
        if not self.available:
            return ""

        # Step 1: find seed entities
        entities = await self.find_entities(
            query, limit=entity_limit, node_types=node_types
        )
        if not entities:
            return ""

        entity_ids = [e["id"] for e in entities]
        entity_names = [e["name"] for e in entities]

        # Step 2: traverse relationships
        relationships = await self.get_relationships(
            entity_names=entity_names, max_hops=max_hops
        )

        # Step 3: get community summaries
        communities = await self.get_communities(entity_ids=entity_ids)

        # Step 4: format
        parts: List[str] = []

        if entities:
            parts.append("KNOWN ENTITIES:")
            for e in entities:
                label = e.get("entity_label", "Concept")
                line = f"  [{label}] {e['name']}"
                # Type-specific detail
                detail_parts: List[str] = []
                if e.get("affiliation"):
                    detail_parts.append(e["affiliation"])
                if e.get("tech_category"):
                    detail_parts.append(e["tech_category"])
                if e.get("maturity"):
                    detail_parts.append(e["maturity"])
                if e.get("org_type"):
                    detail_parts.append(e["org_type"])
                if e.get("start_date"):
                    date_str = e["start_date"]
                    if e.get("end_date"):
                        date_str += f" → {e['end_date']}"
                    detail_parts.append(date_str)
                if e.get("value") is not None and e.get("unit"):
                    detail_parts.append(f"{e['value']} {e['unit']}")
                if detail_parts:
                    line += f" ({', '.join(detail_parts)})"
                if e.get("mention_count", 1) > 1:
                    line += f" — seen {e['mention_count']}x"
                if e.get("ancestors"):
                    line += f" [is-a: {', '.join(e['ancestors'][:2])}]"
                parts.append(line)

        if relationships:
            parts.append("\nKNOWN RELATIONSHIPS:")
            seen_triples: Set[str] = set()
            for r in relationships:
                # Filter by minimum confidence when specified
                if min_confidence > 0.0 and r.get("confidence", 1.0) < min_confidence:
                    continue
                rel_label = r.get("relationship_label", "RELATES_TO")
                triple = f"{r['source_entity']} ─{rel_label}→ {r['target_entity']}"
                if triple in seen_triples:
                    continue
                seen_triples.add(triple)
                conf = r.get("confidence", 1.0)
                conf_str = f" (conf: {conf:.2f})" if conf < 0.8 else ""
                parts.append(f"  • {triple}{conf_str}")

        # Claims
        if include_claims and entity_names:
            try:
                all_claims: List[Dict[str, Any]] = []
                for ename in entity_names[:3]:
                    claims = await self.find_claims(entity_name=ename, limit=3)
                    all_claims.extend(claims)
                # Dedup by id
                seen_claim_ids: Set[str] = set()
                unique_claims: List[Dict[str, Any]] = []
                for c in all_claims:
                    if c["id"] not in seen_claim_ids:
                        seen_claim_ids.add(c["id"])
                        unique_claims.append(c)
                if unique_claims:
                    parts.append("\nCLAIMS:")
                    for c in unique_claims[:5]:
                        status_tag = c.get("status", "unverified")
                        conf = c.get("confidence", 0.0)
                        parts.append(
                            f"  • \"{c['text'][:120]}\" ({status_tag}, conf: {conf:.2f})"
                        )
            except Exception as exc:
                logger.debug("[KnowledgeGraph] Claim recall skipped: %s", exc)

        if communities:
            parts.append("\nTHEMATIC CLUSTERS:")
            for c in communities:
                parts.append(f"  [{c['topic']}] {c['summary']}")

        # Optional: contradictions
        if include_contradictions:
            try:
                contradictions = await self.find_contradictions(
                    entity_names=entity_names, limit=5
                )
                if contradictions:
                    parts.append("\nKNOWN CONTRADICTIONS:")
                    for c in contradictions:
                        explanation = c.get("explanation") or "conflicting claims"
                        parts.append(
                            f"  ⚠ {c['entity_a']} / {c['entity_b']}: {explanation}"
                        )
            except Exception as exc:
                logger.debug("[KnowledgeGraph] Contradiction recall skipped: %s", exc)

        # Optional: provenance / documents
        if include_provenance or include_documents:
            try:
                provenance = await self.get_provenance(entity_names=entity_names[:3])
                if provenance:
                    parts.append("\nSOURCE DOCUMENTS:")
                    for p in provenance[:5]:
                        cred = p.get("credibility_score", 0.5)
                        parts.append(
                            f"  • {p['entity_name']} ← {p['source_url']}"
                            f" (credibility: {cred:.1f})"
                        )
            except Exception as exc:
                logger.debug("[KnowledgeGraph] Provenance recall skipped: %s", exc)

        return "\n".join(parts) if parts else ""

    # ------------------------------------------------------------------
    # Stats (for testing / monitoring)
    # ------------------------------------------------------------------

    async def stats(self) -> Dict[str, int]:
        """Return counts of all graph elements."""
        if not self.available:
            return {
                "entities": 0,
                "relationships": 0,
                "communities": 0,
                "contradictions": 0,
                "documents": 0,
                "claims": 0,
                "hierarchies": 0,
            }

        assert self._driver is not None
        all_labels = "|".join(ENTITY_TYPES)

        async with self._driver.session(database=self._database) as session:
            try:
                result = await session.run(
                    f"CALL {{ MATCH (e:{all_labels}) RETURN count(e) AS c }} "
                    "WITH c AS entities "
                    f"CALL {{ MATCH ()-[r:{_FACTUAL_REL_CYPHER}]->() RETURN count(r) AS c }} "
                    "WITH entities, c AS relationships "
                    "CALL { MATCH (co:Community) RETURN count(co) AS c } "
                    "WITH entities, relationships, c AS communities "
                    "CALL { MATCH ()-[r:CONTRADICTS]->() RETURN count(r) AS c } "
                    "WITH entities, relationships, communities, c AS contradictions "
                    "CALL { MATCH (d:Document) RETURN count(d) AS c } "
                    "WITH entities, relationships, communities, contradictions, c AS documents "
                    "CALL { MATCH (cl:Claim) RETURN count(cl) AS c } "
                    "WITH entities, relationships, communities, contradictions, documents, c AS claims "
                    "CALL { MATCH ()-[r:IS_A]->() RETURN count(r) AS c } "
                    "RETURN entities, relationships, communities, contradictions, "
                    "       documents, claims, c AS hierarchies"
                )
                record = await result.single()
                if record:
                    return {
                        "entities": record["entities"],
                        "relationships": record["relationships"],
                        "communities": record["communities"],
                        "contradictions": record["contradictions"],
                        "documents": record["documents"],
                        "claims": record["claims"],
                        "hierarchies": record["hierarchies"],
                    }
            except Exception as exc:
                logger.debug("[KnowledgeGraph] Stats query failed: %s", exc)

        return {
            "entities": 0,
            "relationships": 0,
            "communities": 0,
            "contradictions": 0,
            "documents": 0,
            "claims": 0,
            "hierarchies": 0,
        }

    # ------------------------------------------------------------------
    # Hierarchy (IS_A)
    # ------------------------------------------------------------------

    async def store_hierarchy(
        self,
        child_name: str,
        parent_name: str,
    ) -> Dict[str, Any]:
        """Create an IS_A edge from *child_name* to *parent_name* (idempotent).

        Both entities must already exist in the graph.  If either is missing
        the call is silently ignored (returns success=False).
        """
        if not self.available:
            return {"success": False, "message": "Graph not available"}
        if not child_name.strip() or not parent_name.strip():
            return {"success": False, "message": "Both names required"}
        if child_name.strip().lower() == parent_name.strip().lower():
            return {"success": False, "message": "Self-hierarchy not allowed"}

        assert self._driver is not None
        all_labels = "|".join(ENTITY_TYPES)

        async with self._driver.session(database=self._database) as session:
            try:
                result = await session.run(
                    f"MATCH (child:{all_labels} {{name: $child}}) "
                    f"MATCH (parent:{all_labels} {{name: $parent}}) "
                    "MERGE (child)-[:IS_A]->(parent) "
                    "RETURN 'ok' AS result",
                    child=child_name.strip(),
                    parent=parent_name.strip(),
                )
                record = await result.single()
                if record:
                    return {"success": True}
                return {"success": False, "message": "One or both entities not found"}
            except Exception as exc:
                logger.debug("[KnowledgeGraph] Hierarchy store failed: %s", exc)
                return {"success": False, "message": str(exc)}

    # ------------------------------------------------------------------
    # Contradiction tracking
    # ------------------------------------------------------------------

    async def store_contradiction(
        self,
        rel_id_a: str,
        rel_id_b: str,
        explanation: str = "",
        session_id: str = "",
    ) -> Dict[str, Any]:
        """Record that two relationships contradict each other.

        Creates a CONTRADICTS edge between the *source* entities of the two
        relationships, carrying both relationship IDs and an explanation.
        """
        if not self.available:
            return {"success": False, "message": "Graph not available"}

        assert self._driver is not None
        now = datetime.now().isoformat()
        cid = str(uuid.uuid4())

        # Match any factual relationship type for both IDs
        async with self._driver.session(database=self._database) as session:
            try:
                result = await session.run(
                    f"MATCH ()-[r1:{_FACTUAL_REL_CYPHER} {{id: $id_a}}]->() "
                    f"MATCH ()-[r2:{_FACTUAL_REL_CYPHER} {{id: $id_b}}]->() "
                    "WITH startNode(r1) AS e1, startNode(r2) AS e2 "
                    "CREATE (e1)-[:CONTRADICTS { "
                    "  id: $cid, rel_id_a: $id_a, rel_id_b: $id_b, "
                    "  explanation: $explanation, session_id: $session_id, "
                    "  created_at: $now "
                    "}]->(e2) "
                    "RETURN 'ok' AS result",
                    id_a=rel_id_a,
                    id_b=rel_id_b,
                    cid=cid,
                    explanation=explanation.strip(),
                    session_id=session_id,
                    now=now,
                )
                record = await result.single()
                if record:
                    return {"success": True, "contradiction_id": cid}
                return {
                    "success": False,
                    "message": "One or both relationship IDs not found",
                }
            except Exception as exc:
                logger.error("[KnowledgeGraph] Contradiction store failed: %s", exc)
                return {"success": False, "message": str(exc)}

    async def find_contradictions(
        self,
        entity_names: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Return CONTRADICTS edges involving the given entities (or all if None)."""
        if not self.available:
            return []

        assert self._driver is not None
        all_labels = "|".join(ENTITY_TYPES)

        async with self._driver.session(database=self._database) as session:
            try:
                if entity_names:
                    result = await session.run(
                        f"MATCH (e1:{all_labels})-[c:CONTRADICTS]->(e2:{all_labels}) "
                        "WHERE e1.name IN $names OR e2.name IN $names "
                        f"OPTIONAL MATCH ()-[r1:{_FACTUAL_REL_CYPHER} {{id: c.rel_id_a}}]->() "
                        f"OPTIONAL MATCH ()-[r2:{_FACTUAL_REL_CYPHER} {{id: c.rel_id_b}}]->() "
                        "RETURN e1.name AS entity_a, e2.name AS entity_b, "
                        "       c.explanation AS explanation, c.id AS id, "
                        "       r1.relation_type AS rel_a_type, "
                        "       r1.evidence AS rel_a_evidence, "
                        "       r2.relation_type AS rel_b_type, "
                        "       r2.evidence AS rel_b_evidence "
                        "LIMIT $limit",
                        names=entity_names,
                        limit=limit,
                    )
                else:
                    result = await session.run(
                        f"MATCH (e1:{all_labels})-[c:CONTRADICTS]->(e2:{all_labels}) "
                        f"OPTIONAL MATCH ()-[r1:{_FACTUAL_REL_CYPHER} {{id: c.rel_id_a}}]->() "
                        f"OPTIONAL MATCH ()-[r2:{_FACTUAL_REL_CYPHER} {{id: c.rel_id_b}}]->() "
                        "RETURN e1.name AS entity_a, e2.name AS entity_b, "
                        "       c.explanation AS explanation, c.id AS id, "
                        "       r1.relation_type AS rel_a_type, "
                        "       r1.evidence AS rel_a_evidence, "
                        "       r2.relation_type AS rel_b_type, "
                        "       r2.evidence AS rel_b_evidence "
                        "LIMIT $limit",
                        limit=limit,
                    )
                records = await result.data()
            except Exception as exc:
                logger.debug("[KnowledgeGraph] Contradiction query failed: %s", exc)
                return []

        return [
            {
                "id": rec.get("id", ""),
                "entity_a": rec.get("entity_a", ""),
                "entity_b": rec.get("entity_b", ""),
                "explanation": rec.get("explanation", ""),
                "rel_a_type": rec.get("rel_a_type", ""),
                "rel_a_evidence": rec.get("rel_a_evidence", ""),
                "rel_b_type": rec.get("rel_b_type", ""),
                "rel_b_evidence": rec.get("rel_b_evidence", ""),
            }
            for rec in records
        ]

    # ------------------------------------------------------------------
    # Temporal / recency queries
    # ------------------------------------------------------------------

    async def recent_entities(
        self,
        since_date: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Return entities added or confirmed since the given ISO-8601 date string."""
        if not self.available:
            return []

        assert self._driver is not None
        all_labels = "|".join(ENTITY_TYPES)

        async with self._driver.session(database=self._database) as session:
            try:
                if since_date:
                    result = await session.run(
                        f"MATCH (e:{all_labels}) "
                        "WHERE e.last_confirmed >= $since_date "
                        "RETURN e ORDER BY e.last_confirmed DESC LIMIT $limit",
                        since_date=since_date,
                        limit=limit,
                    )
                else:
                    result = await session.run(
                        f"MATCH (e:{all_labels}) "
                        "RETURN e ORDER BY e.last_confirmed DESC LIMIT $limit",
                        limit=limit,
                    )
                records = await result.data()
            except Exception as exc:
                logger.debug("[KnowledgeGraph] recent_entities failed: %s", exc)
                return []

        return [
            {
                "id": rec["e"].get("id", ""),
                "name": rec["e"].get("name", ""),
                "entity_type": rec["e"].get("entity_type", ""),
                "entity_label": resolve_entity_label(
                    rec["e"].get("entity_type", "concept")
                ),
                "mention_count": rec["e"].get("mention_count", 1),
                "last_confirmed": rec["e"].get("last_confirmed", ""),
            }
            for rec in records
        ]

    async def recent_relationships(
        self,
        since_date: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Return factual edges created or confirmed since *since_date*."""
        if not self.available:
            return []

        assert self._driver is not None
        all_labels = "|".join(ENTITY_TYPES)

        async with self._driver.session(database=self._database) as session:
            try:
                if since_date:
                    result = await session.run(
                        f"MATCH (s:{all_labels})-[r:{_FACTUAL_REL_CYPHER}]->(t:{all_labels}) "
                        "WHERE r.last_confirmed >= $since_date "
                        "RETURN s.name AS source_entity, r.relation_type AS relation_type, "
                        "       type(r) AS relationship_label, "
                        "       t.name AS target_entity, r.confidence AS confidence, "
                        "       r.last_confirmed AS last_confirmed "
                        "ORDER BY r.last_confirmed DESC LIMIT $limit",
                        since_date=since_date,
                        limit=limit,
                    )
                else:
                    result = await session.run(
                        f"MATCH (s:{all_labels})-[r:{_FACTUAL_REL_CYPHER}]->(t:{all_labels}) "
                        "RETURN s.name AS source_entity, r.relation_type AS relation_type, "
                        "       type(r) AS relationship_label, "
                        "       t.name AS target_entity, r.confidence AS confidence, "
                        "       r.last_confirmed AS last_confirmed "
                        "ORDER BY r.last_confirmed DESC LIMIT $limit",
                        limit=limit,
                    )
                records = await result.data()
            except Exception as exc:
                logger.debug("[KnowledgeGraph] recent_relationships failed: %s", exc)
                return []

        return [
            {
                "source_entity": rec.get("source_entity", ""),
                "relation_type": rec.get("relation_type", ""),
                "relationship_label": rec.get("relationship_label", "RELATES_TO"),
                "target_entity": rec.get("target_entity", ""),
                "confidence": rec.get("confidence", 0.0),
                "last_confirmed": rec.get("last_confirmed", ""),
            }
            for rec in records
        ]

    async def session_diff(
        self,
        session_id: str,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Return entities and relationships created during a specific session."""
        if not self.available or not session_id:
            return {"entities": [], "relationships": []}

        assert self._driver is not None
        all_labels = "|".join(ENTITY_TYPES)

        async with self._driver.session(database=self._database) as session:
            try:
                ent_result = await session.run(
                    f"MATCH (e:{all_labels}) "
                    "WHERE $session_id IN e.source_sessions OR "
                    "      e.source_sessions CONTAINS $session_id "
                    "RETURN e.name AS name, e.entity_type AS entity_type, "
                    "       e.description AS description",
                    session_id=session_id,
                )
                entities = [
                    {
                        "name": r.get("name", ""),
                        "entity_type": r.get("entity_type", ""),
                        "entity_label": resolve_entity_label(
                            r.get("entity_type", "concept")
                        ),
                        "description": r.get("description", ""),
                    }
                    for r in await ent_result.data()
                ]

                rel_result = await session.run(
                    f"MATCH (s:{all_labels})-[r:{_FACTUAL_REL_CYPHER}]->(t:{all_labels}) "
                    "WHERE r.session_id = $session_id "
                    "RETURN s.name AS source_entity, r.relation_type AS relation_type, "
                    "       type(r) AS relationship_label, "
                    "       t.name AS target_entity, r.confidence AS confidence",
                    session_id=session_id,
                )
                relationships = [
                    {
                        "source_entity": r.get("source_entity", ""),
                        "relation_type": r.get("relation_type", ""),
                        "relationship_label": r.get("relationship_label", "RELATES_TO"),
                        "target_entity": r.get("target_entity", ""),
                        "confidence": r.get("confidence", 0.0),
                    }
                    for r in await rel_result.data()
                ]
            except Exception as exc:
                logger.debug("[KnowledgeGraph] session_diff failed: %s", exc)
                return {"entities": [], "relationships": []}

        return {"entities": entities, "relationships": relationships}

    # ------------------------------------------------------------------
    # Graph path queries
    # ------------------------------------------------------------------

    async def find_paths(
        self,
        source_name: str,
        target_name: str,
        max_depth: int = 4,
    ) -> List[Dict[str, Any]]:
        """Find shortest paths between two named entities.

        Traverses all factual relationship types plus IS_A.
        """
        if not self.available:
            return []
        if not source_name.strip() or not target_name.strip():
            return []
        max_depth = min(max_depth, 6)

        assert self._driver is not None
        all_labels = "|".join(ENTITY_TYPES)

        async with self._driver.session(database=self._database) as session:
            try:
                result = await session.run(
                    f"MATCH (src:{all_labels} {{name: $source}}), "
                    f"      (tgt:{all_labels} {{name: $target}}) "
                    "MATCH p = shortestPath("
                    f"  (src)-[:{_FACTUAL_REL_CYPHER}|IS_A*1.."
                    + str(max_depth)
                    + "]-(tgt)"
                    ") "
                    "RETURN [n IN nodes(p) | n.name] AS node_names, "
                    "       length(p) AS path_length "
                    "LIMIT 5",
                    source=source_name.strip(),
                    target=target_name.strip(),
                )
                records = await result.data()
            except Exception as exc:
                logger.debug("[KnowledgeGraph] find_paths failed: %s", exc)
                return []

        return [
            {
                "node_names": rec.get("node_names", []),
                "path_length": rec.get("path_length", 0),
            }
            for rec in records
        ]

    async def find_common_neighbors(
        self,
        entity_names: List[str],
        min_shared: int = 2,
    ) -> List[Dict[str, Any]]:
        """Return entities connected to at least *min_shared* of the named entities."""
        if not self.available or not entity_names:
            return []

        assert self._driver is not None
        all_labels = "|".join(ENTITY_TYPES)

        async with self._driver.session(database=self._database) as session:
            try:
                result = await session.run(
                    f"MATCH (seed:{all_labels})-[:{_FACTUAL_REL_CYPHER}]-(common:{all_labels}) "
                    "WHERE seed.name IN $names AND NOT common.name IN $names "
                    "WITH common, count(DISTINCT seed) AS shared_count "
                    "WHERE shared_count >= $min_shared "
                    "RETURN common.name AS name, common.entity_type AS entity_type, "
                    "       common.description AS description, shared_count "
                    "ORDER BY shared_count DESC LIMIT 10",
                    names=entity_names,
                    min_shared=min_shared,
                )
                records = await result.data()
            except Exception as exc:
                logger.debug("[KnowledgeGraph] find_common_neighbors failed: %s", exc)
                return []

        return [
            {
                "name": rec.get("name", ""),
                "entity_type": rec.get("entity_type", ""),
                "description": rec.get("description", ""),
                "shared_connections": rec.get("shared_count", 0),
            }
            for rec in records
        ]

    # ------------------------------------------------------------------
    # Confidence decay
    # ------------------------------------------------------------------

    async def decay_confidence(
        self,
        half_life_days: int = 30,
    ) -> int:
        """Exponentially decay the confidence of factual edges not confirmed recently.

        Applies to ALL typed factual relationship labels, not just RELATES_TO.
        """
        if not self.available or half_life_days <= 0:
            return 0

        assert self._driver is not None
        total_updated = 0

        async with self._driver.session(database=self._database) as session:
            for rel_type in _FACTUAL_REL_TYPES:
                try:
                    result = await session.run(
                        f"MATCH ()-[r:{rel_type}]->() "
                        "WHERE r.last_confirmed IS NOT NULL "
                        "WITH r, "
                        "     toInteger("
                        "       (datetime().epochSeconds "
                        "        - datetime(r.last_confirmed).epochSeconds) / 86400"
                        "     ) AS days_old "
                        "WHERE days_old > 0 "
                        "WITH r, days_old, "
                        "     r.confidence * exp(-0.693 * toFloat(days_old) / $half_life) "
                        "     AS decayed "
                        "SET r.confidence = CASE WHEN decayed < 0.01 THEN 0.01 ELSE decayed END "
                        "RETURN count(r) AS updated",
                        half_life=float(half_life_days),
                    )
                    record = await result.single()
                    if record:
                        total_updated += record["updated"]
                except Exception as exc:
                    logger.debug(
                        "[KnowledgeGraph] decay_confidence failed for %s: %s",
                        rel_type,
                        exc,
                    )

        return total_updated

    # ------------------------------------------------------------------
    # Graph pruning
    # ------------------------------------------------------------------

    async def prune(
        self,
        min_confidence: float = 0.1,
        max_age_days: int = 180,
        dry_run: bool = True,
    ) -> Dict[str, int]:
        """Remove stale, low-confidence graph elements.

        Prunes in four passes:

        1. **Stale relationships** — factual edges where confidence has
           decayed below *min_confidence* OR where ``last_confirmed`` is more
           than *max_age_days* old.
        2. **Orphaned entities** — typed entity nodes left with no factual,
           IS_A, SOURCED_FROM, or MEMBER_OF edges after pass 1.
        3. **Dangling CONTRADICTS edges** — CONTRADICTS edges whose referenced
           relationship IDs no longer exist in the graph.
        4. **Orphaned claims** — Claim nodes with no ASSERTS edges.
        """
        if not self.available:
            return {"relationships": 0, "entities": 0, "contradictions": 0, "claims": 0}

        assert self._driver is not None
        results: Dict[str, int] = {
            "relationships": 0,
            "entities": 0,
            "contradictions": 0,
            "claims": 0,
        }
        all_labels = "|".join(ENTITY_TYPES)

        async with self._driver.session(database=self._database) as session:
            # ── Pass 1: stale relationships (all factual types) ────────────
            for rel_type in _FACTUAL_REL_TYPES:
                try:
                    age_clause = ""
                    params: Dict[str, Any] = {"min_conf": min_confidence}
                    if max_age_days > 0:
                        params["max_age_days"] = float(max_age_days)
                        age_clause = (
                            " OR ("
                            "r.last_confirmed IS NOT NULL AND "
                            "toInteger("
                            "  (datetime().epochSeconds "
                            "   - datetime(r.last_confirmed).epochSeconds) / 86400"
                            ") > $max_age_days"
                            ")"
                        )

                    count_q = (
                        f"MATCH ()-[r:{rel_type}]->() "
                        f"WHERE r.confidence < $min_conf{age_clause} "
                        "RETURN count(r) AS cnt"
                    )
                    cnt_result = await session.run(count_q, **params)
                    cnt_record = await cnt_result.single()
                    rel_count = cnt_record["cnt"] if cnt_record else 0
                    results["relationships"] += rel_count

                    if not dry_run and rel_count > 0:
                        delete_q = (
                            f"MATCH ()-[r:{rel_type}]->() "
                            f"WHERE r.confidence < $min_conf{age_clause} "
                            "DELETE r"
                        )
                        await session.run(delete_q, **params)
                except Exception as exc:
                    logger.debug(
                        "[KnowledgeGraph] prune (rel %s) failed: %s", rel_type, exc
                    )

            if not dry_run and results["relationships"] > 0:
                logger.info(
                    "[KnowledgeGraph] Pruned %d stale relationships.",
                    results["relationships"],
                )

            # ── Pass 2: orphaned entities ─────────────────────────────────
            try:
                orphan_q = (
                    f"MATCH (e:{all_labels}) "
                    f"WHERE NOT (e)-[:{_FACTUAL_REL_CYPHER}]-() "
                    "  AND NOT (e)-[:IS_A]-() "
                    "  AND NOT (e)-[:SOURCED_FROM]->() "
                    "  AND NOT (e)-[:MEMBER_OF]->() "
                    "RETURN count(e) AS cnt"
                )
                cnt_result = await session.run(orphan_q)
                cnt_record = await cnt_result.single()
                ent_count = cnt_record["cnt"] if cnt_record else 0
                results["entities"] = ent_count

                if not dry_run and ent_count > 0:
                    await session.run(
                        f"MATCH (e:{all_labels}) "
                        f"WHERE NOT (e)-[:{_FACTUAL_REL_CYPHER}]-() "
                        "  AND NOT (e)-[:IS_A]-() "
                        "  AND NOT (e)-[:SOURCED_FROM]->() "
                        "  AND NOT (e)-[:MEMBER_OF]->() "
                        "DELETE e"
                    )
                    logger.info(
                        "[KnowledgeGraph] Pruned %d orphaned entities.", ent_count
                    )
            except Exception as exc:
                logger.debug("[KnowledgeGraph] prune (entities) failed: %s", exc)

            # ── Pass 3: dangling CONTRADICTS edges ─────────────────────────
            try:
                dangle_q = (
                    f"MATCH (e1:{all_labels})-[c:CONTRADICTS]->(e2:{all_labels}) "
                    f"WHERE NOT EXISTS {{ MATCH ()-[r:{_FACTUAL_REL_CYPHER} {{id: c.rel_id_a}}]->() }} "
                    f"   OR NOT EXISTS {{ MATCH ()-[r:{_FACTUAL_REL_CYPHER} {{id: c.rel_id_b}}]->() }} "
                    "RETURN count(c) AS cnt"
                )
                cnt_result = await session.run(dangle_q)
                cnt_record = await cnt_result.single()
                contra_count = cnt_record["cnt"] if cnt_record else 0
                results["contradictions"] = contra_count

                if not dry_run and contra_count > 0:
                    await session.run(
                        f"MATCH (e1:{all_labels})-[c:CONTRADICTS]->(e2:{all_labels}) "
                        f"WHERE NOT EXISTS {{ MATCH ()-[r:{_FACTUAL_REL_CYPHER} {{id: c.rel_id_a}}]->() }} "
                        f"   OR NOT EXISTS {{ MATCH ()-[r:{_FACTUAL_REL_CYPHER} {{id: c.rel_id_b}}]->() }} "
                        "DELETE c"
                    )
                    logger.info(
                        "[KnowledgeGraph] Pruned %d dangling contradiction edges.",
                        contra_count,
                    )
            except Exception as exc:
                logger.debug("[KnowledgeGraph] prune (contradictions) failed: %s", exc)

            # ── Pass 4: orphaned claims ────────────────────────────────────
            try:
                claim_q = (
                    "MATCH (cl:Claim) "
                    "WHERE NOT (cl)-[:ASSERTS]->() "
                    "RETURN count(cl) AS cnt"
                )
                cnt_result = await session.run(claim_q)
                cnt_record = await cnt_result.single()
                claim_count = cnt_record["cnt"] if cnt_record else 0
                results["claims"] = claim_count

                if not dry_run and claim_count > 0:
                    await session.run(
                        "MATCH (cl:Claim) " "WHERE NOT (cl)-[:ASSERTS]->() " "DELETE cl"
                    )
                    logger.info(
                        "[KnowledgeGraph] Pruned %d orphaned claims.", claim_count
                    )
            except Exception as exc:
                logger.debug("[KnowledgeGraph] prune (claims) failed: %s", exc)

        return results
