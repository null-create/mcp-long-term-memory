"""
embeddings.py

In-process sentence-transformer embedding and similarity utilities.

A single model instance is shared for the lifetime of the backend process.
All blocking encode() calls are offloaded to a dedicated ThreadPoolExecutor
so the FastAPI event loop is never stalled.

Public API
----------
  embed_texts(texts, model_name, normalize)  → List[List[float]]
  embed_query(text, model_name)              → List[float]
  find_similar(query, corpus, top_k, model)  → List[{"text", "similarity_score", "index"}]
  cosine_similarity(v1, v2)                  → float
"""

from __future__ import annotations

import os
import asyncio
import functools
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, cast

import numpy as np

from config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Set to False (via EMBEDDINGS_ENABLED=false) to disable in-process embeddings
# entirely on memory-constrained hosts.  When disabled, embed_* functions return
# empty results and the SearchResultStore falls back to insertion-order ranking.
EMBEDDINGS_ENABLED: bool = os.getenv("EMBEDDINGS_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
)

# Cap concurrent encode() threads so that a burst of parallel requests does
# not saturate all CPU cores and stall the uvicorn event loop.
_MAX_WORKERS = int(os.getenv("MAX_EMBEDDING_WORKERS", "4"))

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

# Thread pool for offloading CPU-bound encode() calls.
_executor = ThreadPoolExecutor(
    max_workers=_MAX_WORKERS,
    thread_name_prefix="embed",
)

# Per-model cache — avoids re-loading the same weights for every call.
_model_cache: Dict[str, Any] = {}

# Asyncio semaphore — created lazily inside the running event loop so it is
# always bound to the correct loop (avoids ClosedResourceError on Python 3.10+).
_semaphore: Optional[asyncio.Semaphore] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class EmbeddingError(Exception):
    """Raised when an embedding or similarity operation fails."""


def _get_semaphore() -> asyncio.Semaphore:
    """Return (or lazily create) the per-loop semaphore."""
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(_MAX_WORKERS)
    return _semaphore


def _load_model(model_name: str) -> Any:
    """
    Load and cache a SentenceTransformer model (blocking, call from executor).

    Raises EmbeddingError if the import or load fails — e.g. when
    sentence-transformers is not installed in the current environment.
    """
    if model_name in _model_cache:
        return _model_cache[model_name]

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.warning(
            "[embeddings] HF_TOKEN is not set. Downloads from HuggingFace Hub will "
            "use the unauthenticated rate limit, which may be throttled. "
            "Set the HF_TOKEN environment variable to avoid this."
        )

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import]

        # Try loading from local cache first (no network round-trip, no file locks
        # beyond a stat check).  This eliminates the HF Hub network dependency on
        # every startup when the model is already downloaded, and avoids the ~44
        # filelock acquire/release operations caused by HuggingFace Hub's cache
        # validation pass.  Fall back to a normal (possibly downloading) load only
        # if the model is not yet cached locally.
        logger.info("[embeddings] Loading sentence-transformer model: %s", model_name)
        try:
            model = SentenceTransformer(
                model_name, token=hf_token, local_files_only=True
            )
            logger.info("[embeddings] Model loaded from local cache: %s", model_name)
        except Exception:
            logger.info(
                "[embeddings] Model not in local cache — downloading: %s", model_name
            )
            model = SentenceTransformer(model_name, token=hf_token)
            logger.info("[embeddings] Model downloaded and loaded: %s", model_name)

        _model_cache[model_name] = model
        return model
    except ImportError as exc:
        raise EmbeddingError(
            "sentence-transformers is not installed. "
            "Add it to backend/requirements.txt and rebuild."
        ) from exc
    except Exception as exc:
        raise EmbeddingError(f"Failed to load model '{model_name}': {exc}") from exc


def _encode_texts(
    texts: List[str],
    model_name: str = DEFAULT_MODEL,
    normalize: bool = True,
) -> List[List[float]]:
    """
    Blocking encode — must be called from the ThreadPoolExecutor.
    Returns a list of float vectors, one per input text.
    """
    model = _load_model(model_name)
    raw = model.encode(
        texts,
        normalize_embeddings=normalize,
        show_progress_bar=len(texts) > 10,
    )
    if isinstance(raw, np.ndarray):
        return raw.tolist()
    return [v.tolist() if isinstance(v, np.ndarray) else v for v in raw]


def _encode_single(
    text: str,
    model_name: str = DEFAULT_MODEL,
    normalize: bool = True,
) -> List[float]:
    """Blocking single-text encode — must be called from the ThreadPoolExecutor."""
    vecs = _encode_texts([text], model_name=model_name, normalize=normalize)
    return vecs[0]


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------


async def embed_texts(
    texts: List[str],
    model_name: str = DEFAULT_MODEL,
    normalize: bool = True,
) -> List[List[float]]:
    """
    Embed a batch of texts and return their vectors.

    Parameters
    ----------
    texts:
        Non-empty list of strings to embed.
    model_name:
        Sentence-transformer model identifier (HuggingFace hub or local path).
    normalize:
        Whether to L2-normalize the output vectors (required for cosine dot-product).

    Returns
    -------
    List[List[float]]
        One float vector per input text, in the same order.

    Raises
    ------
    EmbeddingError
        If the model cannot be loaded or encoding fails.
    ValueError
        If *texts* is empty.
    """
    if not EMBEDDINGS_ENABLED:
        logger.debug(
            "[embeddings] embed_texts called but EMBEDDINGS_ENABLED=false; returning empty."
        )
        return []

    if not texts:
        raise ValueError("embed_texts: texts must be a non-empty list")

    cleaned = [t.strip() for t in texts if t and t.strip()]
    if not cleaned:
        raise ValueError("embed_texts: all provided texts are empty after stripping")

    loop = asyncio.get_running_loop()
    sem = _get_semaphore()
    async with sem:
        vectors: List[List[float]] = await loop.run_in_executor(
            _executor,
            cast(
                Callable[[], List[List[float]]],
                functools.partial(_encode_texts, cleaned, model_name, normalize),
            ),
        )
    return vectors


async def embed_query(
    text: str,
    model_name: str = DEFAULT_MODEL,
) -> List[float]:
    """
    Embed a single query string and return its vector.

    Convenience wrapper around :func:`embed_texts` for the common single-text
    case (e.g. embedding a step description before cosine ranking).
    """
    vecs = await embed_texts([text], model_name=model_name)
    return vecs[0]


async def find_similar(
    query: str,
    corpus: List[str],
    top_k: int = 5,
    model_name: str = DEFAULT_MODEL,
) -> List[Dict[str, Any]]:
    """
    Find the *top_k* most semantically similar texts in *corpus* to *query*.

    Parameters
    ----------
    query:
        The query string.
    corpus:
        List of candidate strings to rank.
    top_k:
        Maximum number of results to return.
    model_name:
        Sentence-transformer model identifier.

    Returns
    -------
    List[dict]
        Each dict has keys ``"text"``, ``"similarity_score"`` (float in [−1, 1]),
        and ``"index"`` (position in the original *corpus* list), sorted
        best-first.
    """
    if not query or not query.strip():
        raise ValueError("find_similar: query must be non-empty")
    if not corpus:
        raise ValueError("find_similar: corpus must be non-empty")

    actual_top_k = min(top_k, len(corpus))
    all_texts = [query] + corpus

    loop = asyncio.get_running_loop()
    sem = _get_semaphore()
    async with sem:
        all_vecs: List[List[float]] = await loop.run_in_executor(
            _executor,
            cast(
                Callable[[], List[List[float]]],
                functools.partial(_encode_texts, all_texts, model_name, True),
            ),
        )

    q_vec = np.array(all_vecs[0], dtype=np.float32)
    corpus_matrix = np.array(all_vecs[1:], dtype=np.float32)
    similarities = corpus_matrix.dot(q_vec).tolist()

    top_indices = sorted(
        range(len(similarities)), key=lambda i: similarities[i], reverse=True
    )[:actual_top_k]
    return [
        {
            "text": corpus[i],
            "similarity_score": float(similarities[i]),
            "index": i,
        }
        for i in top_indices
    ]


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """
    Compute cosine similarity between two pre-computed embedding vectors.

    Raises EmbeddingError if the vectors have different dimensions.
    """
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    if a.shape != b.shape:
        raise EmbeddingError(
            f"cosine_similarity: dimension mismatch {a.shape} vs {b.shape}"
        )
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Background warm-up
# ---------------------------------------------------------------------------


async def warm_up(model_name: str = DEFAULT_MODEL) -> None:
    """
    Pre-load the embedding model in the background so the first real encode()
    call is fast.  Called from the FastAPI lifespan (api_server.py).
    Does NOT raise — model load failures are logged as warnings so they don't
    block server startup.
    """
    if not EMBEDDINGS_ENABLED:
        logger.info("[embeddings] EMBEDDINGS_ENABLED=false — skipping model warm-up.")
        return

    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            _executor,
            functools.partial(_load_model, model_name),
        )
        logger.info("[embeddings] Warm-up complete for model: %s", model_name)
    except Exception as exc:
        logger.warning(
            "[embeddings] Model warm-up failed (will retry on first use): %s", exc
        )
