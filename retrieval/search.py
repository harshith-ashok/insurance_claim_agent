import difflib
import logging

from qdrant_client import QdrantClient, models
from sentence_transformers import CrossEncoder

from config import DEDUP_SIMILARITY_THRESHOLD, PREFETCH_LIMIT, RRF_LIMIT
from exceptions import QdrantConnectionError
from models import ChunkMetadata, RetrievalCandidate

logger = logging.getLogger(__name__)


def build_filter(
    target_section: str | None = None, target_source: str | None = None
) -> models.Filter | None:
    conditions = []
    if target_section:
        conditions.append(
            models.FieldCondition(
                key="metadata.section", match=models.MatchValue(value=str(target_section))
            )
        )
    if target_source:
        conditions.append(
            models.FieldCondition(
                key="metadata.source", match=models.MatchValue(value=str(target_source))
            )
        )
    return models.Filter(must=conditions) if conditions else None


def hybrid_search(
    client: QdrantClient,
    collection_name: str,
    query: str,
    query_dense: list[float],
    query_filter: models.Filter | None,
) -> list[RetrievalCandidate]:
    """Dense+sparse hybrid retrieval fused server-side via RRF, deduped on
    chunk_id (dense and sparse prefetch can both surface the same chunk)."""
    try:
        response = client.query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(query=query_dense, using="dense", limit=PREFETCH_LIMIT),
                models.Prefetch(
                    query=models.Document(text=query, model="Qdrant/bm25"),
                    using="sparse",
                    limit=PREFETCH_LIMIT,
                ),
            ],
            query=models.RrfQuery(rrf=models.Rrf()),
            query_filter=query_filter,
            limit=RRF_LIMIT,
        )
    except Exception as exc:
        raise QdrantConnectionError(f"Qdrant query failed: {exc}") from exc

    seen_chunks = set()
    candidates = []
    for point in response.points:
        meta = ChunkMetadata.model_validate(point.payload["metadata"])
        if meta.chunk_id in seen_chunks:
            continue
        seen_chunks.add(meta.chunk_id)
        candidates.append(
            RetrievalCandidate(text=point.payload["text"], rrf_score=point.score, meta=meta)
        )
    return candidates


def rerank(
    reranker: CrossEncoder, query: str, candidates: list[RetrievalCandidate]
) -> list[RetrievalCandidate]:
    """Cross-encoder rerank, always on — a confidence-gated skip is a Phase 4
    latency optimization (refactor.md), not implemented here."""
    if not candidates:
        return []
    pairs = [[query, c.text] for c in candidates]
    scores = reranker.predict(pairs)
    for candidate, score in zip(candidates, scores):
        candidate.cross_score = float(score)
    return sorted(candidates, key=lambda c: c.cross_score, reverse=True)


def deduplicate(
    candidates: list[RetrievalCandidate], threshold: float = DEDUP_SIMILARITY_THRESHOLD
) -> list[RetrievalCandidate]:
    """Drops a candidate whose text is near-identical (difflib ratio >=
    threshold) to an already-kept, higher-ranked candidate from the SAME
    (source, page) — sliding-window chunking (config.CHUNK_OVERLAP) can split
    one page into two adjacent, mostly-overlapping chunks that both rank
    highly and waste two context slots on the same content.

    Deliberately scoped to same-page pairs only: the golden set's
    near-duplicate-chunk traps (golden_set.jsonl ambig-02/ambig-03) are
    textually similar tables repeated across *different* pages/endorsements,
    and those must stay distinct candidates — collapsing them on text
    similarity alone would erase the exact distinction retrieval is being
    tested on.
    """
    kept: list[RetrievalCandidate] = []
    for candidate in candidates:
        is_duplicate = any(
            k.meta.source == candidate.meta.source
            and k.meta.page == candidate.meta.page
            and difflib.SequenceMatcher(None, candidate.text, k.text).ratio() >= threshold
            for k in kept
        )
        if is_duplicate:
            logger.debug(
                "Dropping near-duplicate candidate: %s p.%s", candidate.meta.source, candidate.meta.page
            )
            continue
        kept.append(candidate)
    return kept
