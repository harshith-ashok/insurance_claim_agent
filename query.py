import os

import ollama
from qdrant_client import QdrantClient, models
from sentence_transformers import CrossEncoder, SentenceTransformer

from config import (
    COLLECTION_NAME,
    CONFIDENCE_THRESHOLD,
    DENSE_MODEL_NAME,
    LLM_MODEL_NAME,
    QDRANT_URL,
    RERANKER_MODEL_NAME,
)

client = QdrantClient(url=QDRANT_URL)
dense_model = SentenceTransformer(DENSE_MODEL_NAME)
reranker = CrossEncoder(RERANKER_MODEL_NAME)

ABSTENTION_MESSAGE = "This information is not available in the provided policy documents."

SYSTEM_PROMPT = (
    "You are an insurance system auditor. Provide highly precise answers derived ONLY from the provided text context.\n"
    "Rules:\n"
    "1. Every single assertion must include an inline location token explicitly showing page numbers and sections, matching this exact format: (Page X, Section Y).\n"
    f"2. If the context does not explicitly contain the details needed to form a direct answer, state exactly: "
    f"'{ABSTENTION_MESSAGE}' Do not extrapolate.\n"
    "3. Explicitly enumerate all benefits, requirements, or exclusions mentioned in the context without truncation."
)


def _build_filter(target_section: str = None, target_source: str = None):
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


def _get_tracer():
    """Best-effort Langfuse tracing client. Returns None (no-op) unless both the
    `langfuse` package and LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY are present —
    tracing is opt-in and never blocks the pipeline if unconfigured."""
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        return None
    try:
        from langfuse import Langfuse

        return Langfuse()
    except Exception:
        return None


def run_query(
    query: str,
    target_section: str = None,
    target_source: str = None,
    top_k: int = 4,
    confidence_threshold: float | None = CONFIDENCE_THRESHOLD,
    on_retrieved=None,
    on_generation_start=None,
    on_token=None,
):
    """Programmatic pipeline entry point: retrieve, rerank, optionally gate, generate.

    Returns a result dict instead of printing, so it can be called from the eval
    suite. `diagnostic_pipeline` below is a thin printing/streaming wrapper around
    this for interactive/manual use (test.py).

    `on_retrieved(result)` fires once reranked candidates are available (before any
    gate/generation decision), `on_generation_start()` fires only if the LLM call is
    about to happen, and `on_token(str)` fires per streamed token during generation.
    All three are optional and unused by eval callers.
    """
    result = {
        "query": query,
        "target_section": target_section,
        "target_source": target_source,
        "error": None,
        "candidates": [],
        "context_candidates": [],
        "top_score": None,
        "gated": False,
        "answer": None,
    }

    tracer = _get_tracer()
    trace = (
        tracer.trace(
            name="diagnostic_pipeline",
            input={"query": query, "target_section": target_section, "target_source": target_source},
        )
        if tracer
        else None
    )

    if not client.collection_exists(COLLECTION_NAME):
        result["error"] = f"Collection '{COLLECTION_NAME}' does not exist. Run ingest.py first."
        return result

    query_dense = dense_model.encode(query).tolist()
    query_filter = _build_filter(target_section, target_source)

    try:
        retrieval_response = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(query=query_dense, using="dense", limit=30),
                models.Prefetch(
                    query=models.Document(text=query, model="Qdrant/bm25"),
                    using="sparse",
                    limit=30,
                ),
            ],
            query=models.RrfQuery(rrf=models.Rrf()),
            query_filter=query_filter,
            limit=15,
        )
    except Exception as exc:
        result["error"] = f"Qdrant query failed: {exc}"
        return result

    seen_chunks = set()
    candidates = []
    for point in retrieval_response.points:
        cid = point.payload["metadata"]["chunk_id"]
        if cid in seen_chunks:
            continue
        seen_chunks.add(cid)
        candidates.append(
            {"text": point.payload["text"], "rrf_score": point.score, "meta": point.payload["metadata"]}
        )

    if not candidates:
        result["gated"] = True
        result["answer"] = ABSTENTION_MESSAGE
        return result

    pairs = [[query, c["text"]] for c in candidates]
    rerank_scores = reranker.predict(pairs)
    for candidate, score in zip(candidates, rerank_scores):
        candidate["cross_score"] = float(score)

    sorted_candidates = sorted(candidates, key=lambda x: x["cross_score"], reverse=True)
    result["candidates"] = sorted_candidates
    result["top_score"] = sorted_candidates[0]["cross_score"]
    result["context_candidates"] = sorted_candidates[:top_k]

    if on_retrieved:
        on_retrieved(result)

    if confidence_threshold is not None and result["top_score"] < confidence_threshold:
        result["gated"] = True
        result["answer"] = ABSTENTION_MESSAGE
        if trace:
            trace.update(output={"answer": result["answer"], "gated": True, "top_score": result["top_score"]})
        return result

    if on_generation_start:
        on_generation_start()

    context_str = "\n\n".join(
        f"[Location -> Source: {c['meta']['source']}, Page: {c['meta']['page']}, Section: {c['meta']['section']}]\n{c['text']}"
        for c in result["context_candidates"]
    )

    try:
        stream = ollama.chat(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context_str}\n\nQuery: {query}"},
            ],
            stream=True,
        )
        tokens = []
        for chunk in stream:
            token = chunk["message"]["content"]
            tokens.append(token)
            if on_token:
                on_token(token)
        result["answer"] = "".join(tokens)
    except Exception as exc:
        result["error"] = f"LLM inference failed: {exc}"
        return result

    if trace:
        trace.update(output={"answer": result["answer"], "gated": False, "top_score": result["top_score"]})
    return result


def diagnostic_pipeline(query: str, target_section: str = None, target_source: str = None):
    def _print_diagnostics(res):
        print("\n" + "=" * 80)
        print(f" RETRIEVAL DIAGNOSTICS FOR QUERY: '{query}'")
        print("=" * 80)
        for rank, c in enumerate(res["context_candidates"]):
            print(
                f"Rank {rank + 1} | Source: {c['meta']['source']} | Page: {c['meta']['page']} | "
                f"Chunk: {c['meta']['chunk_id']} | Section: {c['meta']['section']}"
            )
            print(
                f"       Scores -> Internal RRF Distribution: {c['rrf_score']:.5f} | Cross-Encoder Logit: {c['cross_score']:.4f}"
            )
            print(f"       Snippet preview: {c['text'][:110].strip()}...")
            print("-" * 50)

    def _print_header():
        print("\n" + "=" * 40)
        print(" STREAMED ENGINE RESPONSE")
        print("=" * 40 + "\n")

    result = run_query(
        query,
        target_section=target_section,
        target_source=target_source,
        on_retrieved=_print_diagnostics,
        on_generation_start=_print_header,
        on_token=lambda token: print(token, end="", flush=True),
    )

    if result["error"]:
        print(f"[ERROR] {result['error']}")
        return

    if not result["candidates"]:
        print(f"\n[DIAGNOSTICS] No matches found for query: '{query}'")
        return

    if result["gated"]:
        print(
            f"\n[GATED] Top cross-encoder score {result['top_score']:.4f} is below the "
            "confidence threshold — skipped the LLM call.\n"
        )
        print(result["answer"])
        print("\n\n" + "=" * 80 + "\n")
        return

    print("\n\n" + "=" * 80 + "\n")
