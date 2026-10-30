import logging
import os
import time
from typing import Any

from config import COLLECTION_NAME, CONFIDENCE_THRESHOLD, TOP_K_CONTEXT
from exceptions import LLMInferenceError, QdrantConnectionError
from inference.guardrail import verify_citations
from inference.llm import ABSTENTION_MESSAGE, generate
from retrieval.dependencies import PipelineDependencies
from retrieval.search import build_filter, deduplicate, hybrid_search, rerank

logger = logging.getLogger(__name__)

_default_deps = PipelineDependencies.build()

# Backward-compatible module-level aliases — eval/ imports these names
# directly (`from query import dense_model`). The DI refactor (Phase 2) moved
# the actual client construction into PipelineDependencies, but existing
# import sites keep working unchanged.
client = _default_deps.client
dense_model = _default_deps.dense_model
reranker = _default_deps.reranker


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
        logger.debug("Langfuse import/init failed; tracing disabled", exc_info=True)
        return None


def run_query(
    query: str,
    target_section: str | None = None,
    target_source: str | None = None,
    top_k: int = TOP_K_CONTEXT,
    confidence_threshold: float | None = CONFIDENCE_THRESHOLD,
    deps: PipelineDependencies | None = None,
    on_retrieved=None,
    on_generation_start=None,
    on_token=None,
):
    """Programmatic pipeline entry point: retrieve, rerank, dedup, optionally
    gate, generate, verify citations.

    Returns a result dict instead of printing, so it can be called from the eval
    suite. `diagnostic_pipeline` below is a thin printing/streaming wrapper around
    this for interactive/manual use (test.py).

    `deps` lets a caller inject an alternate PipelineDependencies (different
    Qdrant URL, different models); defaults to the module-level singleton built
    at import time. `on_retrieved(result)` fires once reranked candidates are
    available (before any gate/generation decision), `on_generation_start()`
    fires only if the LLM call is about to happen, and `on_token(str)` fires
    per streamed token during generation. All three are optional and unused by
    eval callers.
    """
    deps = deps or _default_deps

    result: dict[str, Any] = {
        "query": query,
        "target_section": target_section,
        "target_source": target_source,
        "error": None,
        "candidates": [],
        "context_candidates": [],
        "top_score": None,
        "gated": False,
        "answer": None,
        "citation_verifications": [],
        # Real per-stage wall-clock timings (ms), not simulated — populated as
        # each stage completes; a stage that never runs (e.g. gen_ms when
        # gated) stays 0. Consumed by api/ to drive the frontend trace view.
        "timings": {"embed_ms": 0.0, "hybrid_ms": 0.0, "rerank_ms": 0.0, "gen_ms": 0.0},
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

    if not deps.client.collection_exists(COLLECTION_NAME):
        result["error"] = f"Collection '{COLLECTION_NAME}' does not exist. Run ingest.py first."
        return result

    t0 = time.perf_counter()
    query_dense = deps.dense_model.encode(query).tolist()
    result["timings"]["embed_ms"] = (time.perf_counter() - t0) * 1000

    query_filter = build_filter(target_section, target_source)

    t1 = time.perf_counter()
    try:
        candidates = hybrid_search(deps.client, COLLECTION_NAME, query, query_dense, query_filter)
    except QdrantConnectionError as exc:
        logger.exception("Retrieval failed for query %r", query)
        result["error"] = str(exc)
        return result
    result["timings"]["hybrid_ms"] = (time.perf_counter() - t1) * 1000

    if not candidates:
        result["gated"] = True
        result["answer"] = ABSTENTION_MESSAGE
        return result

    t2 = time.perf_counter()
    sorted_candidates = rerank(deps.reranker, query, candidates)
    sorted_candidates = deduplicate(sorted_candidates)
    result["timings"]["rerank_ms"] = (time.perf_counter() - t2) * 1000

    result["candidates"] = [c.model_dump() for c in sorted_candidates]
    result["top_score"] = sorted_candidates[0].cross_score
    result["context_candidates"] = [c.model_dump() for c in sorted_candidates[:top_k]]

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

    t3 = time.perf_counter()
    try:
        result["answer"] = generate(query, context_str, on_token=on_token)
    except LLMInferenceError as exc:
        result["error"] = str(exc)
        return result
    result["timings"]["gen_ms"] = (time.perf_counter() - t3) * 1000

    result["citation_verifications"] = verify_citations(
        result["answer"], result["context_candidates"], deps.dense_model
    )
    unverified = [v for v in result["citation_verifications"] if not v["verified"]]
    if unverified:
        logger.warning(
            "%d/%d citations failed verification for query %r",
            len(unverified),
            len(result["citation_verifications"]),
            query,
        )

    if trace:
        trace.update(output={"answer": result["answer"], "gated": False, "top_score": result["top_score"]})
    return result


def diagnostic_pipeline(query: str, target_section: str | None = None, target_source: str | None = None):
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

    unverified = [v for v in result["citation_verifications"] if not v["verified"]]
    if unverified:
        print(f"\n[GUARDRAIL] {len(unverified)} citation(s) failed verification:")
        for v in unverified:
            print(
                f"       (Page {v['citation']['page']}, Section {v['citation']['section']}) — {v.get('reason', 'similarity below threshold')}"
            )

    print("\n\n" + "=" * 80 + "\n")
