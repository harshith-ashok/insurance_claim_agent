import ollama
from qdrant_client import QdrantClient, models
from sentence_transformers import CrossEncoder, SentenceTransformer

from config import (
    COLLECTION_NAME,
    DENSE_MODEL_NAME,
    LLM_MODEL_NAME,
    QDRANT_URL,
    RERANKER_MODEL_NAME,
)

client = QdrantClient(url=QDRANT_URL)
dense_model = SentenceTransformer(DENSE_MODEL_NAME)
reranker = CrossEncoder(RERANKER_MODEL_NAME)

SYSTEM_PROMPT = (
    "You are an insurance system auditor. Provide highly precise answers derived ONLY from the provided text context.\n"
    "Rules:\n"
    "1. Every single assertion must include an inline location token explicitly showing page numbers and sections, matching this exact format: (Page X, Section Y).\n"
    "2. If the context does not explicitly contain the details needed to form a direct answer, state exactly: "
    "'This information is not available in the provided policy documents.' Do not extrapolate.\n"
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


def diagnostic_pipeline(query: str, target_section: str = None, target_source: str = None):
    if not client.collection_exists(COLLECTION_NAME):
        print(
            f"[ERROR] Collection '{COLLECTION_NAME}' does not exist. Run ingest.py first."
        )
        return

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
        print(f"[ERROR] Qdrant query failed: {exc}")
        return

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
        print(f"\n[DIAGNOSTICS] No matches found for query: '{query}'")
        return

    pairs = [[query, c["text"]] for c in candidates]
    rerank_scores = reranker.predict(pairs)
    for candidate, score in zip(candidates, rerank_scores):
        candidate["cross_score"] = float(score)

    sorted_candidates = sorted(candidates, key=lambda x: x["cross_score"], reverse=True)

    print("\n" + "=" * 80)
    print(f" RETRIEVAL DIAGNOSTICS FOR QUERY: '{query}'")
    print("=" * 80)
    for rank, c in enumerate(sorted_candidates[:4]):
        print(
            f"Rank {rank + 1} | Source: {c['meta']['source']} | Page: {c['meta']['page']} | "
            f"Chunk: {c['meta']['chunk_id']} | Section: {c['meta']['section']}"
        )
        print(
            f"       Scores -> Internal RRF Distribution: {c['rrf_score']:.5f} | Cross-Encoder Logit: {c['cross_score']:.4f}"
        )
        print(f"       Snippet preview: {c['text'][:110].strip()}...")
        print("-" * 50)

    context_str = "\n\n".join(
        f"[Location -> Source: {c['meta']['source']}, Page: {c['meta']['page']}, Section: {c['meta']['section']}]\n{c['text']}"
        for c in sorted_candidates[:4]
    )

    print("\n" + "=" * 40)
    print(" STREAMED ENGINE RESPONSE")
    print("=" * 40 + "\n")

    try:
        stream = ollama.chat(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context_str}\n\nQuery: {query}"},
            ],
            stream=True,
        )
        for chunk in stream:
            print(chunk["message"]["content"], end="", flush=True)
    except Exception as exc:
        print(f"[ERROR] LLM inference failed: {exc}")
        return

    print("\n\n" + "=" * 80 + "\n")
