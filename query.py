import time
import ollama
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer, CrossEncoder

# --- 1. CONFIGURATION & INITIALIZATION ---
COLLECTION_NAME = "insurance_policies"
DENSE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

client = QdrantClient(url="http://localhost:6333")
dense_model = SentenceTransformer(DENSE_MODEL_NAME)
reranker = CrossEncoder(RERANKER_MODEL_NAME)

def execution_pipeline(query: str, target_section: str = None):
    start_time = time.time()

    # --- 2. VECTOR GENERATION ---
    query_dense = dense_model.encode(query).tolist()

    # Construct an optional semantic metadata pre-filter
    query_filter = None
    if target_section:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.section",
                    match=models.MatchValue(value=target_section)
                )
            ]
        )

    # --- 3. NATIVE HYBRID RRF SEARCH VIA QDRANT ---
    # Query dense & sparse spaces concurrently; fuse ranks internally via RRF
    retrieval_response = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=query_dense,
                using="dense",
                limit=15
            ),
            models.Prefetch(
                query=models.Document(text=query, model="Qdrant/bm25"),
                using="sparse",
                limit=15
            )
        ],
        query=models.RrfQuery(rrf=models.Rrf()),
        query_filter=query_filter,
        limit=5
    )

    candidates = [
        {
            "text": point.payload["text"],
            "score": point.score,
            "meta": point.payload["metadata"]
        }
        for point in retrieval_response.points
    ]

    retrieval_latency = time.time() - start_time

    if not candidates:
        print("No matching documentation layers found.")
        return

    # --- 4. CONDITIONAL RERANKING GATEWAY ---
    # Qdrant's raw RRF score distribution scales relative to rank configurations.
    # For high confidence gating, we test our highest candidate rank score.
    # If the top candidate's relative RRF distribution score is weak, invoke the Cross-Encoder.
    top_initial_score = candidates[0]["score"]
    rerank_triggered = False

    if top_initial_score < 0.032:  # Adjusted threshold index relative to RRF score formulas
        rerank_triggered = True
        pairs = [[query, c["text"]] for c in candidates]
        rerank_scores = reranker.predict(pairs)

        for idx, score in enumerate(rerank_scores):
            candidates[idx]["rerank_score"] = float(score)

        # Re-sort using cross-encoder scores
        candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

    rerank_latency = (time.time() - start_time) - retrieval_latency

    # --- 5. CONTEXT ARCHITECTURE ASSEMBLE ---
    context_blocks = []
    for c in candidates[:3]: # Extract top 3 candidates for final processing
        section_info = f"[Section: {c['meta'].get('section', 'N/A')}]"
        context_blocks.append(f"{section_info}\nContent: {c['text']}")

    context_str = "\n\n---\n\n".join(context_blocks)

    # --- 6. LLM ORCHESTRATION WITH STREAMING ---
    system_prompt = (
        "You are an insurance policy assistant. You answer queries ONLY from the provided context.\n\n"
        "Rules:\n"
        "1. Every single claim must cite its structural source explicitly inline using the exact [Section X] tag format provided in the context context.\n"
        "2. If the answer is not explicitly contained within the context text, respond exactly with: "
        "\"This information is not available in the provided policy documents.\"\n"
        "3. Never infer, extrapolate, or utilize external knowledge weights.\n"
        "4. Keep answers concise, factual, and strictly literal."
    )

    user_prompt = f"Context:\n{context_str}\n\nQuery: {query}"

    print("\n" + "="*40)
    print(f"METRICS: Retrieval Latency: {retrieval_latency:.4f}s | Rerank Triggered: {rerank_triggered} ({rerank_latency:.4f}s)")
    print("="*40 + "\n")

    stream = ollama.chat(
        model="qwen3:8b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        stream=True
    )

    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)
    print("\n")

# --- TEST PIPELINE EXECUTION ---
if __name__ == "__main__":
    # Test sample query execution
    user_query = "What if it's an accidental drug overdose?"
    execution_pipeline(query=user_query)
