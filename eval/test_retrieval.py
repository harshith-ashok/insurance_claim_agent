import json
import os

from eval import metrics as m

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
K = 4


def test_retrieval_metrics(golden_set, pipeline_results):
    """Recall@k/Precision@k/MRR: did the ground-truth (source, page) chunk make it
    into the reranked candidate list at all. Only meaningful for questions with
    expected_citations (answerable + some ambiguous entries)."""
    rows = []
    for entry in golden_set:
        result = pipeline_results[entry["id"]]
        candidates = result["candidates"]
        expected = entry["expected_citations"]
        rows.append(
            {
                "id": entry["id"],
                "bucket": entry["bucket"],
                "recall_at_k": m.recall_at_k(candidates, expected, K),
                "precision_at_k": m.precision_at_k(candidates, expected, K),
                "mrr": m.mrr(candidates, expected),
            }
        )

    with open(os.path.join(RESULTS_DIR, "retrieval_metrics.json"), "w") as f:
        json.dump({"k": K, "rows": rows}, f, indent=1)

    applicable = [r for r in rows if r["recall_at_k"] is not None]
    assert applicable, "no golden-set questions had expected_citations to score retrieval against"

    mean_recall = sum(r["recall_at_k"] for r in applicable) / len(applicable)
    # Regression guard, not a correctness gate — this catches total pipeline
    # breakage (wrong collection, broken embeddings) rather than enforcing a
    # target score. Tune actual targets from the numbers in eval/report.md.
    assert mean_recall > 0.0, "retrieval found zero ground-truth chunks across the entire golden set"
