import json
import os
from collections import Counter

from eval import metrics as m

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def test_abstention_confusion_matrix(golden_set, pipeline_results):
    """Classifies every golden-set run as correctly_answered / correctly_abstained
    / hallucinated / wrongly_refused, per-bucket and overall."""
    rows = []
    for entry in golden_set:
        result = pipeline_results[entry["id"]]
        rows.append(
            {
                "id": entry["id"],
                "bucket": entry["bucket"],
                "classification": m.classify_abstention(entry["expected_answer"], result["answer"] or ""),
            }
        )

    overall = dict(Counter(r["classification"] for r in rows))
    by_bucket = {
        bucket: dict(Counter(r["classification"] for r in rows if r["bucket"] == bucket))
        for bucket in ("answerable", "unanswerable", "ambiguous")
    }

    with open(os.path.join(RESULTS_DIR, "abstention_matrix.json"), "w") as f:
        json.dump({"overall": overall, "by_bucket": by_bucket, "rows": rows}, f, indent=1)

    hallucinated = overall.get("hallucinated", 0)
    unanswerable_total = sum(1 for r in rows if r["bucket"] == "unanswerable")
    if unanswerable_total:
        # Regression guard: hallucinating on most of the unanswerable bucket means
        # the citation-only abstention prompt isn't being honored at all.
        assert hallucinated / unanswerable_total <= 0.5, (
            f"hallucinated on {hallucinated}/{unanswerable_total} unanswerable questions"
        )
