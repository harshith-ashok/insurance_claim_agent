import json
import os

from eval import metrics as m
from query import dense_model

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def test_citation_accuracy_and_verification(golden_set, pipeline_results):
    """Citation accuracy: do the (Page X, ...) tokens in the generated answer match
    expected_citations. Citation verification (not just presence): re-embed the
    claim text next to each citation and check it against the specific chunk on
    that cited page — catches a plausible-but-wrong citation that naive
    format-checking would pass."""
    rows = []
    for entry in golden_set:
        result = pipeline_results[entry["id"]]
        answer = result["answer"] or ""
        parsed = m.parse_citations(answer)
        accuracy = m.citation_accuracy(parsed, entry["expected_citations"])
        verification = m.verify_citations(answer, result["context_candidates"], dense_model)
        rows.append(
            {
                "id": entry["id"],
                "bucket": entry["bucket"],
                "parsed_citations": parsed,
                "citation_accuracy": accuracy,
                "citations_checked": len(verification),
                "citations_verified": sum(1 for v in verification if v["verified"]),
                "verification_detail": verification,
            }
        )

    with open(os.path.join(RESULTS_DIR, "citation_metrics.json"), "w") as f:
        json.dump(rows, f, indent=1)

    checked_total = sum(r["citations_checked"] for r in rows)
    assert checked_total > 0, "no citations were parsed from any generated answer"
