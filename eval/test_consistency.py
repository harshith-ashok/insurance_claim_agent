import json
import os
import random

from eval import metrics as m
from query import dense_model, run_query

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def test_consistency(golden_set, consistency_n, consistency_repeats):
    """Runs a sample of answerable questions N times each and measures answer
    stability (mean pairwise embedding similarity across runs) and citation
    stability (agreement rate on the cited-page set across runs)."""
    answerable = [e for e in golden_set if e["bucket"] == "answerable"]
    sample = random.Random(42).sample(answerable, min(consistency_n, len(answerable)))

    rows = []
    for entry in sample:
        answers = []
        citation_runs = []
        for _ in range(consistency_repeats):
            result = run_query(entry["question"], target_source=entry.get("target_source"))
            answer = result["answer"] or ""
            answers.append(answer)
            citation_runs.append(m.parse_citations(answer))

        pairwise = [
            m.answer_similarity(dense_model, answers[i], answers[j])
            for i in range(len(answers))
            for j in range(i + 1, len(answers))
        ]
        rows.append(
            {
                "id": entry["id"],
                "question": entry["question"],
                "runs": consistency_repeats,
                "mean_pairwise_answer_similarity": sum(pairwise) / len(pairwise) if pairwise else 1.0,
                "citation_agreement": m.citation_agreement(citation_runs),
            }
        )

    with open(os.path.join(RESULTS_DIR, "consistency.json"), "w") as f:
        json.dump(rows, f, indent=1)

    assert rows, "no answerable golden-set questions were available to sample for consistency testing"
