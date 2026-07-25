"""Simulates config.CONFIDENCE_THRESHOLD at several candidate values against the
already-recorded pipeline_results.json, without re-running the LLM: gating just
means "top_score below threshold -> answer becomes the abstention message,"
which classify_abstention() can evaluate offline. Run after the core eval suite:

    pytest eval/test_retrieval.py eval/test_citations.py eval/test_abstention.py
    python -m eval.gate_analysis
"""

import json
import os
from collections import Counter

from eval import metrics as m
from eval.dataset import load_golden_set
from query import ABSTENTION_MESSAGE

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CANDIDATE_THRESHOLDS = [None, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]


def simulate(threshold, golden_by_id, pipeline_results):
    counts = Counter()
    for eid, r in pipeline_results.items():
        entry = golden_by_id[eid]
        top_score = r["top_score"]
        gated = threshold is not None and top_score is not None and top_score < threshold
        actual_answer = ABSTENTION_MESSAGE if gated else (r["answer"] or "")
        counts[m.classify_abstention(entry["expected_answer"], actual_answer)] += 1
    return dict(counts)


def run():
    golden_by_id = {e["id"]: e for e in load_golden_set()}
    with open(os.path.join(RESULTS_DIR, "pipeline_results.json")) as f:
        pipeline_results = json.load(f)

    rows = [
        {"threshold": t, **simulate(t, golden_by_id, pipeline_results)} for t in CANDIDATE_THRESHOLDS
    ]
    with open(os.path.join(RESULTS_DIR, "gate_analysis.json"), "w") as f:
        json.dump(rows, f, indent=1)
    return rows


if __name__ == "__main__":
    for row in run():
        print(row)
