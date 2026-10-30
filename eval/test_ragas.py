import json
import os
import random

import pytest

from eval.ragas_eval import get_ragas_components

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def test_ragas_metrics(golden_set, pipeline_results, ragas_n):
    """context_precision, context_recall, faithfulness, answer_relevancy on a
    sample of answerable golden-set questions, using the LLM (via Ollama's
    OpenAI-compatible endpoint) and embeddings the pipeline already runs with.
    Skips (not fails) if ragas/openai aren't installed — see requirements.txt."""
    components = get_ragas_components()
    if components is None:
        pytest.skip(
            "ragas not usable in this environment — install with "
            "`uv pip install ragas` (see eval/README.md for the known "
            "langchain-community/langchain-openai pin required alongside it)"
        )

    answerable = [e for e in golden_set if e["bucket"] == "answerable" and e["expected_citations"]]
    sample = random.Random(7).sample(answerable, min(ragas_n, len(answerable)))

    rows = []
    for entry in sample:
        result = pipeline_results[entry["id"]]
        answer = result["answer"] or ""
        contexts = [c["text"] for c in result["context_candidates"]]
        row = {"id": entry["id"], "question": entry["question"]}
        if not contexts or not answer:
            row["error"] = "no answer/context available (pipeline may have errored or been gated)"
            rows.append(row)
            continue
        try:
            row["faithfulness"] = (
                components["faithfulness"]
                .score(user_input=entry["question"], response=answer, retrieved_contexts=contexts)
                .value
            )
            row["answer_relevancy"] = (
                components["answer_relevancy"].score(user_input=entry["question"], response=answer).value
            )
            row["context_precision"] = (
                components["context_precision"]
                .score(
                    user_input=entry["question"],
                    reference=entry["expected_answer"],
                    retrieved_contexts=contexts,
                )
                .value
            )
            row["context_recall"] = (
                components["context_recall"]
                .score(
                    user_input=entry["question"],
                    retrieved_contexts=contexts,
                    reference=entry["expected_answer"],
                )
                .value
            )
        except Exception as exc:
            row["error"] = str(exc)
        rows.append(row)

    with open(os.path.join(RESULTS_DIR, "ragas_metrics.json"), "w") as f:
        json.dump(rows, f, indent=1)

    assert rows, "no answerable golden-set questions were available to score with RAGAS"
