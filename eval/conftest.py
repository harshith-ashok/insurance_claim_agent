import json
import os

import pytest
from qdrant_client import QdrantClient

from config import COLLECTION_NAME, QDRANT_URL
from eval.dataset import load_golden_set
from query import run_query

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def pytest_addoption(parser):
    parser.addoption(
        "--consistency-n",
        type=int,
        default=5,
        help="Number of answerable golden-set questions sampled for the consistency test (default: 5)",
    )
    parser.addoption(
        "--consistency-repeats",
        type=int,
        default=5,
        help="Number of repeated runs per sampled question in the consistency test (default: 5)",
    )
    parser.addoption(
        "--ragas-n",
        type=int,
        default=12,
        help="Number of answerable golden-set questions sampled for RAGAS metrics (default: 12)",
    )


@pytest.fixture(scope="session")
def consistency_n(request):
    return request.config.getoption("--consistency-n")


@pytest.fixture(scope="session")
def consistency_repeats(request):
    return request.config.getoption("--consistency-repeats")


@pytest.fixture(scope="session")
def ragas_n(request):
    return request.config.getoption("--ragas-n")


@pytest.fixture(scope="session", autouse=True)
def _require_live_services():
    """The eval suite calls the real pipeline end to end (no mocks — see
    CLAUDE.md's testing philosophy) so it needs Qdrant + Ollama actually running
    with a populated collection. Skip with a clear reason instead of a wall of
    connection-refused tracebacks."""
    client = QdrantClient(url=QDRANT_URL)
    try:
        if not client.collection_exists(COLLECTION_NAME):
            pytest.skip(f"Qdrant collection '{COLLECTION_NAME}' does not exist — run `python ingest.py` first.")
    except Exception as exc:
        pytest.skip(f"Qdrant is not reachable at {QDRANT_URL}: {exc}")

    try:
        import ollama

        ollama.list()
    except Exception as exc:
        pytest.skip(f"Ollama is not reachable: {exc}")


@pytest.fixture(scope="session")
def golden_set():
    return load_golden_set()


@pytest.fixture(scope="session")
def pipeline_results(golden_set):
    """Runs the full pipeline once per golden-set question and caches results for
    reuse across retrieval/citation/abstention/RAGAS test modules — avoids
    re-paying LLM latency per metric computed on the same run."""
    results = {}
    for entry in golden_set:
        results[entry["id"]] = run_query(entry["question"], target_source=entry.get("target_source"))

    with open(os.path.join(RESULTS_DIR, "pipeline_results.json"), "w") as f:
        json.dump(
            {
                eid: {
                    "answer": r["answer"],
                    "error": r["error"],
                    "gated": r["gated"],
                    "top_score": r["top_score"],
                    "context_candidates": [
                        {"meta": c["meta"], "cross_score": c["cross_score"]} for c in r["context_candidates"]
                    ],
                }
                for eid, r in results.items()
            },
            f,
            indent=1,
        )
    return results
