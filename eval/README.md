# Evaluation suite

Phase 1 of `refactor.md`: measures the pipeline against a hand-built golden
dataset instead of eyeballing `test.py` output. Calls `query.run_query()`
programmatically — no mocks, real Qdrant + real Ollama (see CLAUDE.md).

## Prerequisites

- Qdrant running with the collection populated: `docker run -d -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant && python ingest.py`
- Ollama running and signed in, with `gpt-oss:120b-cloud` pulled
- `uv pip install -r requirements.txt` (includes `pytest`; RAGAS deps are also
  in there but skip gracefully if something in that stack fails to import —
  see the comment in `requirements.txt` about the exact pins required)

If either service isn't reachable, the whole suite skips with a clear reason
(`eval/conftest.py`) rather than failing with connection-refused tracebacks.

## Running

```bash
# from the repo root
pytest eval/ -v

# tune sample sizes (defaults: 5 questions x 5 repeats for consistency, 12 for RAGAS)
pytest eval/ -v --consistency-n 10 --consistency-repeats 5 --ragas-n 20

# just one metric family
pytest eval/test_retrieval.py -v

# regenerate the report after any run
python -m eval.report
```

Each test module writes its raw results to `eval/results/*.json`; `eval/report.py`
aggregates them into `eval/report.md`.

## What's measured

| File | Metric |
| --- | --- |
| `test_retrieval.py` | Recall@k / Precision@k / MRR — did the ground-truth `(source, page)` chunk make it into the reranked candidates |
| `test_citations.py` | Citation accuracy (cited page vs. expected page) **and** citation verification (re-embed the claim text next to each citation, check it against the specific chunk on that cited page — catches a plausible-but-wrong citation, not just a malformed one) |
| `test_abstention.py` | Confusion matrix: correctly_answered / correctly_abstained / hallucinated / wrongly_refused, per bucket |
| `test_consistency.py` | Repeated-run stability: mean pairwise answer-embedding similarity + citation-set agreement rate |
| `test_ragas.py` | RAGAS `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall` |

Ground truth is matched on `(source, page)` only, deliberately excluding
`section` — the section-tagging regex in `ingest.py` is a known-noisy
heuristic (see README's Known Limitations), and treating it as a hard match
key would understate retrieval quality rather than measure it.

## Golden set

`golden_set.jsonl` — 53 hand-written questions grounded in the actual indexed
chunks (pulled from a live Qdrant scroll, not just the raw PDF text), split
into three buckets:

- **answerable** (31) — verifiable answers pulled from real clauses in both PDFs
- **unanswerable** (12) — plausible insurance questions these specific documents don't cover
- **ambiguous** (10) — multi-hop reasoning, near-duplicate-chunk traps (e.g. the
  same compensation-scale table repeated for owner-driver/paid-driver/hirer
  endorsements), and adversarial cross-document / vague queries

Each entry: `id`, `bucket`, `question`, `target_source` (filter to pass to
`run_query`, or `null` to test unfiltered cross-document behavior),
`expected_answer`, `expected_citations` (list of `{source, page, section}`),
`notes` (why the entry is interesting, where relevant).

## Confidence gate

`config.CONFIDENCE_THRESHOLD` (default `None` = disabled) is passed to
`run_query()`. When set, `run_query` abstains deterministically — skipping the
LLM call entirely — if the top reranked cross-encoder score is below the
threshold. Tune it using the score distributions in `eval/report.md`
(`retrieval_metrics.json` / `pipeline_results.json` carry per-question
`cross_score`s) rather than guessing, then re-run
`pytest eval/test_abstention.py` to see the effect on the confusion matrix.

## Tracing (optional)

`query.run_query()` emits a Langfuse trace per call if `langfuse` is
installed and `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` are set in the
environment. Unset or uninstalled, it's a no-op — nothing to configure to run
the rest of the suite.
