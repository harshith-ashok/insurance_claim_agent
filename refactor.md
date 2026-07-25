# Refactor & Hardening Plan

Goal: turn this from a working prototype into a portfolio-grade RAG system —
one that demonstrates measured accuracy/consistency/abstention behavior, and
follows enterprise engineering standards (testing, observability, structure).

Priority tags: **P0** = do first / highest resume impact, **P1** = important,
**P2** = polish.

---

## Phase 1 — Evaluation framework (P0, the core ask)

This is the most important phase. Right now there is no measurement of
accuracy, consistency, or abstention — `test.py` only prints output for a
human to eyeball. Everything below turns that into real, reportable numbers.

- [ ] Build a golden evaluation dataset (`eval/golden_set.jsonl`) with three buckets, each entry carrying `question`, `expected_answer`, `expected_citations` (page/section/source), and `bucket`:
  - [ ] **Answerable** — questions with a verifiable answer in the PDFs (pull ~15-20 per document from real clauses)
  - [ ] **Unanswerable** — questions plausible for an insurance policy but not covered by these specific documents (tests false-positive/hallucination risk)
  - [ ] **Ambiguous / multi-hop / adversarial** — cross-section reasoning, long-tail clause language, near-duplicate-chunk traps
- [ ] Write a `eval/` package with a real test runner (pytest) that loads the golden set and calls `diagnostic_pipeline` (or its refactored equivalent) programmatically — no more manual script + eyeballing
- [ ] Implement retrieval metrics: Recall@k, Precision@k, MRR — did the ground-truth chunk make it into the top-4 context at all
- [ ] Implement citation accuracy: parse `(Page X, Section Y)` tokens out of the generated answer and diff against expected citations
- [ ] Implement an abstention confusion matrix — for each bucket, classify each run as: correctly answered / correctly abstained / hallucinated (answered when it shouldn't have) / wrongly refused (abstained when answerable)
- [ ] Implement a faithfulness/groundedness check — verify generated claims are traceable to the retrieved context (LLM-as-judge or RAGAS `faithfulness`); this is the actual hallucination metric, not just "did it cite something"
- [ ] Implement a consistency test — run each golden question N times (e.g. N=5) and measure answer/citation stability (embedding similarity across runs, citation agreement rate)
- [ ] Wire in RAGAS (or a hand-rolled equivalent) for `context_precision`, `context_recall`, `faithfulness`, `answer_relevance` — the README already names this as a goal, actually build it
- [ ] Generate a results report (`eval/report.md` or a small HTML/artifact) summarizing all of the above with numbers — this is the deliverable to put in the README/resume, not prose claims
- [ ] Add a retrieval-confidence gate: if the top reranked cross-encoder score is below a tuned threshold, skip the LLM call and return "not available" deterministically, instead of relying solely on prompt compliance — then measure how much this improves the abstention confusion matrix

## Phase 2 — Architecture refactor (P0/P1)

- [ ] Split into a real package: `ingestion/`, `retrieval/`, `inference/`, `eval/`, `api/` (matches the structure the README already promises — actually build it instead of just describing it)
- [ ] Replace module-level global singletons in `query.py` (client, dense_model, reranker) with dependency-injected objects/classes — needed for unit testing without live services
- [ ] Add pydantic models for `Chunk`, `RetrievalCandidate`, `QueryResult` instead of raw dicts passed around by string key
- [ ] Replace `print()` with the standard `logging` module — structured logs with levels (DEBUG for diagnostics, INFO for pipeline steps, ERROR for failures)
- [ ] Replace bare `except Exception: print(...)` with specific exception types (`QdrantConnectionError`, `LLMInferenceError`, etc.) and proper logging, still degrading gracefully but traceable
- [ ] Move hardcoded retrieval params (prefetch limit=30, RRF limit=15, top-k context=4, chunk_size/overlap) into `config.py` as named, documented, tunable constants — then justify each via Phase 1 eval results, not guesswork
- [ ] Deduplicate near-identical chunks in the reranked candidate list before taking top-4 (overlap window currently allows near-duplicates to occupy multiple context slots)

## Phase 3 — Reliability & observability (P1)

- [ ] Add latency instrumentation around each pipeline stage (embed, retrieve, rerank, LLM call) and log/report them
- [ ] Add basic token/cost tracking for the cloud LLM calls
- [ ] Add incremental ingestion support (upsert changed/added PDFs) instead of full destructive collection wipe every run; keep the full-rebuild path as an explicit `--force` flag
- [ ] Add a lightweight collection-versioning scheme (e.g. alias swap) so re-ingest doesn't leave a query-time window with an empty/partial collection
- [ ] Validate table-heavy pages (premium schedules, benefit tables) separately — `pymupdf4llm` markdown conversion is a known weak point for tabular insurance data; add golden-set cases specifically targeting tables

## Phase 4 — Enterprise coding standards (P1/P2)

- [ ] Add `pyproject.toml` with `ruff` (lint) and `mypy` (type check) configured; add type hints across all functions
- [ ] Add a GitHub Actions CI workflow: lint, type-check, run the Phase 1 eval suite on every PR
- [ ] Add pre-commit hooks for lint/format
- [ ] Add docstrings (one style, consistently — Google or NumPy) to all public functions
- [ ] Add a `Dockerfile`/`docker-compose.yml` covering the app + Qdrant for reproducible setup (README already implies this should exist)
- [ ] Wrap `diagnostic_pipeline` in a FastAPI service (`/query` endpoint) with request/response pydantic schemas and OpenAPI docs — makes the project demoable, not just importable

## Phase 5 — Documentation & presentation (P2, resume polish)

- [ ] Rewrite `README.md` to describe only what's actually implemented, with an explicit "Roadmap" section for anything aspirational (stop the doc/code drift CLAUDE.md currently has to warn about)
- [ ] Add the Phase 1 eval report numbers to the README as a results table (recall@k, citation accuracy, abstention confusion matrix, consistency score) — concrete metrics read far stronger on a resume than architecture description alone
- [ ] Add an architecture diagram (ingestion pipeline + query pipeline) to the README
- [ ] Add a short "known limitations" section (section-regex heuristic, table extraction, single-turn only, etc.) — showing awareness of a system's limits is itself a signal of seniority

---

## Suggested order of attack

1. Phase 1 (golden set + eval runner + confusion matrix) — without this, nothing else is measurable, and it's the single highest-leverage piece for demonstrating expertise.
2. Phase 2 architecture cleanup, done *while* wiring the eval suite (dependency injection is much easier to justify once you're writing tests against it).
3. Phase 4 CI/lint, so the eval suite actually gates future changes.
4. Phase 3 and Phase 5 as time allows — good depth signals but lower priority than having real numbers.
