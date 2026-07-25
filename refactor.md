# Refactor & Hardening Plan

Goal: turn this from a working prototype into a portfolio-grade RAG system —
one that demonstrates measured accuracy/consistency/abstention behavior, and
follows enterprise engineering standards (testing, observability, structure).

Priority tags: **P0** = do first / highest resume impact, **P1** = important,
**P2** = polish.

---

## Phase 0 — Fix the credibility gap (P0, do this FIRST — cheapest fix, highest risk)

The README currently describes a different, more sophisticated system than the
one that exists (FastAPI, RAGAS with populated metrics, conditional reranking,
`mistral:7b`, header-aware chunking, `bge-m3`) — none of which is implemented.
This is not a documentation nitpick. To a technical reader, a README that
describes unimplemented architecture as if it exists reads as either
dishonesty or not understanding your own system — either read is disqualifying
in an interview, and it's a bigger liability than any of the actual design
drawbacks below. Fix this before anything else; it's also the cheapest item
on this whole list.

- [ ] Rewrite `README.md` to describe **only** what's actually running today: 4 flat scripts, `pymupdf4llm` + `RecursiveCharacterTextSplitter`, `bge-small-en-v1.5`, always-on reranking, `gpt-oss:120b-cloud` via Ollama, no API layer
- [ ] Move every aspirational item (FastAPI, RAGAS, conditional reranking, hierarchical chunking, local LLM, `ingestion/`/`retrieval/`/`inference/` packages) into an explicit **"Roadmap"** section, clearly labeled as not-yet-built
- [ ] Remove or clearly mark the metrics table so it doesn't show unpopulated `—` rows next to claimed capabilities — either delete it until Phase 1 produces real numbers, or label it "target metrics, not yet measured"
- [ ] Add a one-paragraph "Current State vs. Roadmap" framing at the top of the README — this framing itself is a seniority signal (knowing the gap between what you built and what you'd build next is more impressive than hiding it)

## Phase 1 — Evaluation framework (P0, the core technical differentiator)

This is the single highest-leverage phase. Right now there is no measurement
of accuracy, consistency, or abstention — `test.py` only prints output for a
human to eyeball. "I built a RAG system" is a common resume line; "I measured
faithfulness/precision/citation-accuracy against a hand-labeled set and
iterated on the numbers" is the line that differentiates.

- [ ] Build a golden evaluation dataset (`eval/golden_set.jsonl`) with three buckets, each entry carrying `question`, `expected_answer`, `expected_citations` (page/section/source), and `bucket`:
  - [ ] **Answerable** — questions with a verifiable answer in the PDFs (pull ~15-20 per document from real clauses)
  - [ ] **Unanswerable** — questions plausible for an insurance policy but not covered by these specific documents (tests false-positive/hallucination risk)
  - [ ] **Ambiguous / multi-hop / adversarial** — cross-section reasoning, long-tail clause language, near-duplicate-chunk traps
- [ ] Write an `eval/` package with a real test runner (pytest) that loads the golden set and calls the pipeline programmatically — no more manual script + eyeballing
- [ ] Implement retrieval metrics: Recall@k, Precision@k, MRR — did the ground-truth chunk make it into the top-4 context at all
- [ ] Implement citation accuracy: parse `(Page X, Section Y)` tokens out of the generated answer and diff against expected citations
- [ ] Implement a citation *verification* pass, not just citation *presence* — see Phase 2's guardrail-layer task; a model can cite a plausible-sounding page/section that doesn't actually support the claim, and naive citation-format checking won't catch that
- [ ] Implement an abstention confusion matrix — for each bucket, classify each run as: correctly answered / correctly abstained / hallucinated (answered when it shouldn't have) / wrongly refused (abstained when answerable)
- [ ] Wire in RAGAS for `context_precision`, `context_recall`, `faithfulness`, `answer_relevance` — with actual populated values, replacing the current unpopulated README table
- [ ] Implement a consistency test — run each golden question N times (e.g. N=5) and measure answer/citation stability (embedding similarity across runs, citation agreement rate)
- [ ] Add tracing via Langfuse or Arize Phoenix so each eval run is inspectable per-query — this is what turns "it works" into "I can diagnose why a specific answer was wrong: retrieval miss vs. reranking miss vs. generation hallucination"
- [ ] Generate a results report (`eval/report.md` or a small HTML/artifact) with real numbers — this becomes the README metrics table and the resume evidence
- [ ] Add a retrieval-confidence gate: if the top reranked cross-encoder score is below a tuned threshold, skip the LLM call and return "not available" deterministically instead of relying solely on prompt compliance — then measure how much this improves the abstention confusion matrix

## Phase 2 — Architecture refactor (P0/P1)

Stack decisions for this phase (rationale below in "Stack Decisions"):
swap `pymupdf4llm` → **Docling**, keep **Qdrant**, add a citation-verification
guardrail, add content-hash chunk IDs, consider **LlamaIndex** for
orchestration if rebuilding the retrieval layer.

- [ ] Split into a real package: `ingestion/`, `retrieval/`, `inference/`, `eval/`, `api/` — matches what the (now-corrected) README roadmap promises
- [ ] Replace `pymupdf4llm` with **Docling** for PDF parsing — it preserves table structure as structured objects instead of flattened markdown (directly fixes the table-flattening failure mode on premium/benefit tables), and its `HybridChunker` does structure-aware chunking natively
- [ ] Replace the sliding-window `RecursiveCharacterTextSplitter` with structure-aware chunking (Docling's `HybridChunker`, or equivalent) — sliding-window chunking currently cuts clauses mid-sentence and separates a clause number from its content, which is the exact top RAG failure mode for legal/policy text that the (corrected) README should name as solved, not just described
- [ ] Replace the forward-carrying section-tagging regex with structure-derived section labels from the parser — the current heuristic mislabels any page without an explicit header with whatever section came before, which is a structural crack in a system whose entire value proposition is citation accuracy
- [ ] Add a citation-verification guardrail pass: after generation, re-embed each cited claim and check it against the specific chunk it claims to cite (cosine similarity threshold or a cheap NLI model) — nothing currently stops the model from citing a plausible but wrong `(Page X, Section Y)`, which is "retrieval failure masquerading as grounding"
- [ ] Generate deterministic chunk IDs from a content hash (e.g. hash of source+page+text) instead of positional integers — this makes re-ingesting unchanged pages a no-op and is what actually enables incremental ingestion, essentially for free
- [ ] Replace module-level global singletons in `query.py` (client, dense_model, reranker) with dependency-injected objects/classes — needed for unit testing without live services
- [ ] Add pydantic models for `Chunk`, `RetrievalCandidate`, `QueryResult` instead of raw dicts passed around by string key
- [ ] Replace `print()` with the standard `logging` module — structured logs with levels (DEBUG for diagnostics, INFO for pipeline steps, ERROR for failures)
- [ ] Replace bare `except Exception: print(...)` with specific exception types (`QdrantConnectionError`, `LLMInferenceError`, etc.) and proper logging, still degrading gracefully but traceable
- [ ] Move hardcoded retrieval params (prefetch limit=30, RRF limit=15, top-k context=4, chunk size/overlap) into `config.py` as named, documented, tunable constants — then justify each via Phase 1 eval results, not guesswork
- [ ] Deduplicate near-identical chunks in the reranked candidate list before taking top-4 (overlap window currently allows near-duplicates to occupy multiple context slots)
- [ ] (Optional, bigger lift) Evaluate migrating orchestration to **LlamaIndex** if the retrieval layer is being substantially rebuilt anyway — see rationale below; not required if the hand-rolled Qdrant pipeline stays as-is, since it already works and rewriting working retrieval logic just to use a framework is not automatically a win

## Phase 3 — Reliability & observability (P1)

- [ ] Add latency instrumentation around each pipeline stage (embed, retrieve, rerank, LLM call) and log/report them
- [ ] Add basic token/cost tracking for the cloud LLM calls
- [ ] Add incremental ingestion support using the content-hash chunk IDs from Phase 2 (upsert changed/added PDFs only) instead of full destructive collection wipe every run; keep the full-rebuild path as an explicit `--force` flag
- [ ] Add a lightweight collection-versioning scheme (e.g. alias swap) so re-ingest doesn't leave a query-time window with an empty/partial collection
- [ ] Add golden-set eval cases specifically targeting table-heavy pages (premium schedules, benefit tables) to verify the Docling migration actually fixed table handling, not just assumed to

## Phase 4 — Enterprise coding standards (P1/P2)

- [ ] Add `pyproject.toml` with `ruff` (lint) and `mypy` (type check) configured; add type hints across all functions
- [ ] Add a GitHub Actions CI workflow: lint, type-check, run the Phase 1 eval suite on every PR
- [ ] Add pre-commit hooks for lint/format
- [ ] Add docstrings (one style, consistently — Google or NumPy) to all public functions
- [ ] Add a `Dockerfile`/`docker-compose.yml` covering the app + Qdrant for reproducible setup
- [ ] Add a real FastAPI service: one `/query` SSE-streaming endpoint, one `/ingest` endpoint gated behind a content-hash check (see Phase 2) so re-ingesting unchanged docs is a no-op — small and real, not a scaffold. This is what makes the "production-grade" framing true rather than aspirational.
- [ ] Stop committing `qdrant_storage/` binaries to git — it's regenerated destructively on every ingest run, which is a repo-hygiene red flag in a portfolio project; add it to `.gitignore` and document the `docker run` + `python ingest.py` bootstrap instead
- [ ] Remove the always-on reranking inefficiency: skip reranking when the retrieval-confidence gate (Phase 1) already indicates a clear top match, reranking only when candidates are close — real latency win, not just a correctness fix

## Phase 5 — Documentation & presentation (P2, resume polish)

- [ ] Add the Phase 1 eval report numbers to the README as a results table (recall@k, citation accuracy, abstention confusion matrix, consistency score) — concrete metrics read far stronger than architecture description alone
- [ ] Add an architecture diagram (ingestion pipeline + query pipeline) to the README
- [ ] Add a short "known limitations" section (table extraction edge cases, single-turn only, etc.) — showing awareness of a system's limits is itself a signal of seniority
- [ ] Write a short ADR-style note (or fold into README) explaining *why* Docling/Qdrant/RAGAS/Langfuse were chosen over alternatives (LlamaIndex vs. LangChain, Docling vs. pymupdf4llm) — demonstrating evaluated tradeoffs, not first-tutorial defaults, is exactly what a senior-engineering reviewer is looking for

---

## Stack Decisions (rationale, for the ADR note in Phase 5)

- **Orchestration**: keep the hand-rolled Qdrant pipeline unless doing a substantial retrieval-layer rewrite anyway. If rebuilding: **LlamaIndex** over LangChain for this specific use case — stronger native support for hierarchical/sentence-window retrieval, sub-question query engines for multi-hop (directly addresses the cross-document synthesis failure mode), first-class structured-table integration. LangChain (LCEL + LangGraph) is defensible if resume recognizability matters more — just be able to articulate the tradeoff in an interview, don't default to it silently.
- **Parsing**: **Docling** over `pymupdf4llm` — preserves table structure as structured objects rather than flattened markdown, and its `HybridChunker` gives structure-aware chunking without a hand-rolled regex.
- **Evaluation**: **RAGAS** for faithfulness/precision/relevance metrics, plus **Langfuse** or **Arize Phoenix** for per-query tracing — metrics tell you something's wrong, tracing tells you why (retrieval miss vs. generation hallucination).
- **Retrieval/store**: keep **Qdrant** — legitimately a good choice already, no need to change it. Pair with content-hash chunk IDs so incremental ingestion falls out for free.
- **Serving**: **FastAPI**, small and real — one `/query` SSE-streaming endpoint, one `/ingest` endpoint gated behind a hash-check.
- **Guardrails**: a lightweight citation-verification pass (re-embed each cited claim, check similarity against the specific chunk it claims to cite) — demonstrates understanding of grounding failure modes rather than just naming them in prose.

---

## Suggested order of attack

1. **Phase 0** (README honesty pass) — cheapest fix, removes the single biggest credibility risk, do this in an afternoon before anything else.
2. **Phase 1** (golden set + eval runner + confusion matrix) — without this, nothing else is measurable, and it's the highest-leverage piece for demonstrating expertise.
3. **Phase 2** architecture cleanup (Docling, citation verification, content-hash IDs), done *while* wiring the eval suite so every change is validated against real numbers rather than guesswork.
4. **Phase 4** CI/lint/FastAPI, so the eval suite actually gates future changes and the project is demoable as a service.
5. **Phase 3** and **Phase 5** as time allows — good depth signals but lower priority than having real numbers and an honest README.
