# Insurance Policy Q&A (RAG prototype)

A document-grounded Q&A system for insurance policy PDFs. Retrieves relevant
policy text via hybrid dense+sparse search in Qdrant, reranks it with a
cross-encoder, and asks an LLM to answer strictly from the retrieved context
with inline page/section citations.

## Current State vs. Roadmap

This is a working prototype, not a productionized service: four flat Python
scripts plus a pytest-based `eval/` harness, no package structure, and no API
layer. Everything under **Architecture** below is what actually runs today,
verified against `ingest.py` and `query.py`; **Evaluation** below has real,
measured numbers from that harness. Everything under **Roadmap** is
intentionally not built — those are the next steps toward structure-aware
chunking, runtime guardrails, and a service layer, tracked in `refactor.md`.

---

## Architecture (as implemented)

```
docs/*.pdf
     │
     ▼
pymupdf4llm.to_markdown(page_chunks=True)     — per-page markdown extraction
     │
     ▼
RecursiveCharacterTextSplitter                — sliding-window chunking
(chunk_size=1500, overlap=300)                  (not clause/header-aware)
     │
     ▼
Regex section tagging                          — Section|Clause|Article|Policy
(forward-carried across chunks/pages             Provision \d, last match
 within a document)                              carried forward as a heuristic
     │
     ▼
BAAI/bge-small-en-v1.5 (sentence-transformers) — dense embeddings, batch-encoded
Qdrant/bm25 (server-side)                      — sparse vectors
     │
     ▼
Qdrant collection "insurance_policies"          — dense + sparse vectors, full
(wiped and recreated on every `ingest.py` run)   collection rebuild, no incremental
                                                  ingestion
     │
   query time
     │
     ▼
Query embedded with the same bge-small-en-v1.5 model
     │
     ▼
Qdrant query_points: Prefetch(dense, limit=30) + Prefetch(sparse, limit=30)
fused server-side via models.RrfQuery (RRF), optional metadata filter
on section/source
     │
     ▼
cross-encoder/ms-marco-MiniLM-L-6-v2            — reranks every candidate,
                                                    always on; an optional
                                                    confidence gate exists
                                                    (config.CONFIDENCE_THRESHOLD)
                                                    but ships disabled — see
                                                    Evaluation
     │
     ▼
Top 4 reranked chunks → context block, tagged with source/page/section
     │
     ▼
gpt-oss:120b-cloud via Ollama                   — cloud-hosted model, streamed
Citation-only system prompt                       to stdout; Qdrant/Ollama calls
                                                    are try/except-wrapped, print
                                                    [ERROR] on failure instead of
                                                    raising
```

`config.py` is the single source of truth for shared constants
(`QDRANT_URL`, `COLLECTION_NAME`, `DOCS_DIR`, `DENSE_MODEL_NAME`,
`RERANKER_MODEL_NAME`, `LLM_MODEL_NAME`). `ingest.py` and `query.py` must use
the same `DENSE_MODEL_NAME` or query embeddings silently become incompatible
with stored vectors.

## Known Limitations

- **Section labels are a heuristic, not a structural parse.** The regex
  carries the last matched `Section|Clause|Article|Policy Provision \d`
  forward across chunks and pages within a document — any page without an
  explicit header inherits whatever section came before it, which can
  mislabel chunks.
- **Chunking is sliding-window, not structure-aware.** `RecursiveCharacterTextSplitter`
  can cut a clause mid-sentence or separate a clause number from its content.
- **Tables are flattened** by `pymupdf4llm`'s markdown conversion — multi-tier
  premium/benefit tables lose relational structure.
- **Reranking is always on** — no gate to skip it on high-confidence
  retrievals (a latency optimization, not yet implemented).
- **No *runtime* citation verification.** Nothing stops the model from citing
  a plausible-but-wrong `(Page X, Section Y)` before the answer reaches the
  user. `eval/` verifies citations *after the fact* against the golden set
  (see Evaluation) — that's a measurement, not a guardrail; a query-time check
  is still on the Roadmap.
- **Confidence gate exists but doesn't help.** `config.CONFIDENCE_THRESHOLD`
  can abstain deterministically below a score cutoff, but `eval/report.md`
  found answerable/unanswerable questions produce overlapping top-1
  cross-encoder scores on this corpus — any threshold that suppresses a
  hallucination also wrongly refuses several correct answers. Ships disabled.
- **Full destructive reingest.** `ingest.py` deletes and recreates the
  Qdrant collection on every run — there's no incremental/upsert path.
- **`qdrant_storage/` is checked into git** — it's regenerated destructively
  by `ingest.py`, so this is repo hygiene debt, not source of truth.

## Setup

### Prerequisites

- [Ollama](https://ollama.com) installed, running, and signed in
  (`ollama signin`) — generation uses a cloud-hosted Ollama model
- Docker (for Qdrant)
- Python 3.12 ([uv](https://docs.astral.sh/uv/) manages the virtualenv, see
  `.venv/pyvenv.cfg`)

### Install

```bash
uv pip install -r requirements.txt
```

Embeddings and reranking run locally via `sentence-transformers`
(`BAAI/bge-small-en-v1.5` and `cross-encoder/ms-marco-MiniLM-L-6-v2`,
downloaded automatically on first run). Pull the cloud LLM used for
generation:

```bash
ollama pull gpt-oss:120b-cloud
```

### Start Qdrant

```bash
docker run -d -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

### Ingest documents

```bash
python ingest.py            # ingests every *.pdf in docs/
python ingest.py --docs-dir <path>
```

This deletes and recreates the `insurance_policies` collection from every
PDF found — all documents present are ingested, not just one.

### Query

```bash
python test.py
```

runs a fixed set of example queries through `diagnostic_pipeline()` and
prints the retrieval diagnostics and streamed answer — there's no assertion
or scoring, it's a manual smoke test. To ask a single question interactively:

```python
from query import diagnostic_pipeline
diagnostic_pipeline(query="What are the key benefits of the policy?", target_source="icici-car.pdf")
```

`target_section` and `target_source` are optional filters.

## Tech Stack

| Layer            | Technology                          |
| ---------------- | ------------------------------------ |
| Document parsing | `pymupdf4llm`                       |
| Chunking         | `RecursiveCharacterTextSplitter` (LangChain), sliding-window |
| Section tagging  | Regex, forward-carried heuristic     |
| Dense embeddings | `BAAI/bge-small-en-v1.5` (sentence-transformers) |
| Sparse vectors   | `Qdrant/bm25`, server-side           |
| Vector store     | Qdrant (dense + sparse hybrid, RRF fusion via `RrfQuery`) |
| Reranking        | `cross-encoder/ms-marco-MiniLM-L-6-v2`, always on |
| Generation       | `gpt-oss:120b-cloud` via Ollama, streamed |

## Evaluation

`eval/` is a real pytest-based harness (Phase 1 of `refactor.md`) that calls
`query.run_query()` programmatically against a 53-question hand-built golden
set (`eval/golden_set.jsonl`) — no mocks, real Qdrant + real Ollama. See
`eval/README.md` for how to run it and `eval/report.md` for the full writeup;
current headline numbers on the two included policy PDFs:

| Metric | Value |
| --- | --- |
| Retrieval Recall@4 / Precision@4 / MRR (overall) | 88.8% / 33.1% / 0.840 |
| Citation accuracy (precision / recall) | 74.4% / 82.5% |
| Citation verification (claim re-embedded vs. cited chunk) | 86.8% |
| Abstention confusion matrix (of 53) | 38 correct, 12 correctly abstained, 1 hallucinated, 2 wrongly refused |
| Consistency (mean pairwise answer similarity, 5×5 sample) | 0.963 |
| RAGAS faithfulness / answer_relevancy / context_precision / context_recall | 0.825 / 0.825 / 0.900 / 0.900 |

`test.py` is still eyeball-only for quick manual spot-checks; it is not the
harness these numbers come from. One notable finding from building this
harness: a retrieval-confidence gate (`config.CONFIDENCE_THRESHOLD`) was
implemented and tested but found **not** to help on this corpus — see the
"Confidence gate analysis" section of `eval/report.md` — so it ships disabled
by default.

## Roadmap

Not implemented — tracked in detail in `refactor.md`:

- **Architecture refactor**: split into `ingestion/`/`retrieval/`/`inference/`/`eval/`/`api/`
  packages; replace `pymupdf4llm` with **Docling** for structure-preserving
  parsing and its `HybridChunker` for structure-aware chunking (fixes table
  flattening and clause fragmentation); replace the section-tagging regex
  with parser-derived section labels; add a *runtime* citation-verification
  guardrail that blocks ungrounded answers before they reach the user (the
  eval suite already verifies citations post-hoc — see Evaluation below —
  but nothing enforces it at query time yet); content-hash chunk IDs for
  incremental ingestion; dependency-injected clients instead of module-level
  singletons; pydantic models instead of raw dicts; `logging` instead of
  `print`.
- **Reliability**: per-stage latency instrumentation, token/cost tracking,
  incremental ingestion with collection versioning.
- **Engineering standards**: `ruff`/`mypy` via `pyproject.toml`, CI (lint +
  type-check + eval suite on PRs), pre-commit hooks, a small real FastAPI
  service (`/query` SSE-streaming, `/ingest` gated on content-hash), and
  removing `qdrant_storage/` from git in favor of a documented bootstrap.

None of the above is currently running. Do not assume behavior beyond what's
described in **Architecture (as implemented)** and **Evaluation**.

## License

MIT
