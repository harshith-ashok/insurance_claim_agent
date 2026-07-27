# Policy Ledger — frontend

Vue 3 + TypeScript + Vite UI for the insurance-policy RAG system in the
parent directory. Streams a real query through the FastAPI backend
(`api/main.py`) via Server-Sent Events — retrieved chunks and the retrieval
trace appear as soon as they're known, the answer streams token by token,
then citations become clickable once the backend's citation-verification
guardrail has run.

## Running

This needs the backend up first: Qdrant + a populated collection + the
FastAPI service (see the repo root `README.md`/`CLAUDE.md`):

```bash
# from the repo root, in one terminal
uv pip install -r requirements.txt
docker run -d -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant   # if not already running
python ingest.py    # if the collection isn't already populated
uvicorn api.main:app --port 8000
```

```bash
# from frontend/, in another terminal
npm install
npm run dev
```

Open the printed `localhost:5173` URL. The frontend talks to
`http://localhost:8000` by default; override with a `VITE_API_BASE_URL` env
var (e.g. in `frontend/.env.local`) if the API runs elsewhere.

## Structure

- `src/composables/useQueryDemo.ts` — all the API-calling logic (`EventSource`
  against `/query`), state, and the client-side parsing that turns the raw
  answer text + citation-verification results into clickable, source-linked
  citations. Despite the name (kept from before the real backend existed),
  this is no longer a demo/mock.
- `src/components/` — `QueryCard` (input + filters), `AnswerPanel` (streamed
  answer text, citations), `TracePanel` (retrieval trace + reranked chunk
  list), `DiagStrip` (static eval numbers from the repo root's `eval/report.md`).

## Known gap

`DiagStrip`'s numbers are hardcoded from the last `eval/report.md` run, not
fetched live — there's no `/eval` endpoint. The sidebar's "Ingestion" and
"Evaluation" nav items are also inert; only the query path is wired up.
