import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from api.streaming import stream_query
from config import COLLECTION_NAME
from logging_config import configure_logging
from query import client

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="Policy Ledger API")

# Vite's default dev server origin. Small and real for local dev — if this
# ever needs to serve a deployed frontend, read the allowed origin from an
# env var instead of hardcoding a second one here.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    try:
        collection_exists = client.collection_exists(COLLECTION_NAME)
    except Exception as exc:
        return {"qdrant": "down", "error": str(exc)}
    return {"qdrant": "up", "collection": COLLECTION_NAME, "collection_exists": collection_exists}


@app.get("/query")
def query_endpoint(
    query: str, target_section: str | None = None, target_source: str | None = None
) -> StreamingResponse:
    """SSE stream of a single run_query() call. Events, in order:
    `retrieved` (context candidates + timings so far, fires once — skipped
    entirely if retrieval found zero candidates) → `generation_start` (only
    if not gated) → zero or more `token` → exactly one of `done` (full
    result + `abstained` flag) or `query_error` (pipeline failure — named to
    avoid colliding with EventSource's own reserved `error` event, which
    fires separately on a connection-level failure).
    """
    return StreamingResponse(
        stream_query(query, target_section, target_source),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
