import json
import logging
import queue
import threading
from collections.abc import Iterator

from inference.llm import ABSTENTION_MESSAGE
from query import run_query

logger = logging.getLogger(__name__)

_SENTINEL = object()


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def stream_query(query: str, target_section: str | None, target_source: str | None) -> Iterator[str]:
    """Runs run_query() on a background thread and yields SSE frames as its
    existing on_retrieved/on_generation_start/on_token callbacks fire, so the
    frontend gets retrieved chunks and streamed tokens progressively instead
    of waiting for the whole pipeline to finish. Terminates with either a
    "done" or "error" event, always followed by the connection closing.
    """
    events: queue.Queue = queue.Queue()

    def on_retrieved(result: dict) -> None:
        events.put(
            _sse(
                "retrieved",
                {
                    "candidates": result["context_candidates"],
                    "top_score": result["top_score"],
                    "gated": result["gated"],
                    "timings": result["timings"],
                },
            )
        )

    def on_generation_start() -> None:
        events.put(_sse("generation_start", {}))

    def on_token(token: str) -> None:
        events.put(_sse("token", {"token": token}))

    def worker() -> None:
        try:
            result = run_query(
                query,
                target_section=target_section,
                target_source=target_source,
                on_retrieved=on_retrieved,
                on_generation_start=on_generation_start,
                on_token=on_token,
            )
            if result["error"]:
                events.put(_sse("query_error", {"error": result["error"]}))
            else:
                abstained = result["gated"] or result["answer"] == ABSTENTION_MESSAGE
                events.put(_sse("done", {**result, "abstained": abstained}))
        except Exception as exc:  # last-resort guard — the stream must always terminate cleanly
            logger.exception("Unhandled error streaming query %r", query)
            events.put(_sse("query_error", {"error": str(exc)}))
        finally:
            events.put(_SENTINEL)

    threading.Thread(target=worker, daemon=True).start()

    while True:
        item = events.get()
        if item is _SENTINEL:
            break
        yield item
