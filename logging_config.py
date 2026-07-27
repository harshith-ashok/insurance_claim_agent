import logging

_configured = False


def configure_logging(level: int = logging.INFO) -> None:
    """Call once from a script entry point (ingest.py, test.py __main__ blocks).
    Library modules (ingestion/, retrieval/, inference/, query.py) only ever
    call logging.getLogger(__name__) — they never configure handlers, so
    importing this package as a library doesn't clobber a host app's logging."""
    global _configured
    if _configured:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    _configured = True
