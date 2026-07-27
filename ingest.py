import argparse

from config import DOCS_DIR
from ingestion.pipeline import ingest
from logging_config import configure_logging

if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser(description="Ingest policy PDFs into Qdrant.")
    parser.add_argument(
        "--docs-dir",
        default=DOCS_DIR,
        help=f"Directory containing PDF files to ingest (default: {DOCS_DIR})",
    )
    args = parser.parse_args()
    ingest(args.docs_dir)
