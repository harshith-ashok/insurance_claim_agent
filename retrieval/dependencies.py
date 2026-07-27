from dataclasses import dataclass

from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

from config import DENSE_MODEL_NAME, QDRANT_URL, RERANKER_MODEL_NAME


@dataclass
class PipelineDependencies:
    """Injectable holder for the pipeline's three live clients. Replaces bare
    module-level globals in query.py — run_query() can now accept an
    alternate instance instead of always reaching for shared process-wide
    state, without changing the default (eager, constructed once at import)
    behavior callers already rely on."""

    client: QdrantClient
    dense_model: SentenceTransformer
    reranker: CrossEncoder

    @classmethod
    def build(cls) -> "PipelineDependencies":
        return cls(
            client=QdrantClient(url=QDRANT_URL),
            dense_model=SentenceTransformer(DENSE_MODEL_NAME),
            reranker=CrossEncoder(RERANKER_MODEL_NAME),
        )
