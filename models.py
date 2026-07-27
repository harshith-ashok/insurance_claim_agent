from pydantic import BaseModel


class ChunkMetadata(BaseModel):
    page: int
    chunk_id: str
    section: str
    source: str


class Chunk(BaseModel):
    """A single ingested unit of text, produced by ingestion/parsing.py."""

    text: str
    metadata: ChunkMetadata


class RetrievalCandidate(BaseModel):
    """A chunk retrieved for a query, carrying its fusion/rerank scores.

    `model_dump()` on this is the dict shape `query.run_query()` has always
    returned (`text`/`rrf_score`/`meta`/`cross_score`) — eval/ and any other
    external caller keep working against plain dicts; pydantic just removes
    stringly-typed-key bugs while the candidate list is being assembled.
    """

    text: str
    rrf_score: float
    meta: ChunkMetadata
    cross_score: float | None = None


class CitationVerification(BaseModel):
    claim: str
    citation: dict
    verified: bool
    similarity: float | None = None
    reason: str | None = None


class QueryResult(BaseModel):
    """Internal representation of a run_query() call; converted to a plain
    dict via model_dump() at the run_query() boundary to preserve the
    existing dict-shaped contract eval/ and test.py already depend on."""

    query: str
    target_section: str | None = None
    target_source: str | None = None
    error: str | None = None
    candidates: list[RetrievalCandidate] = []
    context_candidates: list[RetrievalCandidate] = []
    top_score: float | None = None
    gated: bool = False
    answer: str | None = None
    citation_verifications: list[CitationVerification] = []
