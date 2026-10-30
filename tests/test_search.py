from qdrant_client import models as qm

from models import ChunkMetadata, RetrievalCandidate
from retrieval.search import build_filter, deduplicate


def _cand(text, page, source="a.pdf", cid="x"):
    return RetrievalCandidate(
        text=text,
        rrf_score=1.0,
        meta=ChunkMetadata(page=page, chunk_id=cid, section="I", source=source),
    )


def test_build_filter_none_when_no_args():
    assert build_filter() is None


def test_build_filter_section_and_source():
    f = build_filter("III", "icici-car.pdf")
    assert isinstance(f, qm.Filter)
    assert {c.key for c in f.must} == {"metadata.section", "metadata.source"}


def test_deduplicate_drops_same_page_near_duplicate():
    a = _cand("The insurer shall pay 15 lakhs in aggregate.", 1, cid="1")
    b = _cand("The insurer shall pay 15 lakhs in aggregate.", 1, cid="2")
    assert deduplicate([a, b]) == [a]


def test_deduplicate_keeps_identical_text_on_different_pages():
    a = _cand("Same table text", 1, cid="1")
    b = _cand("Same table text", 2, cid="2")
    assert len(deduplicate([a, b])) == 2
