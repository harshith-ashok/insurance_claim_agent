from collections import Counter

from inference.guardrail import CITATION_RE, citation_claim_spans, parse_citations, verify_citations
from query import ABSTENTION_MESSAGE

__all__ = [
    "CITATION_RE",
    "parse_citations",
    "citation_claim_spans",
    "verify_citations",
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "citation_accuracy",
    "is_abstention",
    "classify_abstention",
    "answer_similarity",
    "citation_agreement",
]


def _matches_ground_truth(candidate_meta: dict, expected_citation: dict) -> bool:
    """Ground truth is matched on (source, page) only — section labels are a known
    heuristic and too noisy to use as a match key (see README Known Limitations)."""
    if candidate_meta.get("source") != expected_citation.get("source"):
        return False
    try:
        return int(candidate_meta.get("page")) == int(expected_citation.get("page"))
    except (TypeError, ValueError):
        return False


def recall_at_k(ranked_candidates: list[dict], expected_citations: list[dict], k: int) -> float | None:
    """Fraction of expected ground-truth chunks that appear in the top-k retrieved
    candidates. None when the question has no ground-truth chunk (unanswerable)."""
    if not expected_citations:
        return None
    top_k = ranked_candidates[:k]
    found = sum(
        1 for exp in expected_citations if any(_matches_ground_truth(c["meta"], exp) for c in top_k)
    )
    return found / len(expected_citations)


def precision_at_k(ranked_candidates: list[dict], expected_citations: list[dict], k: int) -> float | None:
    if not expected_citations:
        return None
    top_k = ranked_candidates[:k]
    if not top_k:
        return 0.0
    relevant = sum(
        1 for c in top_k if any(_matches_ground_truth(c["meta"], exp) for exp in expected_citations)
    )
    return relevant / len(top_k)


def mrr(ranked_candidates: list[dict], expected_citations: list[dict]) -> float | None:
    """Reciprocal rank of the first retrieved candidate that matches any expected
    ground-truth chunk, over the full ranked candidate list (not just top-k)."""
    if not expected_citations:
        return None
    for i, c in enumerate(ranked_candidates, start=1):
        if any(_matches_ground_truth(c["meta"], exp) for exp in expected_citations):
            return 1.0 / i
    return 0.0


def citation_accuracy(parsed_citations: list[dict], expected_citations: list[dict]) -> dict | None:
    """Precision/recall of cited pages in the generated answer against expected
    citations. None when the question has no expected citations (unanswerable)."""
    if not expected_citations:
        return None
    expected_pages = {int(e["page"]) for e in expected_citations}
    cited_pages = {c["page"] for c in parsed_citations}
    if not cited_pages:
        return {"precision": 0.0, "recall": 0.0, "cited_pages": [], "expected_pages": sorted(expected_pages)}
    correct = cited_pages & expected_pages
    return {
        "precision": len(correct) / len(cited_pages),
        "recall": len(correct) / len(expected_pages),
        "cited_pages": sorted(cited_pages),
        "expected_pages": sorted(expected_pages),
    }


def is_abstention(answer_text: str) -> bool:
    if not answer_text:
        return True
    return ABSTENTION_MESSAGE.lower() in answer_text.lower()


def classify_abstention(expected_answer: str, actual_answer: str) -> str:
    """Abstention confusion-matrix bucket for a single run:
    correctly_answered / correctly_abstained / hallucinated / wrongly_refused.
    `expected_answer == ABSTENTION_MESSAGE` is the ground-truth signal for whether
    the question should have been refused, independent of which golden-set bucket
    it came from (the 'ambiguous' bucket mixes both cases)."""
    expected_abstain = (expected_answer or "").strip() == ABSTENTION_MESSAGE
    actual_abstain = is_abstention(actual_answer)
    if expected_abstain and actual_abstain:
        return "correctly_abstained"
    if expected_abstain and not actual_abstain:
        return "hallucinated"
    if not expected_abstain and actual_abstain:
        return "wrongly_refused"
    return "correctly_answered"


def answer_similarity(dense_model, answer_a: str, answer_b: str) -> float:
    if not answer_a or not answer_b:
        return 0.0
    from sentence_transformers.util import cos_sim

    emb_a = dense_model.encode(answer_a, convert_to_tensor=True)
    emb_b = dense_model.encode(answer_b, convert_to_tensor=True)
    return float(cos_sim(emb_a, emb_b))


def citation_agreement(parsed_citations_runs: list[list[dict]]) -> float:
    """Fraction of repeated runs whose cited-page set matches the most common
    cited-page set across all runs — a consistency signal independent of exact
    wording."""
    if not parsed_citations_runs:
        return 0.0
    page_sets = [frozenset(c["page"] for c in citations) for citations in parsed_citations_runs]
    _, count = Counter(page_sets).most_common(1)[0]
    return count / len(page_sets)
