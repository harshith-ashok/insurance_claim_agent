import re

CITATION_RE = re.compile(r"\(Page\s+(\d+),\s*Section\s+([^)]*)\)", re.IGNORECASE)


def parse_citations(answer_text: str) -> list[dict]:
    """Extract (Page X, Section Y) tokens from a generated answer."""
    if not answer_text:
        return []
    return [
        {"page": int(m.group(1)), "section": m.group(2).strip()}
        for m in CITATION_RE.finditer(answer_text)
    ]


def citation_claim_spans(answer_text: str) -> list[tuple[str, dict]]:
    """Pair each citation token with the claim text immediately preceding it
    (used by verify_citations to check the claim against the cited chunk)."""
    if not answer_text:
        return []
    spans = []
    last_end = 0
    for m in CITATION_RE.finditer(answer_text):
        preceding = answer_text[last_end : m.start()]
        claim_text = preceding.splitlines()[-1].strip(" -*•\t") if preceding.strip() else ""
        spans.append((claim_text, {"page": int(m.group(1)), "section": m.group(2).strip()}))
        last_end = m.end()
    return spans


def verify_citations(
    answer_text: str, context_candidates: list[dict], dense_model, similarity_threshold: float = 0.35
) -> list[dict]:
    """Citation *verification*, not just presence: re-embed the claim text next
    to each citation and check it against the specific chunk(s) on that
    cited page, rather than trusting that a plausible-looking (Page X,
    Section Y) token is actually grounded. Flags citations whose claim
    doesn't semantically match any chunk actually retrieved for that page.

    Runtime guardrail (query.run_query logs a warning for anything
    unverified) and the eval suite (eval/test_citations.py, scoring against
    the golden set) both call this same function — one implementation, not
    two copies that can drift apart.
    """
    spans = citation_claim_spans(answer_text)
    if not spans:
        return []

    from sentence_transformers.util import cos_sim

    page_to_candidates: dict[int, list[dict]] = {}
    for c in context_candidates:
        page_to_candidates.setdefault(int(c["meta"]["page"]), []).append(c)

    results = []
    for claim_text, citation in spans:
        candidates_for_page = page_to_candidates.get(citation["page"], [])
        if not candidates_for_page or not claim_text:
            results.append(
                {
                    "claim": claim_text,
                    "citation": citation,
                    "verified": False,
                    "similarity": None,
                    "reason": "no_matching_chunk_in_context" if not candidates_for_page else "empty_claim",
                }
            )
            continue
        claim_emb = dense_model.encode(claim_text, convert_to_tensor=True)
        best_sim = max(
            float(cos_sim(claim_emb, dense_model.encode(c["text"], convert_to_tensor=True)))
            for c in candidates_for_page
        )
        results.append(
            {
                "claim": claim_text,
                "citation": citation,
                "verified": best_sim >= similarity_threshold,
                "similarity": best_sim,
            }
        )
    return results
