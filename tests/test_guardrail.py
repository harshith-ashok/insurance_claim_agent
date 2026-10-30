from inference.guardrail import citation_claim_spans, parse_citations


def test_parse_citations_extracts_page_and_section():
    text = "Limit is 15 lakhs (Page 2, Section III). Also excluded (page 5, Section B-1)."
    assert parse_citations(text) == [
        {"page": 2, "section": "III"},
        {"page": 5, "section": "B-1"},
    ]


def test_parse_citations_empty():
    assert parse_citations("") == []
    assert parse_citations("no citations here") == []


def test_claim_span_uses_preceding_line():
    text = "Intro line\n- Covers theft (Page 1, Section II)"
    [(claim, cite)] = citation_claim_spans(text)
    assert claim == "Covers theft"
    assert cite == {"page": 1, "section": "II"}
