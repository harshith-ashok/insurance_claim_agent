QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "insurance_policies"

DOCS_DIR = "docs"

DENSE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL_NAME = "gpt-oss:120b-cloud"

# Minimum top-1 cross-encoder score required to call the LLM at all. Below this,
# run_query() abstains deterministically instead of relying on prompt compliance.
# None disables the gate. Left disabled: eval/gate_analysis.py found answerable and
# unanswerable questions produce overlapping top-1 scores on this corpus, so any
# threshold that suppresses the one observed hallucination also wrongly refuses
# ~11 correct answers — see "Confidence gate analysis" in eval/report.md.
CONFIDENCE_THRESHOLD = None
