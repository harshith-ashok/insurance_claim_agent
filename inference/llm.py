import logging

import ollama

from config import LLM_MODEL_NAME
from exceptions import LLMInferenceError

logger = logging.getLogger(__name__)

ABSTENTION_MESSAGE = "This information is not available in the provided policy documents."

SYSTEM_PROMPT = (
    "You are an insurance system auditor. Provide highly precise answers derived ONLY from the provided text context.\n"
    "Rules:\n"
    "1. Every single assertion must include an inline location token explicitly showing page numbers and sections, matching this exact format: (Page X, Section Y).\n"
    f"2. If the context does not explicitly contain the details needed to form a direct answer, state exactly: "
    f"'{ABSTENTION_MESSAGE}' Do not extrapolate.\n"
    "3. Explicitly enumerate all benefits, requirements, or exclusions mentioned in the context without truncation."
)


def generate(query: str, context_str: str, on_token=None) -> str:
    """Streams the LLM's answer, optionally firing on_token(str) per chunk,
    and returns the full concatenated text."""
    try:
        stream = ollama.chat(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context_str}\n\nQuery: {query}"},
            ],
            stream=True,
        )
        tokens = []
        for chunk in stream:
            token = chunk["message"]["content"]
            tokens.append(token)
            if on_token:
                on_token(token)
        return "".join(tokens)
    except Exception as exc:
        logger.exception("LLM inference failed")
        raise LLMInferenceError(str(exc)) from exc
