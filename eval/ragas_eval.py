import os


def get_ragas_components():
    """Lazily builds RAGAS metric instances wired to the same models the pipeline
    already uses: the Ollama LLM (via its OpenAI-compatible endpoint, since ragas
    0.4's metric classes require an instructor-compatible client) and the local
    bge-small-en-v1.5 embeddings. Returns None if ragas/openai aren't installed,
    so callers can skip cleanly instead of crashing the whole eval run."""
    try:
        from openai import AsyncOpenAI
        from ragas.embeddings import HuggingFaceEmbeddings
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecisionWithReference,
            ContextRecall,
            Faithfulness,
        )
    except ImportError:
        return None

    from config import DENSE_MODEL_NAME, LLM_MODEL_NAME

    base_url = os.environ.get("OLLAMA_OPENAI_BASE_URL", "http://localhost:11434/v1")
    client = AsyncOpenAI(base_url=base_url, api_key="ollama")
    llm = llm_factory(LLM_MODEL_NAME, provider="openai", client=client)
    embeddings = HuggingFaceEmbeddings(model=DENSE_MODEL_NAME)

    return {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": ContextPrecisionWithReference(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }
