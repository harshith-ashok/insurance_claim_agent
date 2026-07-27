class PipelineError(Exception):
    """Base class for errors raised by the ingestion/retrieval/inference pipeline."""


class QdrantConnectionError(PipelineError):
    """Qdrant was unreachable or returned an error during a query/upload."""


class LLMInferenceError(PipelineError):
    """The Ollama chat call failed or returned an unusable response."""
