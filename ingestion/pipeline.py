import glob
import logging
import os

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from config import COLLECTION_NAME, DENSE_MODEL_NAME, DOCS_DIR, QDRANT_URL
from ingestion.parsing import parse_document
from models import Chunk

logger = logging.getLogger(__name__)


def recreate_collection(client: QdrantClient, dense_model: SentenceTransformer) -> None:
    dense_dim = dense_model.get_sentence_embedding_dimension()
    if dense_dim is None:
        raise ValueError(f"Could not determine embedding dimension for {DENSE_MODEL_NAME}")

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": models.VectorParams(
                size=dense_dim,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )

    for field, schema in [
        ("metadata.section", models.PayloadSchemaType.KEYWORD),
        ("metadata.page", models.PayloadSchemaType.INTEGER),
        ("metadata.source", models.PayloadSchemaType.KEYWORD),
    ]:
        client.create_payload_index(collection_name=COLLECTION_NAME, field_name=field, field_schema=schema)


def ingest(docs_dir: str = DOCS_DIR) -> None:
    pdf_paths = sorted(glob.glob(os.path.join(docs_dir, "*.pdf")))
    if not pdf_paths:
        raise SystemExit(f"No PDF files found in '{docs_dir}'.")

    client = QdrantClient(url=QDRANT_URL)
    dense_model = SentenceTransformer(DENSE_MODEL_NAME)

    recreate_collection(client, dense_model)

    all_chunks: list[Chunk] = []
    for pdf_path in pdf_paths:
        all_chunks.extend(parse_document(pdf_path))

    if not all_chunks:
        raise SystemExit("No text extracted from the provided PDFs.")

    logger.info("Embedding %d chunks from %d document(s)...", len(all_chunks), len(pdf_paths))
    texts = [c.text for c in all_chunks]
    dense_vectors = dense_model.encode(texts, show_progress_bar=True, batch_size=32).tolist()

    ids = [c.metadata.chunk_id for c in all_chunks]
    payloads = [{"text": c.text, "metadata": c.metadata.model_dump()} for c in all_chunks]
    vectors = [
        {"dense": dense_vec, "sparse": models.Document(text=text, model="Qdrant/bm25")}
        for dense_vec, text in zip(dense_vectors, texts, strict=True)
    ]

    logger.info("Uploading %d chunks to '%s'...", len(all_chunks), COLLECTION_NAME)
    client.upload_collection(
        collection_name=COLLECTION_NAME,
        vectors=vectors,
        payload=payloads,
        ids=ids,
        wait=True,
    )

    logger.info("Ingestion complete.")
