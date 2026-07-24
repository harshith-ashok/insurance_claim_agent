import argparse
import glob
import os
import re

import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from config import COLLECTION_NAME, DENSE_MODEL_NAME, DOCS_DIR, QDRANT_URL

SECTION_REGEX = re.compile(
    r"(?:Section|Clause|Article|Policy Provision)\s*([A-Za-z0-9\.]+)", re.IGNORECASE
)

CHILD_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)


def recreate_collection(client: QdrantClient, dense_model: SentenceTransformer):
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": models.VectorParams(
                size=dense_model.get_sentence_embedding_dimension(),
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )

    for field, schema in [
        ("metadata.section", models.PayloadSchemaType.KEYWORD),
        ("metadata.page", models.PayloadSchemaType.INTEGER),
        ("metadata.source", models.PayloadSchemaType.KEYWORD),
    ]:
        client.create_payload_index(
            collection_name=COLLECTION_NAME, field_name=field, field_schema=schema
        )


def parse_document(pdf_path: str, start_chunk_id: int):
    """Parse a single PDF into (chunk_text, metadata) tuples, section tracked across pages."""
    source = os.path.basename(pdf_path)
    print(f"Parsing {source}...")
    pages_data = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)

    chunks = []
    current_section = "N/A"
    chunk_id = start_chunk_id

    for page_idx, page in enumerate(pages_data):
        page_text = page["text"]
        page_num = page.get("metadata", {}).get("page_number", page_idx + 1)

        for chunk_text in CHILD_SPLITTER.split_text(page_text):
            match = SECTION_REGEX.search(chunk_text)
            if match:
                current_section = match.group(1).strip().rstrip(".")

            chunks.append(
                {
                    "text": chunk_text,
                    "metadata": {
                        "page": int(page_num),
                        "chunk_id": chunk_id,
                        "section": current_section,
                        "source": source,
                    },
                }
            )
            chunk_id += 1

    return chunks


def ingest(docs_dir: str):
    pdf_paths = sorted(glob.glob(os.path.join(docs_dir, "*.pdf")))
    if not pdf_paths:
        raise SystemExit(f"No PDF files found in '{docs_dir}'.")

    client = QdrantClient(url=QDRANT_URL)
    dense_model = SentenceTransformer(DENSE_MODEL_NAME)

    recreate_collection(client, dense_model)

    all_chunks = []
    for pdf_path in pdf_paths:
        all_chunks.extend(parse_document(pdf_path, start_chunk_id=len(all_chunks)))

    if not all_chunks:
        raise SystemExit("No text extracted from the provided PDFs.")

    print(f"Embedding {len(all_chunks)} chunks from {len(pdf_paths)} document(s)...")
    texts = [c["text"] for c in all_chunks]
    dense_vectors = dense_model.encode(
        texts, show_progress_bar=True, batch_size=32
    ).tolist()

    ids = [c["metadata"]["chunk_id"] for c in all_chunks]
    payloads = [{"text": c["text"], "metadata": c["metadata"]} for c in all_chunks]
    vectors = [
        {"dense": dense_vec, "sparse": models.Document(text=text, model="Qdrant/bm25")}
        for dense_vec, text in zip(dense_vectors, texts)
    ]

    print(f"Uploading {len(all_chunks)} chunks to '{COLLECTION_NAME}'...")
    client.upload_collection(
        collection_name=COLLECTION_NAME,
        vectors=vectors,
        payload=payloads,
        ids=ids,
        wait=True,
    )

    print("Ingestion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest policy PDFs into Qdrant.")
    parser.add_argument(
        "--docs-dir",
        default=DOCS_DIR,
        help=f"Directory containing PDF files to ingest (default: {DOCS_DIR})",
    )
    args = parser.parse_args()
    ingest(args.docs_dir)
