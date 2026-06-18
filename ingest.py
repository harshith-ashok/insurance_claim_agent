import os
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from tqdm import tqdm

# --- 1. CONFIGURATION ---
PDF_PATH = "docs/icici-life.pdf" # Replace with your insurance document path
COLLECTION_NAME = "insurance_policies"
DENSE_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # Low-latency, high-precision dense model

client = QdrantClient(url="http://localhost:6333")
dense_model = SentenceTransformer(DENSE_MODEL_NAME)

# --- 2. PREPARE QDRANT COLLECTION (HYBRID SUB-SYSTEM) ---
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": models.VectorParams(
                size=dense_model.get_sentence_embedding_dimension(),
                distance=models.Distance.COSINE
            )
        },
        # Enable native sparse vectors for BM25/Lexical fallback matching
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                modifier=models.Modifier.IDF
            )
        }
    )
    # Create payload index for pre-filtering optimizations
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="metadata.section",
        field_schema=models.PayloadSchemaType.KEYWORD
    )
    print(f"Collection '{COLLECTION_NAME}' initialized with hybrid indexes.")

# --- 3. PERSISTENT MARKDOWN PARSING ---
print(f"Parsing structural layers from {PDF_PATH}...")
md_text = pymupdf4llm.to_markdown(PDF_PATH)

# --- 4. HIERARCHICAL CHUNKING GATEWAY ---
headers_to_split_on = [
    ("#", "section"),
    ("##", "subsection"),
    ("###", "clause")
]
md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
structural_chunks = md_splitter.split_text(md_text)

# Token threshold fallback splitter for overly long clauses
fallback_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=150
)
final_documents = fallback_splitter.split_documents(structural_chunks)

# --- 5. COMPUTE AND EMBED VECTOR PAYLOADS ---
print(f"Processing and indexing {len(final_documents)} structural chunks...")

points = []
for idx, doc in enumerate(tqdm(final_documents)):
    text_content = doc.page_content
    metadata = doc.metadata

    # Generate Dense Embeddings
    dense_vector = dense_model.encode(text_content).tolist()

    # Generate Sparse Token Mappings natively via text parsing for storage
    # We pass the document text to both slots; Qdrant handles internal fastembed transformations
    points.append(
        models.PointStruct(
            id=idx,
            vector={
                "dense": dense_vector,
            },
            payload={
                "text": text_content,
                "metadata": metadata
            }
        )
    )

# Upload chunks to the local vector store
client.upload_points(collection_name=COLLECTION_NAME, points=points)
# Compute sparse vectors internally for the uploaded collection texts
client.upload_collection(
    collection_name=COLLECTION_NAME,
    vectors=[
        {
            "dense": p.vector["dense"],
            "sparse": models.Document(text=p.payload["text"], model="Qdrant/bm25")
        }
        for p in points
    ],
    payload=[p.payload for p in points]
)

print("\nIngestion pipeline complete. Storage engine populated.")
