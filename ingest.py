"""
ingest.py
---------
Stage 1-2 of the RAG pipeline: Ingest + Embed & Store.

What this does, step by step:
  1. Reads every PDF in the `data/` folder.
  2. Splits each PDF's text into overlapping chunks (so we don't lose
     context at chunk boundaries).
  3. Converts each chunk into a vector embedding using a small local
     sentence-transformer model (no API key, runs on CPU).
  4. Stores the chunks + embeddings + metadata (source file, page number)
     in a local ChromaDB collection on disk.

Run this once whenever you add/change PDFs:
    python ingest.py
"""

import os
import re
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

DATA_DIR = "data"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "documents"

CHUNK_SIZE = 500       # target characters per chunk
CHUNK_OVERLAP = 80     # characters shared between consecutive chunks


def extract_pages(pdf_path):
    """Return a list of (page_number, text) tuples for a PDF."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            pages.append((i + 1, text))
    return pages


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split text into overlapping chunks on sentence boundaries where possible.
    Overlap matters: without it, a fact split across a chunk boundary
    (e.g. "The results showed | a 40% improvement") becomes unretrievable.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) <= chunk_size:
            current += (" " if current else "") + sentence
        else:
            if current:
                chunks.append(current.strip())
            # start next chunk with overlap from the end of the previous one
            overlap_text = current[-overlap:] if len(current) > overlap else current
            current = overlap_text + " " + sentence

    if current:
        chunks.append(current.strip())

    return [c for c in chunks if len(c) > 20]  # drop near-empty fragments


def build_index():
    if not os.path.isdir(DATA_DIR):
        raise SystemExit(f"'{DATA_DIR}/' folder not found. Put your PDFs there first.")

    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise SystemExit(f"No PDFs found in '{DATA_DIR}/'. Add some and re-run.")

    print(f"Found {len(pdf_files)} PDF(s): {pdf_files}")

    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    # Fresh index each run — simplest correct behavior for a class project.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    all_ids, all_docs, all_metas = [], [], []
    chunk_id = 0

    for pdf_file in pdf_files:
        path = os.path.join(DATA_DIR, pdf_file)
        print(f"  Processing {pdf_file}...")
        pages = extract_pages(path)

        for page_num, page_text in pages:
            for chunk in chunk_text(page_text):
                all_ids.append(f"chunk_{chunk_id}")
                all_docs.append(chunk)
                all_metas.append({"source": pdf_file, "page": page_num})
                chunk_id += 1

    if not all_docs:
        raise SystemExit("No extractable text found in the PDFs (are they scanned images?).")

    print(f"Embedding {len(all_docs)} chunks...")
    embeddings = model.encode(all_docs, show_progress_bar=True).tolist()

    # Chroma has a batch-size limit; add in batches to be safe.
    BATCH = 500
    for i in range(0, len(all_docs), BATCH):
        collection.add(
            ids=all_ids[i:i + BATCH],
            documents=all_docs[i:i + BATCH],
            embeddings=embeddings[i:i + BATCH],
            metadatas=all_metas[i:i + BATCH],
        )

    print(f"Done. Indexed {len(all_docs)} chunks from {len(pdf_files)} PDF(s) into '{CHROMA_DIR}/'.")


if __name__ == "__main__":
    build_index()
