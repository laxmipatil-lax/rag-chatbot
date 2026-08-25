"""
core.py
-------
Stage 3-5 of the RAG pipeline: Retrieve + Augment + Generate.

This module is imported by app.py (Streamlit UI). Keeping it separate
from the UI means you can also test it from a plain Python shell:

    >>> from core import answer_question
    >>> answer_question("What does the paper conclude?", [])
"""

import json
import urllib.request
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "documents"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"   # change to whatever you `ollama pull`ed
TOP_K = 4                      # how many chunks to retrieve per question

_embed_model = None
_collection = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        try:
            _collection = client.get_collection(COLLECTION_NAME)
        except Exception:
            raise RuntimeError(
                "No index found. Run `python ingest.py` first to build the vector store."
            )
    return _collection


def retrieve(query, k=TOP_K):
    """
    Stage 3: Retrieve.
    Embed the query, then find the k most similar chunks by cosine
    distance in Chroma. Returns a list of dicts: {text, source, page}.
    """
    model = _get_embed_model()
    collection = _get_collection()

    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=k)

    chunks = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    for doc, meta in zip(docs, metas):
        chunks.append({"text": doc, "source": meta.get("source"), "page": meta.get("page")})
    return chunks


def build_prompt(query, history, chunks):
    """
    Stage 4: Augment.
    Combine retrieved chunks + prior chat turns + the new question into
    a single prompt. The instruction to only use provided context is
    what keeps the model from hallucinating outside the documents.
    """
    context_block = "\n\n".join(
        f"[Source: {c['source']}, page {c['page']}]\n{c['text']}" for c in chunks
    )

    history_block = ""
    for turn in history[-6:]:  # keep last 6 turns so the prompt doesn't grow unbounded
        role = "User" if turn["role"] == "user" else "Assistant"
        history_block += f"{role}: {turn['content']}\n"

    prompt = f"""You are a research assistant. Answer the question using ONLY the
context below. If the answer is not in the context, say you don't know —
do not make anything up. Cite the source file and page number for any
claim you make, in the form (source, page).

Context:
{context_block}

Conversation so far:
{history_block}

Question: {query}

Answer:"""
    return prompt


def call_ollama(prompt, model=OLLAMA_MODEL):
    """
    Stage 5: Generate.
    Sends the prompt to a locally running Ollama server and returns the
    full text response (non-streaming, simplest for a class project).
    Requires: `ollama serve` running and the model pulled beforehand
    (`ollama pull llama3.1:8b`).
    """
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            return data.get("response", "").strip()
    except Exception as e:
        return (
            f"[Error calling Ollama: {e}]\n"
            "Make sure `ollama serve` is running and you've pulled the model "
            f"with `ollama pull {model}`."
        )


def answer_question(query, history):
    """
    Runs the full retrieve -> augment -> generate pipeline for one turn.
    Returns (answer_text, retrieved_chunks) so the UI can show sources.
    """
    chunks = retrieve(query)
    if not chunks:
        return "I couldn't find anything relevant in the indexed documents.", []
    prompt = build_prompt(query, history, chunks)
    answer = call_ollama(prompt)
    return answer, chunks
