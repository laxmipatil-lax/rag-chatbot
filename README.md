# Document RAG Chatbot

A fully local, no-API-key Retrieval-Augmented Generation (RAG) chatbot that
answers questions about your PDF documents (research papers, company docs,
etc.), with chat history and source citations.

## How it works (5 stages)

1. **Ingest** (`ingest.py`) — reads PDFs from `data/`, extracts text per page.
2. **Chunk** (`ingest.py`) — splits text into overlapping ~500-char chunks
   on sentence boundaries, so context isn't lost at chunk edges.
3. **Embed & Store** (`ingest.py`) — converts chunks to vectors using a
   local `sentence-transformers` model, stores them in ChromaDB on disk.
4. **Retrieve** (`core.py`) — at query time, embeds the question and finds
   the top-k most similar chunks by cosine similarity.
5. **Augment & Generate** (`core.py`) — stuffs retrieved chunks + chat
   history into a prompt, sends it to a local LLM via Ollama, returns the
   answer with source/page citations.

No data leaves your machine. No API keys required.

## Setup

### 1. Install Ollama (the local LLM server)
Download from https://ollama.com and install it. Then pull a model:
```bash
ollama pull llama3.1:8b
```
(If your machine is low on RAM, use a smaller model instead, e.g.
`ollama pull llama3.2:3b`, and update `OLLAMA_MODEL` in `core.py` to match.)

Start the server (leave this running in a separate terminal):
```bash
ollama serve
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your PDFs
Put your research papers / company documents into the `data/` folder.

### 4. Build the index
```bash
python ingest.py
```
Re-run this any time you add or change PDFs.

### 5. Launch the chatbot
```bash
streamlit run app.py
```
This opens a browser tab with the chat interface.

## Project structure
```
rag_project/
├── data/            # put your PDFs here
├── chroma_db/        # auto-generated vector store (after running ingest.py)
├── ingest.py         # Stage 1-3: load, chunk, embed, store
├── core.py           # Stage 4-5: retrieve, build prompt, call LLM
├── app.py            # Streamlit chat UI
├── requirements.txt
└── README.md
```

## Things worth knowing for your viva / demo

- **Why chunk with overlap?** If a fact spans a chunk boundary, no single
  chunk contains it fully, so retrieval can miss it. Overlap reduces this.
- **Why cosine similarity?** Embeddings encode meaning as direction in
  high-dimensional space; cosine similarity measures how aligned two
  vectors are, which correlates with semantic similarity better than
  raw distance for these models.
- **Why does the prompt say "only use the context"?** Without that
  instruction, LLMs often answer from their own training data instead of
  your documents — this is the #1 way RAG demos silently produce wrong
  answers that "look" grounded.
- **Known limitation:** this ingestion pipeline extracts text only —
  scanned/image-based PDFs (no embedded text layer) will produce empty
  chunks. Mention this as a limitation if asked; a fix would be adding
  OCR (e.g. `pytesseract`) as a fallback.
- **Known limitation:** chat history is kept in memory only (Streamlit
  session state) — closing the tab loses it. Fine for a demo, not
  production-grade.
