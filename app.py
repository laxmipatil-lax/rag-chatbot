"""
app.py
------
Chat UI for the RAG system, built with Streamlit.

Run with:
    streamlit run app.py

Prerequisite: you must have already run `python ingest.py` at least once
so that chroma_db/ exists.
"""

import streamlit as st
from core import answer_question

st.set_page_config(page_title="Document RAG Chat", page_icon="📄")
st.title("📄 Document Q&A Chatbot")
st.caption("Ask questions about the PDFs you indexed with ingest.py")

if "history" not in st.session_state:
    st.session_state.history = []  # list of {"role": "user"/"assistant", "content": str}

# Render past turns
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

# New user input
user_input = st.chat_input("Ask a question about your documents...")

if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant chunks and generating an answer..."):
            answer, chunks = answer_question(user_input, st.session_state.history[:-1])
            st.markdown(answer)

            if chunks:
                with st.expander("Sources used"):
                    for c in chunks:
                        st.markdown(f"**{c['source']}**, page {c['page']}")
                        st.caption(c["text"][:300] + ("..." if len(c["text"]) > 300 else ""))

    st.session_state.history.append({"role": "assistant", "content": answer})

with st.sidebar:
    st.header("About")
    st.write(
        "This chatbot answers questions using only the PDFs you indexed. "
        "It retrieves the most relevant chunks from a local vector store "
        "(ChromaDB) and passes them to a local LLM (via Ollama) to generate "
        "an answer with source citations."
    )
    if st.button("Clear conversation"):
        st.session_state.history = []
        st.rerun()
