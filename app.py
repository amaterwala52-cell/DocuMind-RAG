import pickle
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import ollama

APP_DIR = Path(__file__).parent
INDEX_DIR = APP_DIR / "vector_store"
INDEX_FILE = INDEX_DIR / "index.faiss"
META_FILE = INDEX_DIR / "metadata.pkl"

LLM_MODEL = "qwen3:1.7b"
EMBED_MODEL = "all-MiniLM-L6-v2"

st.set_page_config(page_title="DocuMind - Local RAG", page_icon="📚", layout="wide")
st.title("📚 DocuMind — Local RAG Document Assistant")
st.caption("Private document Q&A powered by a locally running LLM.")

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer(EMBED_MODEL)

def extract_pages(uploaded_file):
    reader = PdfReader(uploaded_file)
    pages = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"text": text.strip(), "source": uploaded_file.name, "page": page_no})
    return pages

def chunk_text(text, chunk_size=180, overlap=40):
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap
    return chunks

def build_chunks(files):
    all_chunks = []
    for uploaded_file in files:
        for page in extract_pages(uploaded_file):
            for chunk in chunk_text(page["text"]):
                all_chunks.append({"text": chunk, "source": page["source"], "page": page["page"]})
    return all_chunks

def embed_texts(texts):
    model = get_embedding_model()
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype="float32")

def save_index(chunks, vectors):
    INDEX_DIR.mkdir(exist_ok=True)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(INDEX_FILE))
    with open(META_FILE, "wb") as f:
        pickle.dump(chunks, f)

def load_index():
    if not INDEX_FILE.exists() or not META_FILE.exists():
        return None, None
    index = faiss.read_index(str(INDEX_FILE))
    with open(META_FILE, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def retrieve(question, top_k=5):
    index, chunks = load_index()
    if index is None:
        return []
    query_vector = embed_texts([question])
    scores, indices = index.search(query_vector, min(top_k, index.ntotal))
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            item = dict(chunks[idx])
            item["score"] = float(score)
            results.append(item)
    return results

def generate_answer(question, results):
    if not results:
        return "I couldn't find relevant information in the indexed documents."

    context = "\n\n".join(
        f"[Source {i} | {item['source']} | Page {item['page']}]\n{item['text']}"
        for i, item in enumerate(results, start=1)
    )

    prompt = f"""You are DocuMind, a document-grounded AI assistant.
Answer the user's question ONLY using the supplied context.
Do not invent information. If the answer is not present, say:
"I could not find this information in the uploaded documents."
Cite factual claims using [Source N].

CONTEXT:
{context}

USER QUESTION:
{question}"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "Answer using retrieved document context only."},
            {"role": "user", "content": prompt},
        ],
    )
    return response["message"]["content"]

with st.sidebar:
    st.header("📄 Knowledge Base")
    uploaded_files = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)

    if st.button("🔨 Build / Rebuild Knowledge Base", use_container_width=True):
        if not uploaded_files:
            st.warning("Upload at least one PDF first.")
        else:
            with st.spinner("Extracting, chunking and creating embeddings..."):
                try:
                    chunks = build_chunks(uploaded_files)
                    if not chunks:
                        st.error("No readable text found. Scanned PDFs need OCR.")
                    else:
                        vectors = embed_texts([c["text"] for c in chunks])
                        save_index(chunks, vectors)
                        st.success(f"Indexed {len(chunks)} chunks successfully.")
                except Exception as exc:
                    st.error(f"Indexing failed: {exc}")

    index, _ = load_index()
    if index is not None:
        st.success(f"Knowledge base ready: {index.ntotal} chunks")
    else:
        st.info("No knowledge base found yet.")

    st.divider()
    st.markdown("**Local AI Stack**")
    st.markdown("LLM: Qwen 3 via Ollama\n\nEmbeddings: Sentence Transformers\n\nVector DB: FAISS\n\nUI: Streamlit")

st.subheader("💬 Ask your documents")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask something about your uploaded documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating an answer..."):
            try:
                results = retrieve(question, top_k=5)
                answer = generate_answer(question, results)
                st.markdown(answer)

                if results:
                    with st.expander("🔎 Retrieved Sources"):
                        for i, item in enumerate(results, start=1):
                            st.markdown(f"**Source {i}: {item['source']} — Page {item['page']}**")
                            st.caption(f"Similarity: {item['score']:.3f}")
                            st.write(item["text"][:700] + ("..." if len(item["text"]) > 700 else ""))
            except Exception as exc:
                answer = f"Something went wrong: {exc}"
                st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
