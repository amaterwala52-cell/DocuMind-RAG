# DocuMind — Local RAG Document Assistant

A privacy-focused Retrieval-Augmented Generation application for PDF question answering using a local LLM.

## Stack
Python, Streamlit, Ollama, Qwen 3, Sentence Transformers, FAISS, PyPDF.

## Pipeline
PDF → Text Extraction → Chunking → Local Embeddings → FAISS → Top-K Retrieval → Qwen 3 → Grounded Answer + Sources

## Setup

Install Ollama for Windows:
https://ollama.com/download/windows

Then:
```powershell
ollama pull qwen3:8b
ollama list
```

Create/activate the Python environment:
```powershell
py -m venv .venv
.venv\Scriptsctivate
python -m pip install -r requirements.txt
```

Run:
```powershell
streamlit run app.py
```

## Resume
**DocuMind — Local RAG Document Assistant**
- Built a privacy-focused RAG application for semantic question answering over PDF documents.
- Implemented PDF extraction, overlapping chunking, local sentence-transformer embeddings, FAISS vector similarity search, top-k retrieval, and locally hosted Qwen LLM generation.
- Added source/page references and a Streamlit interface for multi-document querying.
