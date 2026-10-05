# Interview Cheatsheet

**Project:** DocuMind is a local RAG application that retrieves relevant PDF chunks using vector similarity and supplies them as context to a local Qwen LLM.

**Flow:** PDF → extraction → chunking → embeddings → FAISS → retrieval → context → Qwen → answer.

**Why RAG?** It gives an LLM access to external/private document knowledge at query time without retraining the model.

**Embeddings:** Numerical vectors representing semantic meaning.

**Chunking:** Splits documents into smaller overlapping pieces so retrieval can find specific relevant passages.

**FAISS:** Efficient vector similarity search library.

**Why local LLM?** Data can remain on the machine and there are no per-request cloud API charges.

**RAG vs fine-tuning:** RAG supplies knowledge at runtime; fine-tuning changes model behavior through additional training.

**Limitation:** RAG reduces hallucination risk by grounding answers, but does not guarantee zero hallucinations.
