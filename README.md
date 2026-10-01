# 🏛️ Conversational Hybrid RAG System

An end-to-end, production-grade Hybrid Retrieval-Augmented Generation (RAG) assistant specializing in Medieval Indian History (Gurjara-Pratihara Empire). 

The system combines dense semantic search, sparse lexical retrieval, cross-encoder neural reranking, conversational multi-turn context resolution, and LLM answer generation.

---

## ⚡ Architecture Pipeline

```text
User Query + Chat History
       │
       ▼
[ Query Reformulation (Gemini Flash) ] ── (Resolves pronouns & coreferences)
       │
       ├──► [ Dense Retriever: ChromaDB + text-embedding-004 ] (k=20)
       │
       └──► [ Sparse Retriever: BM25 Keyword Search ] (k=20)
       │
       ▼
[ Deduplication & Candidate Merging ] (Unique passages)
       │
       ▼
[ Cross-Encoder Reranker: ms-marco-MiniLM-L-6-v2 ] (Deep query-document scoring)
       │
       ▼ (Top-5 Passages)
[ Context Prompt Formulation ]
       │
       ▼
[ LLM Generation: Gemini 1.5 Flash ]
Key Features
Hybrid Retrieval: Mitigates vector embedding blind spots (e.g., exact proper nouns, dynastic titles) by combining dense semantic embeddings with BM25 keyword matching.

Cross-Encoder Reranking: Applies full cross-attention scoring across top-40 candidate passages using ms-marco-MiniLM-L-6-v2 to prioritize high-precision context chunks.

Conversational Memory & Coreference Resolution: Resolves ambiguous follow-up pronouns (e.g., "his father", "their capital") into self-contained search queries prior to vector execution.

Hallucination Guardrails: Enforces grounded prompt templates with explicit document-only instructions.

Interactive Inspection UI: Visualizes reranker scores and full retrieved context blocks in real time using Streamlit.

 Tech Stack
Orchestration: LangChain / LCEL

LLM & Embeddings: Google Gemini 1.5 Flash, models/text-embedding-004

Dense Vector Store: ChromaDB

Sparse Retrieval: Rank-BM25

Reranker: Sentence-Transformers (cross-encoder/ms-marco-MiniLM-L-6-v2)

Frontend: Streamlit

 Getting Started
1. Clone the Repository
Bash
git clone [https://github.com/SambhavYatendraBhardwaj/conversational-hybrid-rag.git](https://github.com/SambhavYatendraBhardwaj/conversational-hybrid-rag.git)
cd conversational-hybrid-rag
2. Set Up Virtual Environment
Bash
python -m venv myenv
# Windows
myenv\Scripts\activate
# Linux/macOS
source myenv/bin/activate
3. Install Dependencies
Bash
pip install -r requirements.txt
4. Configure Environment Variables
Create a .env file in the root directory:

Code snippet
GOOGLE_API_KEY="your-gemini-api-key"
5. Launch the Application
Bash
streamlit run frontend.py

---

### Isse Push Karne Ke Commands:

Terminal mein ye commands run kar do:

```powershell
git add README.md
git commit -m "docs: add comprehensive project README with architecture diagram"
git push origin main
       │
       ▼
Streamlit Interactive UI (Live relevance scores & expanders)
