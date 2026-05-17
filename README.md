# 🔬 RAG Research Assistant

An end-to-end **Retrieval-Augmented Generation (RAG)** pipeline built over 24 foundational AI/ML research papers. Ask any question — the system retrieves semantically relevant chunks from the papers and generates grounded, faithful answers using a large language model.

**🚀 [Live Demo](https://rag-research-assistant-9rcnopkpsao8cmg87hhq3r.streamlit.app/)**

---

## 📊 RAGAS Evaluation Results

| Metric | Score | Description |
|---|---|---|
| **Faithfulness** | 0.83 | Answers grounded in retrieved context |
| **Answer Relevancy** | 0.71 | Answers address the question asked |
| **Context Recall** | 0.77 | Retrieval fetches the right chunks |
| **Overall Average** | 0.77 | Across 25 standardized test queries |

> Evaluated using [RAGAS](https://github.com/explodinggradients/ragas) — the industry-standard RAG evaluation framework used in production ML systems.

---

## 🏗️ System Architecture

```
📄 24 Research Papers (PDFs)
         │
         ▼
   Text Extraction (PyMuPDF)
         │
         ▼
   Chunking (500 tokens, 50 overlap)
         │
         ▼
   Embeddings (all-MiniLM-L6-v2)
         │
         ▼
   FAISS Vector Store (3,945 chunks)
         │
    ┌────┴────┐
    │         │
User Query   Embed Query
    │         │
    └────┬────┘
         │
    Semantic Search (top-8 chunks)
         │
         ▼
   Groq LLM (Llama 3.1 8B)
         │
         ▼
   Grounded Answer + Sources
         │
         ▼
   RAGAS Evaluation Layer
   (Faithfulness · Relevancy · Recall)
```

---

## 📚 Knowledge Base

24 foundational AI/ML research papers including:

| Category | Papers |
|---|---|
| **Transformers & Attention** | Attention Is All You Need, BERT, RoBERTa, Multilingual BERT |
| **Large Language Models** | GPT-2, GPT-3, LLaMA, Mistral 7B |
| **RAG & Retrieval** | RAG (Lewis et al.), DPR, REALM |
| **Embeddings** | Word2Vec, Sentence-BERT, FAISS |
| **Fine-tuning** | LoRA, QLoRA, Prompt Tuning |
| **Evaluation** | RAGAS, TruthfulQA, ROUGE |
| **Computer Vision** | Vision Transformer (ViT), CLIP |
| **Generative Models** | GAN, DCGAN |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Embeddings** | Hugging Face `all-MiniLM-L6-v2` |
| **Vector Store** | FAISS (Facebook AI Similarity Search) |
| **LLM** | Llama 3.1 8B via Groq API |
| **Orchestration** | LangChain |
| **Evaluation** | RAGAS |
| **PDF Processing** | PyMuPDF |
| **Frontend** | Streamlit |
| **Deployment** | Streamlit Community Cloud |

---

## 📁 Project Structure

```
rag-research-assistant/
│
├── app.py                    # Streamlit frontend
├── requirements.txt
├── runtime.txt               # Python 3.11
│
├── src/
│   ├── __init__.py
│   ├── ingest.py             # PDF loading, chunking, embedding
│   ├── retriever.py          # FAISS vector store loader
│   ├── chain.py              # RAG chain with custom prompt
│   └── utils.py
│
├── evaluation/
│   ├── __init__.py
│   ├── eval_dataset.json     # 25 Q&A pairs with ground truth
│   ├── evaluate.py           # RAGAS evaluation pipeline
│   ├── results.csv           # v1 results (k=4)
│   └── results_v2.csv        # v2 results (k=8, tuned)
│
├── data/                     # 24 research PDFs (not tracked)
└── vectorstore/              # FAISS index files
    ├── index.faiss
    └── index.pkl
```

---

## ⚙️ How It Works

### 1. Ingestion Pipeline (`src/ingest.py`)
- Loads 24 PDFs using PyMuPDF
- Splits into chunks of 500 tokens with 50-token overlap
- Generates embeddings using `all-MiniLM-L6-v2`
- Stores in FAISS vector index (3,945 total chunks)

### 2. Retrieval (`src/retriever.py`)
- Loads pre-built FAISS index
- Performs semantic similarity search
- Returns top-8 most relevant chunks per query

### 3. Generation (`src/chain.py`)
- Custom prompt template enforcing context-only answers
- Llama 3.1 8B generates answer from retrieved chunks
- Returns answer + source documents

### 4. Evaluation (`evaluation/evaluate.py`)
- 25 hand-crafted Q&A pairs as ground truth
- RAGAS evaluates faithfulness, answer relevancy, context recall
- Batched evaluation to handle API rate limits

---

## 🚀 Running Locally

### Prerequisites
- Python 3.11
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Setup

```bash
# Clone the repo
git clone https://github.com/Srividhya09/rag-research-assistant.git
cd rag-research-assistant

# Create virtual environment
conda create -n RAG python=3.11
conda activate RAG

# Install dependencies
pip install -r requirements.txt
```

### Configure environment

Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```

### Add research papers

Download the 24 papers and place them in the `/data` folder. Then run ingestion:
```bash
python src/ingest.py
```

### Run the app

```bash
streamlit run app.py
```

### Run RAGAS evaluation

```bash
python -m evaluation.evaluate
```

---

## 📈 Evaluation Methodology

RAGAS evaluates three dimensions of RAG quality:

- **Faithfulness** — Is the answer factually consistent with the retrieved context? Detects hallucinations.
- **Answer Relevancy** — Does the answer actually address the question asked?
- **Context Recall** — Did the retrieval system fetch the chunks needed to answer the question?

Evaluation was run in batches of 5 questions with 30-second delays to handle free-tier API rate limits.

**Before vs After Tuning:**

| Metric | k=4 (baseline) | k=8 (tuned) | Change |
|---|---|---|---|
| Faithfulness | 0.8056 | 0.8319 | +0.026 |
| Answer Relevancy | 0.5993 | 0.7064 | +0.107 |
| Context Recall | 0.8444 | 0.7734 | -0.071 |
| **Overall** | **0.7498** | **0.7706** | **+0.021** |

---

## 💡 Sample Questions to Try

- What is self-attention?
- How does LoRA reduce the number of trainable parameters?
- What evaluation metrics does RAGAS use?
- What is the Vision Transformer?
- How does QLoRA differ from LoRA?
- What is Dense Passage Retrieval?
- What does ROUGE measure?
- What is multilingual BERT?

---

## 🔮 Future Improvements

- [ ] Add more papers to expand knowledge base
- [ ] Implement hybrid search (dense + sparse BM25)
- [ ] Add conversation memory for multi-turn Q&A
- [ ] Integrate MLflow for experiment tracking
- [ ] Add re-ranking layer for better retrieval precision
- [ ] Support user-uploaded PDFs

---

## 👩‍💻 Author

**Chakilela Srividhya**
- LinkedIn: [linkedin.com/in/srividhya-chakilela](https://linkedin.com/in/srividhya-chakilela)
- GitHub: [github.com/Srividhya09](https://github.com/Srividhya09)
- Live Demo: [rag-research-assistant-9rcnopkpsao8cmg87hhq3r.streamlit.app](https://rag-research-assistant-9rcnopkpsao8cmg87hhq3r.streamlit.app)

---

## 📄 License

MIT License — feel free to use, modify, and build on this project.
