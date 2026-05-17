import os
import streamlit as st
from dotenv import load_dotenv
from src.chain import build_chain

load_dotenv()

# Use Streamlit secrets if available (deployed), else fall back to .env (local)
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# Page config
st.set_page_config(
    page_title="RAG Research Assistant",
    page_icon="🔬",
    layout="wide"
)

# Load chain once and cache it
@st.cache_resource
def load_chain():
    return build_chain(k=8)

# Header
st.title("🔬 RAG Research Assistant")
st.markdown("Ask questions about **24 foundational AI/ML research papers** — answers grounded in retrieved context.")

# Sidebar — RAGAS scores
with st.sidebar:
    st.header("📊 System Performance")
    st.markdown("**RAGAS Evaluation Scores**")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Faithfulness", "0.83")
        st.metric("Answer Relevancy", "0.71")
    with col2:
        st.metric("Context Recall", "0.77")
        st.metric("Overall", "0.77")

    st.divider()
    st.markdown("**Knowledge Base**")
    st.markdown("- 24 AI/ML research papers")
    st.markdown("- 3,945 text chunks")
    st.markdown("- Embedding: all-MiniLM-L6-v2")
    st.markdown("- LLM: Llama 3.1 via Groq")
    st.markdown("- Vector DB: FAISS")

    st.divider()
    st.markdown("**Sample Questions**")
    sample_questions = [
        "What is self-attention?",
        "How does LoRA reduce parameters?",
        "What metrics does RAGAS use?",
        "What is the Vision Transformer?",
        "How does QLoRA work?",
        "What is Dense Passage Retrieval?"
    ]
    for q in sample_questions:
        if st.button(q, use_container_width=True):
            st.session_state.question = q

# Main area
chain = load_chain()

# Question input
question = st.text_input(
    "Ask a question:",
    value=st.session_state.get("question", ""),
    placeholder="e.g. What is self-attention?"
)

if st.button("🔍 Search", type="primary") or st.session_state.get("question"):
    if question.strip():
        with st.spinner("Retrieving from papers and generating answer..."):
            try:
                result = chain.invoke({"query": question})
                answer = result["result"]
                source_docs = result["source_documents"]

                # Clear session state question after use
                if "question" in st.session_state:
                    del st.session_state["question"]

                # Answer
                st.markdown("### 💡 Answer")
                st.markdown(answer)

                # Sources
                st.markdown("### 📄 Retrieved Sources")
                seen = set()
                for i, doc in enumerate(source_docs):
                    source = doc.metadata.get("source", "unknown")
                    source_name = os.path.basename(source)
                    if source_name not in seen:
                        seen.add(source_name)
                    with st.expander(f"Chunk {i+1} — {source_name}"):
                        st.markdown(doc.page_content)

            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a question.")