import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

VECTORSTORE_DIR = Path("vectorstore")

def load_retriever(k=4):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )
    print(f"Retriever loaded — returning top {k} chunks per query")
    return retriever

if __name__ == "__main__":
    retriever = load_retriever()
    # Quick test
    test_query = "What is self-attention?"
    results = retriever.invoke(test_query)
    print(f"\nQuery: {test_query}")
    print(f"Retrieved {len(results)} chunks:")
    for i, doc in enumerate(results):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Source: {doc.metadata.get('source', 'unknown')}")
        print(f"Content: {doc.page_content[:200]}...")