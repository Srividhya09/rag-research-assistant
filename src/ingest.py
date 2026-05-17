import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# Paths
DATA_DIR = Path("data")
VECTORSTORE_DIR = Path("vectorstore")

def load_pdfs():
    all_docs = []
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs...")
    for pdf_path in pdf_files:
        print(f"Loading: {pdf_path.name}")
        loader = PyMuPDFLoader(str(pdf_path))
        docs = loader.load()
        all_docs.extend(docs)
    print(f"Total pages loaded: {len(all_docs)}")
    return all_docs

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    print(f"Total chunks created: {len(chunks)}")
    return chunks

def create_vectorstore(chunks):
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    print("Creating FAISS vectorstore... (this may take a few minutes)")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    VECTORSTORE_DIR.mkdir(exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))
    print(f"Vectorstore saved to: {VECTORSTORE_DIR}")
    return vectorstore

if __name__ == "__main__":
    docs = load_pdfs()
    chunks = chunk_documents(docs)
    vectorstore = create_vectorstore(chunks)
    print("Ingestion complete!")