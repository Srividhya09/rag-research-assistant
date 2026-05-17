import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from src.retriever import load_retriever

load_dotenv()

PROMPT_TEMPLATE = """
You are an expert AI research assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't have enough information in the provided papers to answer this."
Do NOT use any outside knowledge. Be specific and detailed in your answer.

Context:
{context}

Question:
{question}

Answer:
"""

def build_chain(k=8):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )
    retriever = load_retriever(k=k)
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    print("RAG chain built successfully!")
    return chain

if __name__ == "__main__":
    chain = build_chain()
    questions = [
        "What is self-attention?",
        "How does LoRA reduce the number of trainable parameters?",
        "What evaluation metrics does RAGAS use?"
    ]
    for question in questions:
        print(f"\n{'='*60}")
        print(f"Q: {question}")
        result = chain.invoke({"query": question})
        print(f"A: {result['result']}")
        print(f"\nSources:")
        for doc in result['source_documents']:
            print(f"  - {doc.metadata.get('source', 'unknown')}")