import streamlit as st
from src.chain import build_chain

st.set_page_config(page_title="RAG Research Assistant")

st.title("📚 RAG Research Assistant")

query = st.text_input("Ask a question from your PDFs")

if query:
    chain = build_chain()

    response = chain.invoke(query)

    st.write(response["result"])