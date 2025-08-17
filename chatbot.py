import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
import google.generativeai as genai

# ✅ Gemini API key setup
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Streamlit App
st.header("PDF Q&A Chatbot - Gemini 1.5 Flash ")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file", type="pdf")

if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150, length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Create embeddings + FAISS vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)

    # User question
    user_question = st.text_input("Ask a question about your PDF:")

    if user_question:
        match = vector_store.similarity_search(user_question)

        # ✅ Use Gemini 1.5 Flash (Free Tier)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=match, question=user_question)

        st.write("### Answer:")
        st.write(response)
