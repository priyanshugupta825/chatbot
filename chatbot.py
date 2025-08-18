import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import google.generativeai as genai

# 🔑 Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# 🎨 Streamlit UI
st.set_page_config(page_title="PDF Q&A Chatbot", page_icon="📄", layout="wide")
st.title("📄 PDF Q&A Chatbot - Gemini 1.5 Flash")

# Sidebar for file upload
with st.sidebar:
    st.header("Upload your document")
    file = st.file_uploader("Upload a PDF file", type="pdf")

if file is not None:
    # Extract text from PDF
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        if page.extract_text():
            text += page.extract_text()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150, length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Embeddings (Gemini → fallback HuggingFace)
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        _ = embeddings.embed_query("test")  # test call
    except Exception:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Store embeddings in FAISS
    vector_store = FAISS.from_texts(chunks, embeddings)

    # User question
    st.subheader("Ask a question about your PDF:")
    user_question = st.text_input("Type your question here...")

    if user_question:
        # Search relevant chunks
        docs = vector_store.similarity_search(user_question, k=3)

        # LLM for answering
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        chain = load_qa_chain(llm, chain_type="stuff")

        # Response
        response = chain.run(input_documents=docs, question=user_question)

        # Show answer
        st.markdown("### 🧠 Answer:")
        st.write(response)

else:
    st.info("👆 Please upload a PDF to start chatting.")
