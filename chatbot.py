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

# 🎨 Streamlit UI Config
st.set_page_config(page_title="📄 PDF Q&A Chatbot", page_icon="🤖", layout="wide")

st.markdown(
    """
    <h2 style='text-align: center; color: #4CAF50;'>
        📄 PDF Q&A Chatbot 
    </h2>
    <p style='text-align: center; color: gray;'>Upload a PDF and ask questions instantly!</p>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.header("📂 Upload Document")
    file = st.file_uploader("Choose a PDF file", type="pdf")
    st.markdown("---")


# ✅ Use Session State to avoid rebuilding embeddings every time
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if file is not None:
    if st.session_state.vector_store is None:
        with st.spinner("📑 Processing PDF... Please wait."):
            # Extract text from PDF
            pdf_reader = PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                if page.extract_text():
                    text += page.extract_text()

            # Split text
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
            st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)

        st.success("✅ PDF Processed Successfully!")

    # Question input
    st.subheader("💬 Ask a Question")
    user_question = st.text_input("Type your question here...")

    if user_question:
        with st.spinner("🤔 Thinking..."):
            docs = st.session_state.vector_store.similarity_search(user_question, k=3)

            # LLM
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
            chain = load_qa_chain(llm, chain_type="stuff")

            response = chain.run(input_documents=docs, question=user_question)

        # Show answer
        st.markdown("### 🧠 Answer:")
        st.success(response)

else:
    st.info("👆 Upload a PDF file to get started.")
