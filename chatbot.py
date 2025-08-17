import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
import google.generativeai as genai

# ✅ Gemini API key setup
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Streamlit App
st.header("📄 PDF Q&A Chatbot ")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file", type="pdf")

if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""   # safer extraction

    # ✅ Split into chunks (optimized)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )
    chunks = text_splitter.split_text(text)

    # ✅ Use Gemini embeddings (faster than HuggingFace)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embeddings)

    # User question
    user_question = st.text_input("Ask a question about your PDF:")

    if user_question:
        match = vector_store.similarity_search(user_question, k=3)

        # ✅ Build context manually
        context = "\n\n".join([doc.page_content for doc in match])
        prompt = f"""
        You are a helpful assistant. Answer the question using the context below.
        If the context is not relevant, say "Sorry, I couldn’t find that in the PDF."

        Context:
        {context}

        Question: {user_question}
        """

        # ✅ Gemini LLM
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

        # ✅ Streaming response (like ChatGPT)
        placeholder = st.empty()
        full_response = ""

        for chunk in llm.stream(prompt):
            if chunk.content:
                full_response += chunk.content
                placeholder.markdown(full_response)

        st.write("### ✅ Final Answer:")
        st.write(full_response)
