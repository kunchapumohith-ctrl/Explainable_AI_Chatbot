import os
import tempfile
import torch
import streamlit as st
from pypdf import PdfReader

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Explainable AI Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ---------------- UI FIX ----------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: white;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🤖 Explainable AI Chatbot")
st.subheader("Document-based Question Answering (RAG)")

# ---------------- IMPORTS (CORRECT) ----------------
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline
)

# ---------------- CONFIG ----------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("📄 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

# ---------------- VECTORSTORE ----------------
@st.cache_resource
def build_vectorstore(files):
    if not files:
        return None

    documents = []

    for file in files:
        if file.size == 0:
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp.flush()

            try:
                reader = PdfReader(tmp.name)
            except Exception:
                continue

            for page in reader.pages:
                text = page.extract_text()
                if text and text.strip():
                    documents.append(Document(page_content=text))

    if not documents:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore

# ---------------- LLM ----------------
@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)

    pipe = pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256
    )

    return HuggingFacePipeline(pipeline=pipe)

# ---------------- MAIN ----------------
question = st.text_input("Ask a question from the uploaded documents")

if uploaded_files:
    vectorstore = build_vectorstore(uploaded_files)

    if vectorstore is None:
        st.warning("No readable text found in uploaded PDFs.")
        st.stop()

    llm = load_llm()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    if st.button("Ask"):
        if question.strip():
            with st.spinner("Generating answer..."):
                response = qa_chain.invoke(question)

            st.subheader("✅ Answer")
            st.write(response["result"])

            st.subheader("📌 Sources")
            for i, doc in enumerate(response["source_documents"], 1):
                st.markdown(f"**Source {i}:**")
                st.write(doc.page_content[:300] + "...")
        else:
            st.warning("Please enter a question.")
else:
    st.info("Upload PDFs from the sidebar to begin.")
