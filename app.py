import os
import tempfile
import torch
import streamlit as st
from pypdf import PdfReader

# -----------------------------
# LANGCHAIN IMPORTS (UPDATED)
# -----------------------------
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains.retrieval_qa.base import RetrievalQA

# -----------------------------
# TRANSFORMERS
# -----------------------------
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# -----------------------------
# STREAMLIT CONFIG
# -----------------------------
st.set_page_config(
    page_title="Explainable AI Chatbot",
    layout="wide"
)

# -----------------------------
# GLOBAL UI FIX (WHITE THEME)
# -----------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: white; color: black; }
    header[data-testid="stHeader"] { background-color: white; }
    h1, h2, h3, h4, h5, h6 { color: black; }
    .stTextInput input {
        background-color: #ffffff;
        color: black;
        caret-color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# HEADER
# -----------------------------
st.title("🤖 Explainable AI Chatbot")
st.markdown("### Vector Similarity Search + LLMs (RAG)")
st.markdown("---")

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.header("📄 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

# -----------------------------
# CONSTANTS
# -----------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 4

# -----------------------------
# LOAD LLM (CLOUD SAFE)
# -----------------------------
@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        LLM_MODEL,
        torch_dtype=torch.float32
    )

    pipe = pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256
    )

    return HuggingFacePipeline(pipeline=pipe)

# -----------------------------
# BUILD VECTOR STORE
# -----------------------------
@st.cache_resource
def build_vectorstore(files):
    documents = []

    for file in files:
        if file.size == 0:
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        reader = PdfReader(tmp_path)

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": file.name,
                            "page": page_num + 1
                        }
                    )
                )

    if not documents:
        st.error("No readable text found in PDFs.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.from_documents(chunks, embeddings)

# -----------------------------
# BUILD QA CHAIN
# -----------------------------
@st.cache_resource
def build_qa_chain(files):
    vectorstore = build_vectorstore(files)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    llm = load_llm()

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff"
    )

# -----------------------------
# MAIN UI
# -----------------------------
question = st.text_input(
    "Ask a question based on your uploaded documents",
    placeholder="Type your question here..."
)

if uploaded_files:
    qa_chain = build_qa_chain(uploaded_files)

    if st.button("🔍 Ask Question"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating answer..."):
                response = qa_chain.invoke({"query": question})

            st.subheader("✅ Answer")
            st.write(response["result"])

            st.subheader("📚 Sources")
            for i, doc in enumerate(response["source_documents"], 1):
                st.markdown(
                    f"""
                    **Source {i}**  
                    📄 {doc.metadata['source']}  
                    📑 Page {doc.metadata['page']}

                    {doc.page_content[:300]}...
                    """
                )
else:
    st.info("👈 Upload PDF files from the sidebar to begin.")
