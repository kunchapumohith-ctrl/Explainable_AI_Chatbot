import os
import tempfile
import torch
import streamlit as st
from pypdf import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# -------------------------------------------------
# STREAMLIT PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Explainable AI Chatbot",
    page_icon="🤖",
    layout="wide"
)

# -------------------------------------------------
# GLOBAL STYLES (UI FIXES)
# -------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffffff;
        color: #000000;
    }

    header[data-testid="stHeader"] {
        background-color: #ffffff;
        border-bottom: 1px solid #eee;
    }

    h1, h2, h3, h4 {
        color: #000000;
    }

    .stTextInput input {
        background-color: #FFF8E1 !important;
        color: black !important;
        caret-color: black !important;
        border-radius: 8px;
    }

    .stButton > button {
        background: linear-gradient(90deg, #FFA000, #FFD54F);
        color: black;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 24px;
        border: none;
    }

    .card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0px 4px 14px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }

    .source-box {
        background-color: #FFF3CD;
        padding: 14px;
        border-radius: 10px;
        margin-bottom: 12px;
        border-left: 6px solid #FF9800;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFE082, #FFD54F);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.title("🤖 Explainable AI Chatbot")
st.markdown("### Vector Similarity Search + LLM (RAG with Explainability)")

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.header("📄 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.markdown("---")
    st.markdown("### Features")
    st.markdown("- Semantic Search (FAISS)")
    st.markdown("- Explainable Answers")
    st.markdown("- Page-level Citations")
    st.markdown("- Local LLM (FLAN-T5)")

# -------------------------------------------------
# CONSTANTS
# -------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 4

# -------------------------------------------------
# LOAD LLM (CACHED)
# -------------------------------------------------
@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)

    pipe = pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=False
    )

    return HuggingFacePipeline(pipeline=pipe)

# -------------------------------------------------
# BUILD QA CHAIN
# -------------------------------------------------
@st.cache_resource
def build_qa_chain(files):
    documents = []

    for uploaded_file in files:
        if uploaded_file.size == 0:
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        reader = PdfReader(tmp_path)

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": uploaded_file.name,
                            "page": page_num + 1
                        }
                    )
                )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": TOP_K}
    )

    llm = load_llm()

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff"
    )

# -------------------------------------------------
# MAIN UI
# -------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

question = st.text_input(
    "Ask a question based on your uploaded documents",
    placeholder="Type your question here..."
)

st.markdown("</div>", unsafe_allow_html=True)

if uploaded_files:
    qa_chain = build_qa_chain(uploaded_files)

    if st.button("Ask Question"):
        if question.strip():
            with st.spinner("Generating explainable answer..."):
                response = qa_chain.invoke(question)

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("✅ Answer")
            st.write(response["result"])
            st.markdown("</div>", unsafe_allow_html=True)

            st.subheader("📌 Sources & Explanation")
            for i, doc in enumerate(response["source_documents"], 1):
                st.markdown(
                    f"""
                    <div class="source-box">
                        <b>Source {i}</b><br>
                        <b>Document:</b> {doc.metadata['source']}<br>
                        <b>Page:</b> {doc.metadata['page']}<br><br>
                        {doc.page_content[:300]}...
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.warning("Please enter a question.")
else:
    st.info("⬅️ Upload PDF files from the sidebar to begin.")
