import streamlit as st
import tempfile
from typing import List
from pypdf import PdfReader

# ---------------- LANGCHAIN IMPORTS (STABLE) ----------------
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Explainable AI Chatbot",
    layout="wide"
)

# ---------------- GLOBAL UI + VISIBILITY FIX ----------------
st.markdown(
    """
    <style>

    /* MAIN BACKGROUND */
    .stApp {
        background-color: white;
        color: black;
    }

    /* STREAMLIT TOP BAR (DEPLOY / RUNNING FIX) */
    header[data-testid="stHeader"] {
        background-color: white !important;
        border-bottom: 1px solid #eaeaea;
    }

    header[data-testid="stHeader"] svg,
    header[data-testid="stHeader"] span {
        color: black !important;
        fill: black !important;
    }

    /* ALL HEADINGS */
    h1, h2, h3, h4, h5, h6 {
        color: black !important;
    }

    /* TEXT INPUT */
    .stTextInput input {
        background-color: #FFF3E0 !important;
        color: black !important;
        caret-color: black !important;
        border-radius: 8px;
    }

    .stTextInput input::placeholder {
        color: #555 !important;
    }

    /* BUTTONS */
    .stButton > button {
        background: linear-gradient(90deg, #FFA000, #FFD54F);
        color: black !important;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 24px;
        border: none;
    }

    /* CARDS */
    .card {
        background-color: white;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0px 6px 18px rgba(0,0,0,0.12);
        margin-bottom: 20px;
        color: black;
    }

    /* SOURCES */
    .source-box {
        background-color: #FFECB3;
        padding: 14px;
        border-radius: 10px;
        margin-bottom: 12px;
        border-left: 6px solid #FB8C00;
        color: black;
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFE082, #FFD54F);
        color: black;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- HEADER ----------------
st.image(
    "https://cdn-icons-png.flaticon.com/512/4712/4712109.png",
    width=110
)
st.title("Explainable AI Chatbot")
st.markdown("### Vector Similarity Search with LLMs (RAG + XAI)")

# ---------------- CONFIG ----------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"

# ---------------- PDF LOADING ----------------
def load_documents(files) -> List[Document]:
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

    return documents

# ---------------- VECTOR STORE ----------------
@st.cache_resource(show_spinner="🔍 Creating vector store...")
def build_vectorstore(files):
    docs = load_documents(files)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    return FAISS.from_documents(chunks, embeddings)

# ---------------- LLM ----------------
@st.cache_resource(show_spinner="🤖 Loading language model...")
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=False
    )

    return HuggingFacePipeline(pipeline=pipe)

# ---------------- QA CHAIN ----------------
@st.cache_resource(show_spinner="🔗 Building QA chain...")
def build_qa_chain(files):
    vectorstore = build_vectorstore(files)
    llm = load_llm()

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True
    )

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("Upload Documents")
    st.markdown("Upload **one or more PDF files**")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.markdown("---")
    st.markdown("### Capabilities")
    st.markdown("- Semantic Search")
    st.markdown("- Explainable Answers")
    st.markdown("- Page-level Citations")
    st.markdown("- Local LLM (FLAN-T5)")

# ---------------- MAIN UI ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

question = st.text_input(
    "Ask a question based on your uploaded documents",
    placeholder="Type your question here and press Enter to apply"
)

st.markdown("</div>", unsafe_allow_html=True)

if uploaded_files:
    qa_chain = build_qa_chain(uploaded_files)

    if st.button("Ask Question"):
        if question.strip():
            with st.spinner("Generating explainable answer..."):
                result = qa_chain.invoke(question)

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Answer")
            st.write(result["result"])
            st.markdown("</div>", unsafe_allow_html=True)

            st.subheader("Sources & Explanation")
            for i, doc in enumerate(result["source_documents"], 1):
                st.markdown(
                    f"""
                    <div class="source-box">
                        <b>Source {i}</b><br>
                        <b>Document:</b> {doc.metadata.get('source')}<br>
                        <b>Page:</b> {doc.metadata.get('page')}<br><br>
                        {doc.page_content[:300]}...
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.warning("Please enter a question.")
else:
    st.info("Upload PDF files from the sidebar to begin.")
