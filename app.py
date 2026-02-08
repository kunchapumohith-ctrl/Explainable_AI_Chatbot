import streamlit as st
import tempfile
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# -----------------------------
# STREAMLIT CONFIG
# -----------------------------
st.set_page_config(
    page_title="Explainable AI Chatbot",
    layout="wide"
)

# -----------------------------
# BASIC CLEAN UI (CLOUD SAFE)
# -----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffffff;
        color: #000000;
    }
    h1, h2, h3 {
        color: #000000;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# HEADER
# -----------------------------
st.title("📄 Explainable AI Chatbot")
st.markdown("### PDF Question Answering using Vector Search (RAG)")

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.header("Upload PDFs")
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
# VECTORSTORE BUILDER
# -----------------------------
@st.cache_resource(show_spinner="Building document index...")
def build_vectorstore(files):
    documents = []

    for uploaded_file in files:
        if uploaded_file.size == 0:
            continue  # avoid EmptyFileError

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

    if not documents:
        raise ValueError("No readable text found in uploaded PDFs.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# -----------------------------
# LLM LOADER (CPU SAFE)
# -----------------------------
@st.cache_resource(show_spinner="Loading language model...")
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
        max_new_tokens=256,
        do_sample=False,
    )

    return HuggingFacePipeline(pipeline=pipe)

# -----------------------------
# QA CHAIN BUILDER
# -----------------------------
@st.cache_resource(show_spinner="Preparing QA system...")
def build_qa_chain(files):
    vectorstore = build_vectorstore(files)
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": TOP_K}
    )

    llm = load_llm()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff"
    )

    return qa_chain

# -----------------------------
# MAIN UI
# -----------------------------
question = st.text_input(
    "Ask a question based on your uploaded documents",
    placeholder="e.g. What is the document about?"
)

if uploaded_files:
    try:
        qa_chain = build_qa_chain(uploaded_files)

        if st.button("Ask Question"):
            if question.strip():
                with st.spinner("Generating answer..."):
                    response = qa_chain.invoke(question)

                st.subheader("✅ Answer")
                st.write(response["result"])

                st.subheader("📌 Sources")
                for i, doc in enumerate(response["source_documents"], 1):
                    st.markdown(
                        f"""
                        **Source {i}**  
                        File: `{doc.metadata['source']}`  
                        Page: {doc.metadata['page']}  

                        {doc.page_content[:300]}...
                        """
                    )
            else:
                st.warning("Please enter a question.")
    except Exception as e:
        st.error(str(e))
else:
    st.info("Upload PDF files from the sidebar to get started.")
