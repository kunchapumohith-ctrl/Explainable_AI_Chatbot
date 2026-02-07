import os
import tempfile
import streamlit as st
from pypdf import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain_core.documents import Document

# ----------------------------
# PAGE CONFIG (UI FIX)
# ----------------------------
st.set_page_config(
    page_title="Explainable AI Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.markdown(
    """
    <style>
        .stApp {
            background-color: white;
            color: black;
        }
        h1, h2, h3, h4, h5, h6 {
            color: black;
        }
        .stTextInput>div>div>input {
            color: black;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# TITLE
# ----------------------------
st.title("🤖 Explainable AI Chatbot")
st.subheader("Ask questions based on your uploaded documents")

# ----------------------------
# LOAD LLM (CLOUD SAFE)
# ----------------------------
@st.cache_resource
def load_llm():
    return HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={
            "temperature": 0.2,
            "max_length": 512
        }
    )

# ----------------------------
# BUILD QA CHAIN
# ----------------------------
@st.cache_resource
def build_qa_chain(uploaded_files):
    documents = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        reader = PdfReader(tmp_path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                documents.append(Document(page_content=text))

        os.remove(tmp_path)

    if not documents:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    llm = load_llm()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    return qa_chain

# ----------------------------
# FILE UPLOAD
# ----------------------------
uploaded_files = st.file_uploader(
    "📄 Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

qa_chain = None
if uploaded_files:
    with st.spinner("Processing documents..."):
        qa_chain = build_qa_chain(uploaded_files)
    st.success("Documents processed successfully!")

# ----------------------------
# QUESTION INPUT
# ----------------------------
st.markdown("### 💬 Ask a question")

question = st.text_input(
    "Type your question here and press Enter",
    placeholder="What is this document about?"
)

# ----------------------------
# ANSWER
# ----------------------------
if question and qa_chain:
    with st.spinner("Generating answer..."):
        response = qa_chain.invoke({"query": question})

    st.markdown("### ✅ Answer")
    st.write(response["result"])

    st.markdown("### 📚 Source Documents")
    for i, doc in enumerate(response["source_documents"], start=1):
        with st.expander(f"Source {i}"):
            st.write(doc.page_content)

elif question and not qa_chain:
    st.warning("Please upload at least one PDF document first.")
