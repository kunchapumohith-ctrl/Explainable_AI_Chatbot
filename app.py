import streamlit as st
import tempfile
from typing import List

from pypdf import PdfReader

# LangChain (PINNED, STABLE IMPORTS)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ------------------ CONFIG ------------------
st.set_page_config(
    page_title="Explainable AI Chatbot",
    page_icon="🤖",
    layout="wide"
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"

# ------------------ UI FIX (WHITE BACKGROUND) ------------------
st.markdown(
    """
    <style>
        .stApp {
            background-color: white;
            color: black;
        }
        h1, h2, h3, h4 {
            color: black;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------ TITLE ------------------
st.title("🤖 Explainable AI Chatbot")
st.write("Upload PDFs and ask questions based on their content.")

# ------------------ PDF PROCESSING ------------------
def load_documents(files) -> List[Document]:
    documents = []

    for file in files:
        if file.size == 0:
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        reader = PdfReader(tmp_path)
        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": file.name}
                )
            )

    return documents


# ------------------ VECTOR STORE ------------------
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


# ------------------ LLM ------------------
@st.cache_resource(show_spinner="🤖 Loading language model...")
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)

    text_gen_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256
    )

    return HuggingFacePipeline(pipeline=text_gen_pipeline)


# ------------------ QA CHAIN ------------------
@st.cache_resource(show_spinner="🔗 Building QA chain...")
def build_qa_chain(files):
    vectorstore = build_vectorstore(files)
    llm = load_llm()

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )


# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.header("📄 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type="pdf",
        accept_multiple_files=True
    )

# ------------------ MAIN APP ------------------
if uploaded_files:
    qa_chain = build_qa_chain(uploaded_files)

    question = st.text_input("❓ Ask a question from the documents")

    if question:
        with st.spinner("Thinking..."):
            result = qa_chain.invoke(question)

        st.subheader("✅ Answer")
        st.write(result["result"])

        with st.expander("📚 Source Documents"):
            for doc in result["source_documents"]:
                st.markdown(f"**Source:** {doc.metadata.get('source')}")
                st.write(doc.page_content[:500] + "...")
else:
    st.info("⬅️ Upload PDFs from the sidebar to begin.")
