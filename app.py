import tempfile
import streamlit as st
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Explainable AI Chatbot",
    layout="wide"
)

# -------------------------------------------------
# UI FIX
# -------------------------------------------------
st.markdown("""
<style>
.stApp { background-color: white; color: black; }
h1, h2, h3, h4 { color: black; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.title("🤖 Explainable AI Chatbot")
st.markdown("### PDF-based Question Answering (RAG)")

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.header("Upload PDFs")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

# -------------------------------------------------
# MODELS
# -------------------------------------------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"

# -------------------------------------------------
# VECTOR STORE
# -------------------------------------------------
@st.cache_resource
def build_vectorstore(files):
    documents = []

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            path = tmp.name

        reader = PdfReader(path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": file.name, "page": i + 1}
                    )
                )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return FAISS.from_documents(chunks, embeddings)

# -------------------------------------------------
# LOAD LLM (FIXED)
# -------------------------------------------------
@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)

    return pipeline(
        task="text2text-generation",   # 🔥 THIS FIXES THE KEYERROR
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256
    )

# -------------------------------------------------
# MAIN APP
# -------------------------------------------------
question = st.text_input("Ask a question based on uploaded PDFs")

if uploaded_files:
    vectorstore = build_vectorstore(uploaded_files)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = load_llm()

    if st.button("Ask"):
        with st.spinner("Generating answer..."):
            docs = retriever.get_relevant_documents(question)

            context = "\n\n".join(
                f"Source: {d.metadata['source']} (Page {d.metadata['page']})\n{d.page_content}"
                for d in docs
            )

            prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}

Answer:
"""
            result = llm(prompt)[0]["generated_text"]

        st.subheader("Answer")
        st.write(result)

        st.subheader("Sources")
        for d in docs:
            st.markdown(f"- **{d.metadata['source']} (Page {d.metadata['page']})**")
else:
    st.info("Upload PDF files to start.")
