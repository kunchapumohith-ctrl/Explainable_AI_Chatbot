import os
import torch
import streamlit as st
import tempfile
from pypdf import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from langchain_community.llms import HuggingFacePipeline

from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Explainable AI Chatbot",
    layout="wide"
)

# -------------------------------------------------
# STYLING
# -------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: white; color: black; }
    .stTextInput input { background-color: #FFF3E0; color: black; }
    .stButton>button {
        background: linear-gradient(90deg, #FFA000, #FFD54F);
        color: black; font-weight: bold; border-radius: 10px;
    }
    .card {
        background: white; padding: 18px; border-radius: 12px;
        box-shadow: 0px 6px 18px rgba(0,0,0,0.12);
        margin-bottom: 20px;
    }
    .source-box {
        background: #FFECB3; padding: 14px;
        border-left: 6px solid #FB8C00;
        border-radius: 10px; margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.title("Explainable AI Chatbot")
st.markdown("### RAG + Vector Similarity Search")

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 4

# -------------------------------------------------
# EMBEDDINGS
# -------------------------------------------------
class HFTransformerEmbeddings(Embeddings):
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def _embed(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            output = self.model(**inputs)
        embeddings = output.last_hidden_state.mean(dim=1)
        return embeddings[0].cpu().numpy().tolist()

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)

# -------------------------------------------------
# LOAD LLM ✅ CORRECT WAY
# -------------------------------------------------
@st.cache_resource
def load_llm():
    return HuggingFacePipeline.from_model_id(
        model_id=LLM_MODEL,
        task="text2text-generation",
        model_kwargs={
            "max_new_tokens": 256,
            "do_sample": False
        }
    )

# -------------------------------------------------
# BUILD QA CHAIN
# -------------------------------------------------
@st.cache_resource
def build_qa_chain(files):
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
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(documents)

    embeddings = HFTransformerEmbeddings(EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    llm = load_llm()

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

# -------------------------------------------------
# UI
# -------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

question = st.text_input(
    "Ask a question based on uploaded documents"
)

st.markdown("</div>", unsafe_allow_html=True)

if uploaded_files:
    qa_chain = build_qa_chain(uploaded_files)

    if st.button("Ask Question"):
        if question.strip():
            with st.spinner("Generating answer..."):
                result = qa_chain(question)

            st.subheader("Answer")
            st.write(result["result"])

            st.subheader("Sources")
            for doc in result["source_documents"]:
                st.markdown(
                    f"""
                    <div class="source-box">
                    <b>{doc.metadata['source']}</b> (Page {doc.metadata['page']})<br>
                    {doc.page_content[:300]}...
                    </div>
                    """,
                    unsafe_allow_html=True
                )
else:
    st.info("Upload PDF files to begin.")
