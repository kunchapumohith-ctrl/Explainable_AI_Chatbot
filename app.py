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
# GLOBAL THEME
# -------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: white; color: black; }
    header[data-testid="stHeader"] { background-color: white; border-bottom: 1px solid #eee; }
    .stTextInput input { background-color: #FFF3E0; color: black; caret-color: black; }
    .stButton > button {
        background: linear-gradient(90deg, #FFA000, #FFD54F);
        color: black; font-weight: bold; border-radius: 10px;
    }
    .card {
        background-color: white; padding: 18px; border-radius: 12px;
        box-shadow: 0px 6px 18px rgba(0,0,0,0.12);
        margin-bottom: 20px;
    }
    .source-box {
        background-color: #FFECB3; padding: 14px;
        border-radius: 10px; margin-bottom: 12px;
        border-left: 6px solid #FB8C00;
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
st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=120)
st.title("Explainable AI Chatbot")
st.markdown("### Vector Similarity Search with LLMs (RAG + XAI)")

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.header("Upload Documents")
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

    def _mean_pooling(self, output, mask):
        embeddings = output.last_hidden_state
        mask = mask.unsqueeze(-1).expand(embeddings.size()).float()
        return torch.sum(embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)

    def _embed(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            output = self.model(**inputs)
        return self._mean_pooling(output, inputs["attention_mask"])[0].cpu().numpy()

    def embed_documents(self, texts):
        return [self._embed(t).tolist() for t in texts]

    def embed_query(self, text):
        return self._embed(text).tolist()

# -------------------------------------------------
# LOAD LLM (NO PIPELINE → FIXES KEYERROR)
# -------------------------------------------------
@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)
    model.eval()

    def generate(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    return HuggingFacePipeline.from_llm(generate)

# -------------------------------------------------
# BUILD QA CHAIN
# -------------------------------------------------
@st.cache_resource
def build_qa_chain(files):
    documents = []

    for file in files:
        if file.size == 0:
            continue

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
            with st.spinner("Generating answer..."):
                response = qa_chain(question)

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Answer")
            st.write(response["result"])
            st.markdown("</div>", unsafe_allow_html=True)

            st.subheader("Sources")
            for i, doc in enumerate(response["source_documents"], 1):
                st.markdown(
                    f"""
                    <div class="source-box">
                    <b>Source {i}</b><br>
                    <b>File:</b> {doc.metadata['source']}<br>
                    <b>Page:</b> {doc.metadata['page']}<br><br>
                    {doc.page_content[:300]}...
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.warning("Please enter a question.")
else:
    st.info("Upload PDF files to begin.")
