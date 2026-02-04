import os
import torch
import streamlit as st
import tempfile
from pypdf import PdfReader

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from langchain_community.llms import HuggingFacePipeline

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSeq2SeqLM,
    pipeline
)

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Explainable AI Chatbot",
    layout="wide"
)

# -------------------------------------------------
# GLOBAL THEME + FIXES
# -------------------------------------------------
st.markdown(
    """
    <style>

    /* MAIN APP BACKGROUND */
    .stApp {
        background-color: white;
        color: black;
    }

    /* STREAMLIT TOP HEADER (DEPLOY BAR FIX) */
    header[data-testid="stHeader"] {
        background-color: white !important;
        border-bottom: 1px solid #eee;
    }
    header[data-testid="stHeader"] svg {
        fill: black !important;
    }

    /* INPUT FIXES (CURSOR + TEXT VISIBILITY) */
    .stTextInput input {
        background-color: #FFF3E0 !important;
        color: black !important;
        caret-color: black !important;
        border-radius: 8px;
    }

    /* PLACEHOLDER TEXT */
    .stTextInput input::placeholder {
        color: #555 !important;
    }

    /* BUTTON */
    .stButton > button {
        background: linear-gradient(90deg, #FFA000, #FFD54F);
        color: black;
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
    st.markdown("Upload **one or more PDF files** to build your knowledge base.")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.markdown("---")
    st.markdown("### Capabilities")
    st.markdown("- Semantic Search (FAISS)")
    st.markdown("- Explainable Answers")
    st.markdown("- Page-level Citations")
    st.markdown("- Local LLM (FLAN-T5)")

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 4

# -------------------------------------------------
# EMBEDDINGS CLASS
# -------------------------------------------------
class HFTransformerEmbeddings(Embeddings):
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)

    def _embed(self, text):
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            output = self.model(**inputs)
        embedding = self._mean_pooling(output, inputs["attention_mask"])
        return embedding[0].cpu().numpy().tolist()

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

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
                        metadata={"source": uploaded_file.name, "page": page_num + 1}
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
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=False
    )

    llm = HuggingFacePipeline(pipeline=pipe)

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
                response = qa_chain(question)

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Answer")
            st.write(response["result"])
            st.markdown("</div>", unsafe_allow_html=True)

            st.subheader("Sources & Explanation")
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
    st.info("Upload PDF files from the sidebar to begin.")
