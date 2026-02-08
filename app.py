import streamlit as st
import tempfile
from pypdf import PdfReader

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


# -----------------------------
# PAGE CONFIG (UI FIX)
# -----------------------------
st.set_page_config(
    page_title="Explainable AI Chatbot",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
        body {
            background-color: white;
        }
        .stApp {
            background-color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# LOAD LLM (NO PIPELINE ❌)
# -----------------------------
@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    def generate(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = model.generate(
            **inputs,
            max_new_tokens=256
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generate


# -----------------------------
# BUILD VECTOR STORE
# -----------------------------
@st.cache_resource
def build_vectorstore(uploaded_files):
    docs = []

    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            reader = PdfReader(tmp.name)

            for page in reader.pages:
                text = page.extract_text()
                if text:
                    docs.append(Document(page_content=text))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore


# -----------------------------
# UI
# -----------------------------
st.title("📄 Explainable AI Chatbot")
st.write("Ask questions based on your uploaded PDF documents.")

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

question = st.text_input("Ask a question based on your documents")

# -----------------------------
# MAIN LOGIC
# -----------------------------
if uploaded_files and question:
    with st.spinner("Processing documents..."):
        vectorstore = build_vectorstore(uploaded_files)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        docs = retriever.get_relevant_documents(question)
        context = "\n\n".join([d.page_content for d in docs])

        llm = load_llm()

        prompt = f"""
Answer the question using only the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""

        answer = llm(prompt)

    st.subheader("✅ Answer")
    st.write(answer)
