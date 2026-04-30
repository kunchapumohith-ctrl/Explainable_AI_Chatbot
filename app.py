import streamlit as st
import tempfile
import re
from typing import List
from pypdf import PdfReader
import smtplib
from email.mime.text import MIMEText

# -------- LANGCHAIN --------
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# -------- LLM --------
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline

# -------- SIMILARITY --------
from sentence_transformers import SentenceTransformer, util

# ---------------- PAGE ----------------
st.set_page_config(page_title="Explainable AI Chatbot", layout="wide")

# ---------------- SESSION STATE ----------------
if "answer" not in st.session_state:
    st.session_state.answer = ""

if "explanation" not in st.session_state:
    st.session_state.explanation = ""

if "docs" not in st.session_state:
    st.session_state.docs = []

# ---------------- GOLD UI ----------------
st.markdown("""
<style>
.stApp {background:white;color:black;}

header[data-testid="stHeader"]{
 background:white!important;
 border-bottom:1px solid #eee;
}

.stTextInput input{
 background:#FFF3E0!important;
 border-radius:8px;
 color:black !important;
 caret-color:black !important;
 font-weight:500;
}

.stTextInput input::placeholder{
 color:#555 !important;
 opacity:1 !important;
}

.stTextInput input:focus{
 outline:none !important;
 border:2px solid #FFA000 !important;
 color:black !important;
 caret-color:black !important;
}

.stButton>button,.stDownloadButton>button{
 background:linear-gradient(90deg,#FFA000,#FFD54F);
 color:black!important;
 font-weight:bold;
 border-radius:10px;
}

.card{
 background:white;
 padding:18px;
 border-radius:12px;
 box-shadow:0px 6px 18px rgba(0,0,0,0.12);
 margin-bottom:20px;
}

.source-box{
 background:#FFECB3;
 padding:14px;
 border-radius:10px;
 border-left:6px solid #FB8C00;
 margin-bottom:12px;
}

section[data-testid="stSidebar"]{
 background:linear-gradient(180deg,#FFE082,#FFD54F);
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.image(
    "https://cdn-icons-png.flaticon.com/512/4712/4712109.png",
    width=110
)

st.title("Explainable AI Chatbot")
st.markdown("### Using Vector Similarity Search and LLM")

# ---------------- CONFIG ----------------
EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL="google/flan-t5-base"

# ---------------- LOAD PDF ----------------
def load_documents(files)->List[Document]:
    docs=[]
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as tmp:
            tmp.write(file.read())
            path=tmp.name

        reader=PdfReader(path)

        for i,page in enumerate(reader.pages):
            text=page.extract_text()
            if text:
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source":file.name,"page":i+1}
                    )
                )
    return docs

# ---------------- VECTOR STORE ----------------
@st.cache_resource
def build_vectorstore(files):

    docs=load_documents(files)

    splitter=RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks=splitter.split_documents(docs)

    embeddings=HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    return FAISS.from_documents(chunks,embeddings)

# ---------------- MODELS ----------------
@st.cache_resource
def similarity_model():
    return SentenceTransformer(EMBED_MODEL)

@st.cache_resource
def load_llm():
    tokenizer=AutoTokenizer.from_pretrained(LLM_MODEL)
    model=AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)

    pipe=pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=120,
        do_sample=False
    )

    return HuggingFacePipeline(pipeline=pipe)

# ---------------- EXACT ANSWER EXTRACTION ----------------
def extract_exact_answer(docs,question):

    model=similarity_model()
    q_emb=model.encode(question,convert_to_tensor=True)

    best=""
    best_score=-1

    for doc in docs:
        sentences=re.split(r'(?<=[.!?]) +',doc.page_content)

        sent_emb=model.encode(sentences,convert_to_tensor=True)
        scores=util.cos_sim(q_emb,sent_emb)[0]

        for s,sc in zip(sentences,scores):
            if float(sc)>best_score:
                best_score=float(sc)
                best=s

    return best.strip()

# ---------------- LLM EXPLANATION ----------------
def explain_answer(answer,question):

    llm=load_llm()

    prompt=f"""
Explain the following answer simply.

Question: {question}
Answer: {answer}
Explanation:
"""
    return llm.invoke(prompt)

# ---------------- EMAIL FUNCTION ----------------
def send_email(receiver,answer):

    sender=st.secrets["EMAIL"]
    password=st.secrets["APP_PASSWORD"]

    msg=MIMEText(answer,"plain","utf-8")
    msg["Subject"]="Explainable AI Chatbot Answer"
    msg["From"]=sender
    msg["To"]=receiver

    server=smtplib.SMTP("smtp.gmail.com",587)
    server.starttls()
    server.login(sender,password)
    server.sendmail(sender,receiver,msg.as_string())
    server.quit()

# ---------------- HIGHLIGHT ----------------
def highlight_text(text,question,answer):

    text = " ".join(text.split())  # ✅ FIXED EVIDENCE LINE BREAK ISSUE

    model=similarity_model()

    sentences=re.split(r'(?<=[.!?]) +',text)

    ref=question+" "+answer
    ref_emb=model.encode(ref,convert_to_tensor=True)
    sent_emb=model.encode(sentences,convert_to_tensor=True)

    scores=util.cos_sim(ref_emb,sent_emb)[0]

    result=""
    for sent,score in zip(sentences,scores):
        if float(score)>0.55:
            result+=f"<span style='color:#b71c1c;font-weight:bold'>{sent}</span> "
        else:
            result+=sent+" "

    return result

# ---------------- SIDEBAR ----------------
with st.sidebar:

    st.header("Browse Files")

    uploaded_files=st.file_uploader(
        "Upload PDF Documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.markdown("---")
    st.markdown("### Features")
    st.markdown("""
✅ Vector Similarity Search  
✅ Accurate PDF Answers  
✅ LLM Explanation  
✅ Evidence Highlighting  
✅ Email Delivery
""")

# ---------------- INPUT ----------------
question=st.text_input("Ask question from uploaded documents")

# ---------------- PROCESS ----------------
if uploaded_files:

    vectorstore=build_vectorstore(uploaded_files)

    if st.button("Ask Question"):

        docs=vectorstore.similarity_search(question,k=4)

        st.session_state.docs=docs
        st.session_state.answer=extract_exact_answer(docs,question)
        st.session_state.explanation=explain_answer(
            st.session_state.answer,
            question
        )

# ---------------- ANSWER DISPLAY ----------------
if st.session_state.answer:

    st.markdown("<div class='card'>",unsafe_allow_html=True)
    st.subheader("Answer")
    st.write(" ".join(st.session_state.answer.split()))
    st.markdown("</div>",unsafe_allow_html=True)

    st.markdown("<div class='card'>",unsafe_allow_html=True)
    st.subheader("Explanation")
    st.write(st.session_state.explanation)
    st.markdown("</div>",unsafe_allow_html=True)

    st.download_button(
        "Download Answer",
        data=st.session_state.answer,
        file_name="answer.txt"
    )

# ---------------- EMAIL UI ----------------
st.markdown("<div class='card'>",unsafe_allow_html=True)
st.subheader("Send Answer to Email")

email=st.text_input("Enter Receiver Email")

if st.button("Send Email"):

    if not st.session_state.answer:
        st.warning("Generate answer first.")
    elif email.strip()=="":
        st.warning("Enter email address.")
    else:
        send_email(email,st.session_state.answer)
        st.success("✅ Email sent successfully!")

st.markdown("</div>",unsafe_allow_html=True)

# ---------------- SOURCES ----------------
if st.session_state.docs:

    st.subheader("Source Evidence")

    for i,doc in enumerate(st.session_state.docs[:2],1):

        highlighted=highlight_text(
            doc.page_content[:1500],
            question,
            st.session_state.answer
        )

        st.markdown(f"""
        <div class="source-box">
        <b>Source {i}</b><br>
        Document: {doc.metadata['source']}<br>
        Page: {doc.metadata['page']}<br><br>
        {highlighted}
        </div>
        """,unsafe_allow_html=True)