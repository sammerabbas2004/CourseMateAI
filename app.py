
import hashlib
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFacePipeline,
    ChatHuggingFace,
)
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ---------------- CONFIGURATION ----------------

st.set_page_config(
    page_title="Course Mate AI",
    page_icon="📚",
    layout="wide",
)

DB_DIR = Path("Chroma-db")
UPLOAD_DIR = Path("uploaded_books")

DB_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

# ---------------- MODELS ----------------

@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )


@st.cache_resource
def get_chat_model():
    llm = HuggingFacePipeline.from_model_id(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 300,
            "do_sample": False,
            "return_full_text": False,
        },
    )
    return ChatHuggingFace(llm=llm)


embedding_model = get_embedding_model()

PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are Course Mate AI, a helpful study assistant. "
        "Answer using only the supplied book context. "
        "If the context does not contain the answer, say "
        "'I don't know based on the uploaded book.' "
        "Do not invent facts."
    ),
    (
        "human",
        "Book context:\n{context}\n\n"
        "Student question: {question}\n\n"
        "Give a clear, helpful answer."
    ),
])

# ---------------- PDF PROCESSING ----------------

def process_pdf(pdf_bytes, filename):
    # Identify the exact uploaded file
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()
    book_id = file_hash[:20]

    pdf_path = UPLOAD_DIR / f"{book_id}.pdf"
    db_path = DB_DIR / book_id

    if not pdf_path.exists():
        pdf_path.write_bytes(pdf_bytes)

    # Reuse the index if this exact PDF was indexed previously
    if (db_path / "chroma.sqlite3").exists():
        vectorstore = Chroma(
            persist_directory=str(db_path),
            embedding_function=embedding_model,
        )
        return vectorstore, book_id

    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    if not documents:
        raise ValueError("No readable pages were found in this PDF.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError(
            "No text could be extracted. The PDF may be scanned."
        )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=str(db_path),
    )

    return vectorstore, book_id


# ---------------- SESSION STATE ----------------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "book_id" not in st.session_state:
    st.session_state.book_id = None

if "book_name" not in st.session_state:
    st.session_state.book_name = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed_file_hash" not in st.session_state:
    st.session_state.indexed_file_hash = None


# ---------------- CUSTOM STYLING ----------------

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .hero {
        padding: 1.5rem;
        border-radius: 16px;
        background: linear-gradient(120deg, #172554, #1d4ed8);
        color: white;
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        color: white;
        margin-bottom: 0.35rem;
    }
    .hero p {
        color: #dbeafe;
        margin-bottom: 0;
    }
    [data-testid="stChatMessage"] {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.title("📚 Course Mate AI")
    st.caption("Your personal book study assistant")

    st.divider()
    st.subheader("Upload your book")

    uploaded_file = st.file_uploader(
        "Choose a PDF book",
        type=["pdf"],
        help="Upload a text-based PDF to create its searchable index.",
    )

    if uploaded_file:
        st.caption(f"Selected: {uploaded_file.name}")
        st.caption(
            f"File size: {uploaded_file.size / (1024 * 1024):.2f} MB"
        )

        if st.button("Process Book", type="primary", use_container_width=True):
            pdf_bytes = uploaded_file.getvalue()
            current_hash = hashlib.sha256(pdf_bytes).hexdigest()[:20]

            try:
                with st.spinner(
                    "Reading PDF, creating embeddings, and indexing..."
                ):
                    vectorstore, book_id = process_pdf(
                        pdf_bytes, uploaded_file.name
                    )

                st.session_state.vectorstore = vectorstore
                st.session_state.book_id = book_id
                st.session_state.book_name = uploaded_file.name

                if st.session_state.indexed_file_hash != current_hash:
                    st.session_state.messages = []

                st.session_state.indexed_file_hash = current_hash

                st.success("Book is ready to chat with!")
            except Exception as exc:
                st.error(f"Could not process this PDF: {exc}")

    st.divider()

    if st.session_state.vectorstore is not None:
        st.success("Book indexed")
        st.write(f"**Current book:** {st.session_state.book_name}")

        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        if st.button("Unload current book", use_container_width=True):
            st.session_state.vectorstore = None
            st.session_state.book_id = None
            st.session_state.book_name = None
            st.session_state.messages = []
            st.session_state.indexed_file_hash = None
            st.rerun()
    else:
        st.info("Upload and process a PDF to get started.")

    st.caption("Powered by LangChain · Hugging Face · ChromaDB")


# ---------------- MAIN PAGE ----------------

st.markdown("""
<div class="hero">
    <h1>Course Mate AI 🎓</h1>
    <p>Upload a book, explore its contents, and ask questions in natural language.</p>
</div>
""", unsafe_allow_html=True)

if st.session_state.vectorstore is None:
    st.subheader("Start learning with your own book")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📄 Upload")
        st.write("Choose a PDF book from your computer.")

    with col2:
        st.markdown("### 🧠 Understand")
        st.write("The app creates searchable chunks and embeddings.")

    with col3:
        st.markdown("### 💬 Ask")
        st.write("Ask questions and get answers based on your book.")

    st.info(
        "To begin, upload a PDF in the sidebar and click **Process Book**."
    )

else:
    st.subheader(f"💬 Chat about {st.session_state.book_name}")
    st.caption("Answers are generated using relevant passages from your book.")

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("View source pages"):
                    st.write(message["sources"])

    # Chat input
    question = st.chat_input("Ask a question about your book...")

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching your book and generating an answer..."):
                    retriever = st.session_state.vectorstore.as_retriever(
                        search_type="mmr",
                        search_kwargs={
                            "k": 3,
                            "fetch_k": 10,
                            "lambda_mult": 0.5,
                        },
                    )

                    docs = retriever.invoke(question)

                    if not docs:
                        answer = "I don't know based on the uploaded book."
                        source_text = []
                    else:
                        context = "\n\n".join(
                            f"[Page {doc.metadata.get('page', 0) + 1}]\n"
                            f"{doc.page_content}"
                            for doc in docs
                        )

                        model = get_chat_model()
                        chain = PROMPT | model

                        result = chain.invoke({
                            "context": context,
                            "question": question,
                        })

                        answer = result.content
                        source_text = []

                        for doc in docs:
                            page = doc.metadata.get("page", 0) + 1
                            source_text.append(
                                f"Page {page}: "
                                f"{doc.page_content[:250].strip()}..."
                            )

                st.markdown(answer)

                if source_text:
                    with st.expander("View source pages"):
                        for source in source_text:
                            st.markdown(f"- {source}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": source_text,
                })

            except Exception as exc:
                st.error(f"An error occurred: {exc}")
                st.info(
                    "Check your model download, dependencies, and available RAM."
                )
