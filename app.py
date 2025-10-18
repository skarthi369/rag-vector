from datetime import datetime
import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader

from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_groq import ChatGroq

# ------------------ Model Configuration ------------------ #
MODEL_OPTIONS = {
    "Groq": {
        "playground": "https://console.groq.com/",
        "models": ["llama-3.1-8b-instant", "llama3-70b-8192"]
    },
    "Gemini": {
        "playground": "https://generativelanguage.googleapis.com/v1/models",
        "models": ["gemini-2.0-flash", "gemini-2.5-flash"]
    }
}

# ------------------ Utility Functions ------------------ #
def get_pdf_text(pdf_files):
    text = ""
    for file in pdf_files:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    return splitter.split_text(text)

def get_embeddings(provider, api_key=None):
    if provider.lower() == "groq":
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    elif provider.lower() == "gemini":
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    else:
        raise ValueError("Unsupported provider")

def get_vectorstore(chunks, provider, api_key):
    embedding = get_embeddings(provider, api_key)
    store = FAISS.from_texts(chunks, embedding)
    store.save_local(f"./data/{provider.lower()}_vector_store.faiss")
    return store

def load_vectorstore(provider, api_key):
    embedding = get_embeddings(provider, api_key)
    return FAISS.load_local(f"./data/{provider.lower()}_vector_store.faiss", embedding, allow_dangerous_deserialization=True)

def get_qa_chain(provider, model, api_key):
    prompt = PromptTemplate(
        template="""
        Answer the question as detailed as possible.
        If the question cannot be answered using the provided context, please say "I don't know."

        Context:
        {context}

        Question:
        {question}?

        Answer:
        """,
        input_variables=["context", "question"]
    )
    llm = ChatGroq(model=model, api_key=api_key) if provider.lower() == "groq" else ChatGoogleGenerativeAI(model=model, api_key=api_key)
    return load_qa_chain(llm, chain_type="stuff", prompt=prompt)

def process_and_store_pdfs(pdfs, provider, api_key):
    raw_text = get_pdf_text(pdfs)
    chunks = get_text_chunks(raw_text)
    store = get_vectorstore(chunks, provider, api_key)
    st.session_state.vector_store = store
    st.session_state.pdfs_submitted = True

def render_uploaded_files():
    pdf_files = st.session_state.get("pdf_files", [])
    if pdf_files:
        with st.expander("**📎 Uploaded Files:**"):
            for f in pdf_files:
                st.markdown(f"- {f.name}")

def render_download_chat_history():
    df = pd.DataFrame(st.session_state.chat_history, columns=["Question", "Answer", "Model", "Model Name", "PDF File", "Timestamp"])
    with st.expander("**📎 Download Chat History:**"):
        st.sidebar.download_button("📥 Download Chat History", data=df.to_csv(index=False), file_name="chat_history.csv", mime="text/csv")

# ------------------ Main App ------------------ #
def main():
    st.set_page_config(page_title="RAG PDFBot", layout="centered")
    st.title(" RAG PDFBot")
    st.caption("Chat with multiple PDFs :books:")

    for key, default in {
        "chat_history": [],
        "pdfs_submitted": False,
        "vector_store": None,
        "pdf_files": [],
        "last_provider": None,
        "unsubmitted_files": False,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # Sidebar Configuration
    with st.sidebar:
        with st.expander("⚙️ Configuration", expanded=True):
            model_provider = st.selectbox("🔌 Model Provider", ["Select a model provider"] + list(MODEL_OPTIONS.keys()), index=0, key="model_provider")

            if model_provider == "Select a model provider":
                return

            api_key = st.text_input("🔑 Enter your API Key", help=f"Get API key from [here]({MODEL_OPTIONS[model_provider]['playground']})")
            if not api_key:
                return

            models = MODEL_OPTIONS[model_provider]["models"]
            model = st.selectbox("🧠 Select a model", models, key="model")

            uploaded_files = st.file_uploader("📚 Upload PDFs", type=["pdf"], accept_multiple_files=True, key="pdf_uploader")

            if uploaded_files and uploaded_files != st.session_state.pdf_files:
                st.session_state.unsubmitted_files = True

            if st.button("➡️ Submit"):
                if uploaded_files:
                    with st.spinner("Processing PDFs..."):
                        process_and_store_pdfs(uploaded_files, model_provider, api_key)
                        st.session_state.pdf_files = uploaded_files
                        st.session_state.unsubmitted_files = False
                        st.toast("PDFs processed successfully!", icon="✅")
                else:
                    st.warning("No files uploaded.")

            if model_provider != st.session_state.last_provider:
                st.session_state.last_provider = model_provider
                if st.session_state.pdf_files:
                    with st.spinner("Reprocessing PDFs..."):
                        process_and_store_pdfs(st.session_state.pdf_files, model_provider, api_key)
                        st.toast("PDFs reprocessed successfully!", icon="🔁")

        with st.expander("🛠️ Tools", expanded=False):
            col1, col2, col3 = st.columns(3)

            if col1.button("🔄 Reset"):
                st.session_state.clear()
                st.session_state.model_provider = "Select a model provider"
                st.rerun()

            if col2.button("🧹 Clear Chat"):
                st.session_state.chat_history = []
                st.session_state.pdf_files = None
                st.session_state.vector_store = None
                st.session_state.pdfs_submitted = False
                st.toast("Chat and PDF cleared.", icon="🧼")

            if col3.button("↩️ Undo") and st.session_state.chat_history:
                st.session_state.chat_history.pop()
                st.rerun()

    if st.session_state.pdfs_submitted and st.session_state.pdf_files:
        render_uploaded_files()

    for q, a, *_ in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("ai"):
            st.markdown(a)

    if st.session_state.unsubmitted_files:
        st.warning("📄 New PDFs uploaded. Please submit before chatting.")
        return

    if st.session_state.pdfs_submitted:
        question = st.chat_input("💬 Ask a Question from the PDF Files")
        if question:
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("ai"):
                with st.spinner("Thinking..."):
                    try:
                        docs = st.session_state.vector_store.similarity_search(question)
                        chain = get_qa_chain(model_provider, model, api_key)
                        output = chain({"input_documents": docs, "question": question}, return_only_outputs=True)["output_text"]
                        st.markdown(output)
                        pdf_names = [f.name for f in st.session_state.pdf_files]
                        st.session_state.chat_history.append((question, output, model_provider, model, pdf_names, datetime.now()))
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    else:
        st.info("📄 Please upload and submit PDFs to start chatting.")

    if st.session_state.chat_history:
        render_download_chat_history()

if __name__ == "__main__":
    main()
