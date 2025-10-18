from datetime import datetime

import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader

from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
    "models": ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]  # Updated to current production models (post-deprecation)
  },
  "Gemini": {
    "playground": "https://aistudio.google.com/app/apikey",  # More direct link for API key
    "models": ["gemini-1.5-flash", "gemini-1.5-pro"]  # Updated to stable models
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
  llm = ChatGroq(model=model, api_key=api_key) if provider.lower() == "groq" else ChatGoogleGenerativeAI(model=model, google_api_key=api_key)
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

def render_footer():
    """Renders a professional footer with branding and links."""
    st.markdown("---")  # Horizontal divider
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.markdown("**👨‍💼 AugmentIQ**")
        st.markdown("*AI-Powered Knowledge Tools*")
    
    with col2:
        st.markdown("**🔗 Quick Links**")
        st.markdown("[GitHub Repo](https://github.com/yourusername/augmentiq-rag-pdfbot) | [Demo Video](https://www.linkedin.com/posts/your-post-link) | [Contact Us](mailto:hello@augmentiq.com)")
        st.caption("© 2025 AugmentIQ. All rights reserved. | Privacy Policy")
    
    with col3:
        st.markdown("**🛠️ Built With**")
        st.markdown("Streamlit • LangChain • FAISS • Groq/Gemini")
    
    st.markdown("---")

# ------------------ Main App ------------------ #
def main():
  st.set_page_config(page_title="RAG PDFBot", layout="centered")
  st.title(" RAG PDFBot")
  st.caption("Chat with multiple PDFs :books:")

  # Initialize session state
  defaults = {
    "chat_history": [],
    "pdfs_submitted": False,
    "vector_store": None,
    "pdf_files": [],
    "last_provider": None,
    "unsubmitted_files": False,
  }
  for key, default in defaults.items():
    if key not in st.session_state:
      st.session_state[key] = default

  # Sidebar Configuration
  with st.sidebar:
    with st.expander("⚙️ Configuration", expanded=True):
      model_provider_key = "model_provider_select"  # Unique key to avoid conflicts
      model_provider = st.selectbox("🔌 Model Provider", ["Select a model provider"] + list(MODEL_OPTIONS.keys()), index=0, key=model_provider_key)

      if model_provider == "Select a model provider":
        st.stop()  # Early exit instead of return for better flow

      api_key_key = f"api_key_{model_provider.lower()}"
      api_key = st.text_input("🔑 Enter your API Key", help=f"Get API key from [here]({MODEL_OPTIONS[model_provider]['playground']})", key=api_key_key, type="password")
      if not api_key:
        st.stop()

      models_key = f"model_select_{model_provider.lower()}"
      models = MODEL_OPTIONS[model_provider]["models"]
      model = st.selectbox("🧠 Select a model", models, key=models_key)

      uploader_key = "pdf_uploader"
      uploaded_files = st.file_uploader("📚 Upload PDFs", type=["pdf"], accept_multiple_files=True, key=uploader_key)

      if uploaded_files and uploaded_files != st.session_state.pdf_files:
        st.session_state.unsubmitted_files = True

      if st.button("➡️ Submit", key="submit_btn"):
        if uploaded_files:
          with st.spinner("Processing PDFs..."):
            process_and_store_pdfs(uploaded_files, model_provider, api_key)
            st.session_state.pdf_files = uploaded_files
            st.session_state.unsubmitted_files = False
            st.session_state.last_provider = model_provider  # Track here too
            st.toast("PDFs processed successfully!", icon="✅")
        else:
          st.warning("No files uploaded.")

      # Reprocess on provider change
      if model_provider != st.session_state.last_provider and st.session_state.pdf_files:
        st.session_state.last_provider = model_provider
        with st.spinner("Reprocessing PDFs..."):
          process_and_store_pdfs(st.session_state.pdf_files, model_provider, api_key)
          st.toast("PDFs reprocessed successfully!", icon="🔁")

    with st.expander("🛠️ Tools", expanded=False):
      col1, col2, col3 = st.columns(3)

      if col1.button("🔄 Reset", key="reset_btn"):
        for key in list(st.session_state.keys()):
          del st.session_state[key]
        st.rerun()

      if col2.button("🧹 Clear Chat", key="clear_btn"):
        st.session_state.chat_history = []
        st.session_state.pdf_files = []
        st.session_state.vector_store = None
        st.session_state.pdfs_submitted = False
        st.session_state.unsubmitted_files = False
        st.toast("Chat and PDF cleared.", icon="🧼")
        st.rerun()

      if col3.button("↩️ Undo", key="undo_btn") and st.session_state.chat_history:
        st.session_state.chat_history.pop()
        st.rerun()

  if st.session_state.pdfs_submitted and st.session_state.pdf_files:
    render_uploaded_files()

  # Render chat history
  for q, a, *_ in st.session_state.chat_history:
    with st.chat_message("user"):
      st.markdown(q)
    with st.chat_message("ai"):
      st.markdown(a)

  if st.session_state.unsubmitted_files:
    st.warning("📄 New PDFs uploaded. Please submit before chatting.")
    st.stop()

  if not st.session_state.pdfs_submitted:
    st.info("📄 Please upload and submit PDFs to start chatting.")
    st.stop()

  # Chat input
  question = st.chat_input("💬 Ask a Question from the PDF Files")
  if question:
    with st.chat_message("user"):
      st.markdown(question)
    with st.chat_message("ai"):
      with st.spinner("Thinking..."):
        try:
          # Use current provider/model from sidebar (fetch fresh to avoid stale state)
          current_provider = st.session_state[model_provider_key]
          current_model = st.session_state[models_key]
          current_api_key = st.session_state[api_key_key]
          docs = st.session_state.vector_store.similarity_search(question)
          chain = get_qa_chain(current_provider, current_model, current_api_key)
          output = chain({"input_documents": docs, "question": question}, return_only_outputs=True)["output_text"]
          st.markdown(output)
          pdf_names = [f.name for f in st.session_state.pdf_files]
          st.session_state.chat_history.append((question, output, current_provider, current_model, pdf_names, datetime.now()))
        except Exception as e:
          st.error(f"Error: {str(e)}")

  if st.session_state.chat_history:
    render_download_chat_history()

  render_footer()  # Render the footer at the bottom

if __name__ == "__main__":
  main()
