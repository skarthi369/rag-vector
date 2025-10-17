#  RAG PDFBot

RAG PDFBot is a Streamlit-powered chatbot that allows you to upload multiple PDFs, embed their content using vector databases, and query them intelligently using Retrieval-Augmented Generation (RAG). It supports both Groq and Gemini as model providers, letting you configure the model, upload PDFs, and get accurate answers from the documents - all with a clean, chat-based UI.

The code is modular and extensible, allowing easy integration of additional model providers and LLMs.

---

## 🧪 How it looks like

#### Demo

![demo-gif](/assets/rag-bot-basic.gif)

#### UI

![demo-screenshot](/assets/rag-bot-basic-ui.png)

---

## 🏗️ System Architecture

![system-architecture](/assets/rag-bot-basic-architecture.png)

---

## 🚀 Features

- 🔌 **Supports Groq & Gemini models**
- 📚 **Multi-PDF support with chunked embedding**
- 📎 **Session-based chat history with download**
- ⚡ **Live AI responses with streaming UI**
- 🧠 **QA Chain with prompt template**
- 🧹 **Tools panel for reset, undo, and clear chat**
- 📥 **File uploader and context-aware chat input**

---

## 🛠️ Tech Stack

- **Frontend/UI**: Streamlit
- **LLMs**: Groq & Google Gemini via LangChain
- **Vector DB**: FAISS
- **Embeddings**: HuggingFace (for Groq) and Google (for Gemini) Embeddings
- **PDF Parsing**: PyPDF2

---

## 📦 Installation

1. **Clone the repo**

```bash
git https://github.com/skarthi369/rag-vector.git
cd rag-bot-basic
```

2. **Create a virtual environment (optional)**

```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip3 install -r requirements.txt
```

---



or else 

windows PowerShell:

(# 1. Create virtual environment (if not already created)
python -m venv .venv

# 2. Activate the virtual environment
.venv\Scripts\Activate.ps1

# 3. Upgrade pip (optional but recommended)
python -m pip install --upgrade pip

# 4. Install dependencies from requirements.txt
pip install -r requirements.txt)

## 🔐 Setup

You'll need API keys for your selected model provider:

- **Groq**: [console.groq.com](https://console.groq.com/)
- **Gemini**: [ai.google.dev](https://ai.google.dev)

---

## ▶️ Usage

Run the app:

```bash
streamlit run app.py
```

Steps:
1. Select a model provider (Groq or Gemini)
2. Enter your API key
3. Choose a model
4. Upload one or more PDFs
5. Click **Submit**
6. Ask questions about the uploaded PDFs in the chat input

---

## 📁 Project Structure

```
.
├── app.py                  # Main Streamlit app
├── data/                   # Local FAISS vector store
├── requirements.txt        # Required Python packages
└── README.md               # You're reading it!
```

---

## 🧼 Tools Panel

- **🔄 Reset**: Clears session state and reruns app
- **🧹 Clear Chat**: Clears chat + PDF submission
- **↩️ Undo**: Removes last question/response

---

## 📦 Download Chat History

Chat history is saved in the session state and can be exported as a CSV with the following columns:

| Question | Answer | Model Provider | Model Name | PDF File | Timestamp |
|----------|--------|----------------|------------|---------------------|-----------|
| What is this PDF about? | This PDF explains... | Groq | llama3-70b-8192 | file1.pdf, file2.pdf | 2025-07-03 21:00:00 |

---

## 🙏 Acknowledgements

- [LangChain](https://www.langchain.com/)
- [Streamlit](https://streamlit.io/)
- [Groq](https://console.groq.com/)
- [Google Gemini](https://ai.google.dev/)
- [FAISS](https://github.com/facebookresearch/faiss)







.\ra-venv\Scripts\python.exe -m streamlit run app.py --server.port 8503