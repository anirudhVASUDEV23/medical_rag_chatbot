# 🩺 Medical Domain Chatbot (RAG-based)

## 🧠 Project Overview

This application is a **Medical Domain Chatbot** built using **Retrieval-Augmented Generation (RAG)**. Users can upload their own medical documents (e.g., textbooks, reports), and the system intelligently answers queries by retrieving the most relevant content before generating a final response.

---

## 🎓 What is RAG?

**RAG (Retrieval-Augmented Generation)** enhances language models by supplying relevant external context from a knowledge base, preventing hallucinations and improving accuracy—especially for factual or specialized domains like medicine.

---

## 🔄 Architecture

```
User Input
   ↓
Query Embedding → Pinecone Vector DB ← Embedded Chunks ← Chunking ← PDF Loader
   ↓
Retrieved Docs
   ↓
RAG Chain (Groq + LangChain)
   ↓
LLM-generated Answer
```

_For a detailed view, refer to `MedicalAssistant.pdf` in the assets folder._

---

## 📚 Features

- Upload medical PDFs (notes, books, etc.)
- Auto-extracts text and splits into semantic chunks
- Embeds using Google/BGE embeddings
- Stores vectors in Pinecone DB
- Uses Groq's LLaMA3-70B via LangChain
- FastAPI backend with endpoints for file upload and Q&A

---

## 🌐 Tech Stack

| Component    | Tech Used                        |
|--------------|----------------------------------|
| LLM          | Groq API (LLaMA3-70B)            |
| Embeddings   | Google Generative AI / BGE        |
| Vector DB    | Pinecone                         |
| Framework    | LangChain                        |
| Backend      | FastAPI                          |
| Deployment   | Render                           |

---

## 📚 API Endpoints

- `POST /upload_pdfs/` — Upload one or more PDF files
- `POST /ask/` — Ask a question (Form field: `question`)

---

## 📁 Folder Structure

```
medical_bot/
│
├── assets/
│   ├── DIABETES.pdf
│   ├── MedicalAssistant.pdf
│   └── medicalAssistant.png
│
├── client/
│   ├── app.py
│   ├── config.py
│   ├── requirements.txt
│   ├── components/
│   │   ├── chatUI.py
│   │   ├── history_download.py
│   │   └── upload.py
│   └── utils/
│       └── api.py
│
├── server/
│   ├── .env
│   ├── logger.py
│   ├── main.py
│   ├── requirements.txt
│   ├── test.py
│   ├── uploaded_docs/
│   ├── middlewares/
│   │   └── exception_handlers.py
│   ├── modules/
│   │   ├── llm.py
│   │   ├── load_vectorstore.py
│   │   ├── pdf_handlers.py
│   │   └── query_handlers.py
│   └── routes/
│       ├── ask_question.py
│       └── upload_pdfs.py
│
└── README.md
```

---

## ⚡ Quick Setup

### Backend

```sh
# Clone the repo
git clone https://github.com/snsupratim/medicalAssistant.git
cd medicalAssistant/server

# Create virtual env
uv venv
.venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt

# Set environment variables (.env)
GOOGLE_API_KEY=...
GROQ_API_KEY=...
PINECONE_API_KEY=...

# Run the server
uvicorn main:app --reload --port 8000
```

### Frontend

```sh
cd ../client

# Create virtual env
uv venv
.venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

---

## 🌐 Deployment

- Hosted on **Render**
- Configure start command as:
  ```
  uvicorn main:app --host 0.0.0.0 --port 10000
  ```
- Streamlit frontend: https://anirudh-medicalrag.streamlit.app/

---


