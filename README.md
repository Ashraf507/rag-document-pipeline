# RAG Document Q&A API

A production-ready **Retrieval-Augmented Generation (RAG)** backend API that lets you upload documents and ask questions about them using AI.

Built with **FastAPI**, **LangChain**, **ChromaDB**, and **Groq LLMs**.

---

## What It Does

1. **Upload documents** — PDF, DOCX, PPTX, or TXT files.
2. **Automatic indexing** — Documents are split into chunks and stored in a vector database.
3. **Ask questions** — Send a question and get an AI-generated answer grounded in your documents.
4. **Source attribution** — Every answer includes which document it came from.
5. **Corrective RAG** — If the retrieved context isn't relevant, the system automatically rewrites your query and tries again.
6. **Model switching** — Choose between fast, smart, or advanced Groq LLMs per request.

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI |
| LLM Provider | Groq (Llama 3, Qwen) |
| RAG Orchestration | LangChain |
| Vector Database | ChromaDB |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Frontend (optional) | Reflex |

---

## Project Structure

```
RAG/
├── api/                        # FastAPI REST API layer
│   ├── main.py                 # App entry point — run this to start the server
│   ├── routes/
│   │   ├── chat.py             # POST /chat/ask, DELETE /chat/reset
│   │   └── documents.py        # POST /documents/upload, GET /documents/list
│   └── schemas/
│       ├── chat.py             # Pydantic models for chat requests/responses
│       └── documents.py        # Pydantic models for document requests/responses
│
├── RAG/
│   └── backend/
│       └── rag.py              # Core RAG engine (loading, chunking, retrieval, generation)
│
├── tests/
│   ├── test_rag_backend.py     # Unit tests for the RAG engine
│   └── test_api.py             # Integration tests for all API endpoints
│
├── .env.example                # Template for required environment variables
├── requirements.txt            # All Python dependencies
└── README.md                   # This file
```

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd RAG
```

### 2. Create a virtual environment

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
# Copy the template and fill in your API key
cp .env.example .env
```

Open `.env` and set your Groq API key:
```
GROQ_API_KEY=your_groq_api_key_here
```

> Get a free Groq API key at: https://console.groq.com

---

## Running the API

```bash
uvicorn api.main:app --reload
```

The server starts at: **http://127.0.0.1:8000**

### Interactive API Docs

Once running, open your browser to:

- **http://127.0.0.1:8000/docs** → Swagger UI (try all endpoints interactively)
- **http://127.0.0.1:8000/redoc** → ReDoc documentation

---

## API Endpoints

### Health Check

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Check if the server is running |

### Chat

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/chat/ask` | Send a question and get an AI answer |
| `DELETE` | `/chat/reset` | Clear all uploaded documents from the session |

### Documents

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/documents/upload` | Upload a document (PDF, DOCX, PPTX, TXT) |
| `GET` | `/documents/list` | List all documents in the current session |
| `DELETE` | `/documents/clear` | Clear all documents without resetting chat |

---

## Example Usage

### 1. Upload a document

```bash
curl -X POST "http://127.0.0.1:8000/documents/upload" \
  -F "file=@report.pdf"
```

**Response:**
```json
{
  "filename": "report.pdf",
  "chunks_added": 42,
  "message": "File uploaded and indexed successfully."
}
```

### 2. Ask a question

```bash
curl -X POST "http://127.0.0.1:8000/chat/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the key findings in this report?",
    "history": [],
    "model": "llama-3.1-8b-instant"
  }'
```

**Response:**
```json
{
  "answer": "The report highlights three key findings: ...",
  "sources": ["/path/to/report.pdf"],
  "model_used": "llama-3.1-8b-instant"
}
```

### 3. Continue a conversation (with history)

```bash
curl -X POST "http://127.0.0.1:8000/chat/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Can you elaborate on the second finding?",
    "history": [
      {"role": "user", "content": "What are the key findings?"},
      {"role": "assistant", "content": "The three key findings are..."}
    ],
    "model": "llama-3.3-70b-versatile"
  }'
```

---

## Available Models

| Label | Model ID | Best For |
|---|---|---|
| Fast | `llama-3.1-8b-instant` | Basic Q&A, quick summaries |
| Smart | `llama-3.3-70b-versatile` | Deep reasoning, research |
| Advanced | `qwen/qwen3-32b` | Complex technical content |

---

## How Corrective RAG Works

Standard RAG sometimes retrieves irrelevant chunks. This project implements **Corrective RAG** to fix that:

```
User Question
      │
      ▼
[Retrieve top-K chunks from vector DB]
      │
      ▼
[LLM checks: Is this context relevant?]
      │
   YES ──────────────────────────────► [Generate answer] ──► Response
      │
   NO
      │
      ▼
[LLM rewrites the query using chat history]
      │
      ▼
[Retrieve again with improved query]
      │
      ▼
[Generate answer] ──────────────────► Response
```

---

## Running Tests

```bash
pytest tests/ -v
```

To run only unit tests:
```bash
pytest tests/test_rag_backend.py -v
```

To run only API tests:
```bash
pytest tests/test_api.py -v
```

---

## Supported File Types

| Extension | Format |
|---|---|
| `.pdf` | PDF documents |
| `.docx` / `.doc` | Microsoft Word |
| `.pptx` / `.ppt` | Microsoft PowerPoint |
| `.txt` | Plain text |

---

## Optional: Running the Web UI

This project also includes a Reflex-based web frontend.

```bash
reflex run
```

The web app runs at: **http://localhost:3000**

> Note: The web UI and the REST API are separate. You can use either or both.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | Your Groq API key for LLM access |
