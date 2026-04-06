"""
api/main.py — FastAPI application entry point.

This file creates the FastAPI app, configures it, and registers all routes.
Run this file with: uvicorn api.main:app --reload

Once running, visit:
  http://127.0.0.1:8000/docs    → Interactive Swagger UI (try all endpoints here)
  http://127.0.0.1:8000/redoc  → Alternative API documentation
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import chat, documents

# ---------------------------------------------------------------------------
# Logging configuration
# Logs are printed to the console with timestamps for easy debugging.
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG Document Q&A API",
    description=(
        "A Retrieval-Augmented Generation (RAG) API that allows you to:\n\n"
        "1. **Upload documents** (PDF, DOCX, PPTX, TXT)\n"
        "2. **Ask questions** about those documents\n"
        "3. **Get AI-generated answers** with source attribution\n\n"
        "Built with LangChain, ChromaDB, HuggingFace embeddings, and Groq LLMs.\n\n"
        "**Supported Models:**\n"
        "- `llama-3.1-8b-instant` — Fast, great for basic Q&A\n"
        "- `llama-3.3-70b-versatile` — Smart, great for deep reasoning\n"
        "- `qwen/qwen3-32b` — Advanced, great for technical content\n\n"
        "**Corrective RAG:** If retrieved context isn't relevant to the question, "
        "the system automatically rewrites the query and retrieves better results."
    ),
    version="1.0.0",
    contact={
        "name": "RAG API",
    },
    license_info={
        "name": "MIT",
    },
)

# ---------------------------------------------------------------------------
# CORS Middleware
# Allows the API to be called from a browser frontend on any origin.
# In production, replace "*" with your actual frontend domain.
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Allow all origins (change in production)
    allow_credentials=True,
    allow_methods=["*"],        # Allow GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],        # Allow all headers
)

# ---------------------------------------------------------------------------
# Register route groups
# Each router handles a specific area of the API.
# ---------------------------------------------------------------------------

app.include_router(chat.router)         # /chat/ask, /chat/reset
app.include_router(documents.router)    # /documents/upload, /documents/list, /documents/clear


# ---------------------------------------------------------------------------
# Health check endpoint
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"], summary="Health check")
def root() -> dict:
    """
    Check if the API is running.

    Returns a simple status message. Useful for deployment health checks.
    """
    return {
        "status": "running",
        "message": "RAG Document Q&A API is up. Visit /docs to explore the API.",
    }


# ---------------------------------------------------------------------------
# Startup event
# ---------------------------------------------------------------------------

@app.on_event("startup")
def on_startup() -> None:
    """
    Runs once when the server starts.
    Use this to pre-load models or validate configuration.
    """
    logger.info("RAG API server started successfully.")
    logger.info("Visit http://127.0.0.1:8000/docs to explore the API.")
