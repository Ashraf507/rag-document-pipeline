"""
rag.py — Core RAG (Retrieval-Augmented Generation) Engine

This module handles:
  - Loading documents (PDF, DOCX, PPTX, TXT)
  - Splitting them into chunks
  - Storing chunks in a vector database (ChromaDB)
  - Retrieving relevant chunks for a given query
  - Corrective RAG: relevance check + query rewriting if needed
  - Generating final answers using a Groq LLM
"""

import os
from typing import Optional

from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
)
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load API keys from .env file
load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHUNK_SIZE = 3000           # Max characters per text chunk
CHUNK_OVERLAP = 200         # Overlap between chunks to preserve context
TOP_K_RESULTS = 4           # Number of similar chunks to retrieve per query
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SUPPORTED_EXTENSIONS = {
    ".pdf": PyMuPDFLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".ppt": UnstructuredPowerPointLoader,
}

# Simple greetings that don't need document retrieval
GREETINGS = {"hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"}

# ---------------------------------------------------------------------------
# Embeddings (loaded once, shared globally — expensive to reload)
# ---------------------------------------------------------------------------

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if hf_token:
    # Use the server-less Inference API (saves local RAM/disk)
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=hf_token,
        model_name=EMBEDDING_MODEL,
    )
else:
    # Fallback: Load the model locally using sentence-transformers
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# ---------------------------------------------------------------------------
# In-memory session state (one session at a time)
# ---------------------------------------------------------------------------

vectorstore: Optional[Chroma] = None
retriever = None


# ---------------------------------------------------------------------------
# Document Loading
# ---------------------------------------------------------------------------

def load_document(file_path: str) -> list[Document]:
    """
    Load a document from disk based on its file extension.

    Supports: PDF, DOCX, DOC, PPTX, PPT, and plain text files.

    Args:
        file_path: Absolute path to the file on disk.

    Returns:
        A list of LangChain Document objects.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not supported.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext in SUPPORTED_EXTENSIONS:
        loader = SUPPORTED_EXTENSIONS[ext](file_path)
    else:
        # Default to plain text for .txt and unknown extensions
        loader = TextLoader(file_path, encoding="utf-8")

    documents = loader.load()

    # ChromaDB requires all metadata values to be strings
    for doc in documents:
        doc.metadata = {k: str(v) for k, v in doc.metadata.items()}

    return documents


# ---------------------------------------------------------------------------
# Text Splitting
# ---------------------------------------------------------------------------

def split_into_chunks(documents: list[Document]) -> list[Document]:
    """
    Split large documents into smaller overlapping chunks.

    Smaller chunks improve retrieval precision by ensuring each chunk
    focuses on a specific topic rather than spanning multiple topics.

    Args:
        documents: List of LangChain Document objects to split.

    Returns:
        A list of smaller Document chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


# ---------------------------------------------------------------------------
# Vector Database Operations
# ---------------------------------------------------------------------------

def add_document_to_db(file_path: str) -> int:
    """
    Load a document, split it, and add it to the in-memory vector database.

    If no vector database exists yet, one is created. Otherwise, the new
    chunks are appended to the existing database.

    Args:
        file_path: Path to the document file to ingest.

    Returns:
        The number of chunks added to the database.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    global vectorstore, retriever

    documents = load_document(file_path)
    chunks = split_into_chunks(documents)

    if vectorstore is None:
        # Create a new in-memory vector store with the first document
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name="rag_session",
        )
    else:
        # Append new chunks to the existing store
        vectorstore.add_documents(chunks)

    # Refresh the retriever after every update
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RESULTS},
    )

    return len(chunks)


def clear_document_db() -> None:
    """
    Clear all documents from the in-memory vector database.

    Call this to reset the session (e.g., when the user clicks 'New Session').
    """
    global vectorstore, retriever
    vectorstore = None
    retriever = None


# ---------------------------------------------------------------------------
# Corrective RAG
# ---------------------------------------------------------------------------

def run_corrective_rag(
    query: str,
    history_text: str,
    retrieved_docs: list[Document],
    llm: ChatGroq,
) -> tuple[list[Document], str, str]:
    """
    Improve retrieval quality using a two-step correction process:

    Step 1 — Relevance Check:
        Ask the LLM whether the retrieved chunks actually answer the query.
        If YES, return immediately with the original results.

    Step 2 — Query Rewriting (only if Step 1 says NO):
        Ask the LLM to rewrite the query using conversation history,
        then retrieve again with the improved query.

    Args:
        query:          The user's original question.
        history_text:   Formatted conversation history as a plain string.
        retrieved_docs: Documents retrieved using the original query.
        llm:            The Groq LLM instance to use for correction.

    Returns:
        A tuple of (final_docs, final_context, final_query).
    """
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Step 1: Check if retrieved context is relevant to the question
    relevance_prompt = ChatPromptTemplate.from_template(
        "Context:\n{context}\n\n"
        "Question: {query}\n\n"
        "Does the context above contain information that helps answer the question? "
        "Reply with only 'YES' or 'NO'."
    )
    relevance_check = llm.invoke(
        relevance_prompt.format(context=context, query=query)
    ).content.strip().upper()

    # If relevant, return as-is
    if "YES" in relevance_check:
        return retrieved_docs, context, query

    # Step 2: Rewrite the query to improve retrieval
    rewrite_prompt = ChatPromptTemplate.from_template(
        "You are a search query optimizer.\n\n"
        "Rewrite the query below so it retrieves better results from a vector database.\n"
        "Use the conversation history for context.\n\n"
        "RULES:\n"
        "- Output ONLY the rewritten search query.\n"
        "- Do NOT answer the question.\n"
        "- Do NOT add extra commentary.\n\n"
        "Conversation History:\n{history}\n\n"
        "Original Query: {query}\n\n"
        "Rewritten Query:"
    )
    rewritten_query = llm.invoke(
        rewrite_prompt.format(history=history_text, query=query)
    ).content.strip()

    # Retrieve again using the improved query
    new_docs = retriever.invoke(rewritten_query)
    new_context = "\n\n".join([doc.page_content for doc in new_docs])

    return new_docs, new_context, rewritten_query


# ---------------------------------------------------------------------------
# Main Answer Function
# ---------------------------------------------------------------------------

def get_answer(
    query: str,
    chat_history: list[dict],
    model_name: str = "llama-3.1-8b-instant",
) -> tuple[str, list[str]]:
    """
    Generate an answer for the user's query using RAG.

    Flow:
      1. If the query is a simple greeting, return a friendly fixed response.
      2. If no documents are uploaded, answer from the LLM's general knowledge.
      3. If documents exist, retrieve relevant chunks and run Corrective RAG.
      4. Generate a final answer using the retrieved context + conversation history.
      5. Detect whether the answer came from documents or general knowledge,
         and return the appropriate source list.

    Args:
        query:        The user's latest message.
        chat_history: List of previous messages, each a dict with 'role' and 'content'.
        model_name:   Groq model ID to use for generation.

    Returns:
        A tuple of (answer_text, sources).
        - answer_text: The generated answer as a plain string.
        - sources: A list of document file paths, or ["General Knowledge"] if
                   the answer did not come from uploaded documents.
    """
    # Handle greetings directly — no need to hit the vector DB
    if query.lower().strip().rstrip("?!.") in GREETINGS:
        return (
            "Hello! I'm ready to help you with your documents. What would you like to know?",
            [],
        )

    # Format conversation history as plain text for the prompt
    history_text = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history]
    )

    # Initialize the LLM for this request
    llm = ChatGroq(
        model=model_name,
        temperature=0.7,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    # Determine context: from documents or fallback to general knowledge
    if retriever is None:
        final_docs = []
        context = "No documents have been uploaded. Answer using your general knowledge."
        final_query = query
    else:
        retrieved_docs = retriever.invoke(query)
        final_docs, context, final_query = run_corrective_rag(
            query=query,
            history_text=history_text,
            retrieved_docs=retrieved_docs,
            llm=llm,
        )

    # Build the final answer prompt
    answer_prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant.

SOURCING RULES:
1. If the answer is in the Context below, answer using ONLY that context.
2. If the answer is NOT in the context, use your general knowledge but start
   your response with the exact tag: [GENERAL_KNOWLEDGE]
3. Never cite a document source if the answer came from general knowledge.

Conversation History:
{history}

Context (from uploaded documents):
{context}

Question:
{question}
"""
    )

    raw_response = llm.invoke(
        answer_prompt.format(
            history=history_text,
            context=context,
            question=final_query,
        )
    ).content

    # Parse whether the answer is from documents or general knowledge
    if "[GENERAL_KNOWLEDGE]" in raw_response:
        answer = raw_response.replace("[GENERAL_KNOWLEDGE]", "").strip()
        sources = ["General Knowledge"]
    else:
        answer = raw_response
        sources = list({doc.metadata.get("source", "Unknown") for doc in final_docs})

    return answer, sources
