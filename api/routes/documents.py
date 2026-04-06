"""
api/routes/documents.py — Document management endpoints.

Endpoints:
  POST  /documents/upload  — Upload a file and index it into the vector database.
  GET   /documents/list    — List all files currently loaded in this session.
  DELETE /documents/clear  — Remove all documents from the session.
"""

import logging
import os
import shutil

from fastapi import APIRouter, HTTPException, UploadFile, File

from RAG.backend.rag import add_document_to_db, clear_document_db
from api.schemas.documents import DocumentListResponse, UploadResponse

logger = logging.getLogger(__name__)

# Folder where uploaded files are saved temporarily during the session
UPLOAD_DIR = "uploaded_files"

# File extensions that the RAG engine can process
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".txt"}

# In-memory list of uploaded filenames for the current session
session_files: list[str] = []

router = APIRouter(prefix="/documents", tags=["Documents"])


def _validate_extension(filename: str) -> None:
    """
    Raise an HTTP 400 error if the file extension is not supported.

    Args:
        filename: The original filename from the upload.

    Raises:
        HTTPException: 400 Bad Request if the extension is not in ALLOWED_EXTENSIONS.
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"File type '{ext}' is not supported. "
                f"Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )


def _save_file(upload: UploadFile) -> str:
    """
    Save an uploaded file to the UPLOAD_DIR folder on disk.

    Args:
        upload: The FastAPI UploadFile object.

    Returns:
        The absolute path where the file was saved.
    """
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    save_path = os.path.join(UPLOAD_DIR, upload.filename)

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)

    return os.path.abspath(save_path)


@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload a document",
    description=(
        "Upload a document file (PDF, DOCX, PPTX, or TXT). "
        "The file is saved to disk, split into text chunks, "
        "and indexed into the ChromaDB vector database for retrieval."
    ),
)
def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """
    Accept a file upload, validate it, save it, and index it into the vector DB.

    Args:
        file: The uploaded file from the multipart/form-data request.

    Returns:
        UploadResponse with the filename and number of chunks added.
    """
    logger.info(f"Uploading file: '{file.filename}'")

    # Reject unsupported file types early
    _validate_extension(file.filename)

    # Save the file to disk
    try:
        file_path = _save_file(file)
    except Exception as e:
        logger.error(f"Failed to save file '{file.filename}': {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Could not save the file. Reason: {str(e)}",
        )

    # Index the file into the vector database
    try:
        chunks_added = add_document_to_db(file_path)
    except Exception as e:
        logger.error(f"Failed to index file '{file.filename}': {e}")
        raise HTTPException(
            status_code=500,
            detail=f"File was saved but could not be indexed. Reason: {str(e)}",
        )

    # Track the file in the current session
    session_files.append(file.filename)

    logger.info(f"'{file.filename}' indexed successfully — {chunks_added} chunks added.")

    return UploadResponse(
        filename=file.filename,
        chunks_added=chunks_added,
        message="File uploaded and indexed successfully.",
    )


@router.get(
    "/list",
    response_model=DocumentListResponse,
    summary="List uploaded documents",
    description="Returns the names of all documents currently loaded in this session.",
)
def list_documents() -> DocumentListResponse:
    """
    Return the list of files that have been uploaded in the current session.
    """
    return DocumentListResponse(
        uploaded_files=session_files,
        total_files=len(session_files),
    )


@router.delete(
    "/clear",
    response_model=dict,
    summary="Clear all documents",
    description=(
        "Remove all documents from the vector database and reset the document session. "
        "This is separate from /chat/reset — use this when you only want to "
        "clear documents but keep the chat history."
    ),
)
def clear_documents() -> dict:
    """
    Clear all documents from the vector database and the session file list.
    """
    global session_files
    logger.info("Clearing all documents from the session.")

    try:
        clear_document_db()
        session_files = []
    except Exception as e:
        logger.error(f"Failed to clear documents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Could not clear documents. Reason: {str(e)}",
        )

    return {"message": "All documents have been cleared from the session."}
