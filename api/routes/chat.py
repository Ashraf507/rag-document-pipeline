"""
api/routes/chat.py — Chat endpoints.

Endpoints:
  POST   /chat/ask    — Send a question and get an AI-generated answer.
  DELETE /chat/reset  — Clear the current session (chat history + documents).
"""

import logging

from fastapi import APIRouter, HTTPException

from RAG.backend.rag import clear_document_db, get_answer
from api.schemas.chat import ChatRequest, ChatResponse, ResetResponse

logger = logging.getLogger(__name__)

# Create a router — this groups all /chat routes together.
# The prefix "/chat" is applied in main.py when we include this router.
router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post(
    "/ask",
    response_model=ChatResponse,
    summary="Ask a question",
    description=(
        "Send a question to the RAG pipeline. "
        "The system will search uploaded documents for relevant context, "
        "then use the selected Groq model to generate an answer. "
        "If no documents are uploaded, the model answers from general knowledge."
    ),
)
def ask_question(request: ChatRequest) -> ChatResponse:
    """
    Handle a user question and return an AI-generated answer.

    The request body must include:
      - question:  The user's question (required).
      - history:   Previous messages in the conversation (optional).
      - model:     Which Groq model to use (optional, defaults to Llama 8B).
    """
    logger.info(f"Received question: '{request.question}' using model '{request.model}'")

    # Convert Pydantic Message objects to plain dicts for the RAG engine
    history_dicts = [msg.model_dump() for msg in request.history]

    try:
        answer, sources = get_answer(
            query=request.question,
            chat_history=history_dicts,
            model_name=request.model,
        )
    except Exception as e:
        logger.error(f"RAG pipeline error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate an answer. Reason: {str(e)}",
        )

    logger.info(f"Answer generated successfully. Sources: {sources}")

    return ChatResponse(
        answer=answer,
        sources=sources,
        model_used=request.model,
    )


@router.delete(
    "/reset",
    response_model=ResetResponse,
    summary="Reset session",
    description=(
        "Clear all uploaded documents from the vector database and reset the session. "
        "Note: This does not clear the client's chat history — "
        "the client is responsible for clearing message history on their side."
    ),
)
def reset_session() -> ResetResponse:
    """
    Reset the current RAG session by clearing the vector database.
    """
    logger.info("Resetting session — clearing vector database.")

    try:
        clear_document_db()
    except Exception as e:
        logger.error(f"Failed to reset session: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reset session. Reason: {str(e)}",
        )

    return ResetResponse(message="Session reset successfully. All documents have been cleared.")
