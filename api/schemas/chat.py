"""
api/schemas/chat.py — Pydantic models for the /chat endpoints.

Pydantic models define the exact shape of data that the API accepts
and returns. FastAPI uses these to automatically validate requests
and generate interactive API docs (Swagger UI).
"""

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single message in the conversation history."""

    role: str = Field(
        ...,
        description="Who sent this message. Must be 'user' or 'assistant'.",
        examples=["user"],
    )
    content: str = Field(
        ...,
        description="The text content of the message.",
        examples=["What is retrieval-augmented generation?"],
    )


class ChatRequest(BaseModel):
    """Request body for POST /chat/ask."""

    question: str = Field(
        ...,
        min_length=1,
        description="The user's question.",
        examples=["Summarize the uploaded document."],
    )
    history: list[Message] = Field(
        default=[],
        description="All previous messages in the conversation, in order.",
    )
    model: str = Field(
        default="llama-3.1-8b-instant",
        description=(
            "The Groq model ID to use for this request. "
            "Options: 'llama-3.1-8b-instant', 'llama-3.3-70b-versatile', 'qwen/qwen3-32b'."
        ),
        examples=["llama-3.1-8b-instant"],
    )


class ChatResponse(BaseModel):
    """Response body for POST /chat/ask."""

    answer: str = Field(
        ...,
        description="The generated answer from the RAG pipeline.",
    )
    sources: list[str] = Field(
        ...,
        description=(
            "Where the answer came from. "
            "Contains document file paths, or ['General Knowledge'] "
            "if no relevant document was found."
        ),
    )
    model_used: str = Field(
        ...,
        description="The Groq model that generated the answer.",
    )


class ResetResponse(BaseModel):
    """Response body for DELETE /chat/reset."""

    message: str = Field(
        default="Session reset successfully.",
        description="Confirmation message.",
    )
