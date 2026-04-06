"""
api/schemas/documents.py — Pydantic models for the /documents endpoints.
"""

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """Response body for POST /documents/upload."""

    filename: str = Field(
        ...,
        description="The name of the file that was uploaded.",
        examples=["annual_report.pdf"],
    )
    chunks_added: int = Field(
        ...,
        description="Number of text chunks extracted and stored in the vector database.",
        examples=[42],
    )
    message: str = Field(
        ...,
        description="A human-readable status message.",
        examples=["File uploaded and indexed successfully."],
    )


class DocumentListResponse(BaseModel):
    """Response body for GET /documents/list."""

    uploaded_files: list[str] = Field(
        ...,
        description="Names of all files currently loaded in the active session.",
    )
    total_files: int = Field(
        ...,
        description="Total number of files currently in the session.",
    )
