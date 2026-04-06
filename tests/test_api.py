"""
tests/test_api.py — Integration tests for the FastAPI endpoints.

These tests spin up the FastAPI app using TestClient (no real server needed)
and send real HTTP requests to verify that:
  - Endpoints return the correct HTTP status codes.
  - Response bodies match the expected Pydantic schemas.
  - Validation errors are returned for bad input.

We mock the RAG engine functions so tests don't make real LLM or embedding calls.

Run with:
    pytest tests/test_api.py -v
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.main import app

# TestClient lets us call the FastAPI app like a real HTTP server in tests.
client = TestClient(app)


# ---------------------------------------------------------------------------
# Tests: Health check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    """Test the root health check endpoint."""

    def test_root_returns_200(self):
        """GET / should return HTTP 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_returns_status_running(self):
        """GET / should return {'status': 'running', ...}."""
        response = client.get("/")
        data = response.json()
        assert data["status"] == "running"
        assert "message" in data


# ---------------------------------------------------------------------------
# Tests: POST /chat/ask
# ---------------------------------------------------------------------------

class TestChatAsk:
    """Integration tests for the POST /chat/ask endpoint."""

    @patch("api.routes.chat.get_answer", return_value=("This is the answer.", ["doc.pdf"]))
    def test_ask_returns_200(self, mock_get_answer):
        """A valid question should return HTTP 200."""
        response = client.post("/chat/ask", json={
            "question": "What is RAG?",
            "history": [],
            "model": "llama-3.1-8b-instant",
        })
        assert response.status_code == 200

    @patch("api.routes.chat.get_answer", return_value=("Answer here.", ["report.pdf"]))
    def test_ask_response_structure(self, mock_get_answer):
        """Response should contain 'answer', 'sources', and 'model_used' fields."""
        response = client.post("/chat/ask", json={
            "question": "Summarize the document.",
            "history": [],
        })
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "model_used" in data

    @patch("api.routes.chat.get_answer", return_value=("Answer.", ["file.pdf"]))
    def test_ask_with_conversation_history(self, mock_get_answer):
        """Sending a previous conversation history should work correctly."""
        response = client.post("/chat/ask", json={
            "question": "What did we discuss earlier?",
            "history": [
                {"role": "user", "content": "Tell me about neural networks."},
                {"role": "assistant", "content": "Neural networks are..."},
            ],
            "model": "llama-3.3-70b-versatile",
        })
        assert response.status_code == 200

    def test_ask_empty_question_returns_422(self):
        """An empty question string should return HTTP 422 Unprocessable Entity."""
        response = client.post("/chat/ask", json={
            "question": "",  # Violates min_length=1
        })
        assert response.status_code == 422

    def test_ask_missing_question_returns_422(self):
        """A request with no 'question' field should return HTTP 422."""
        response = client.post("/chat/ask", json={})
        assert response.status_code == 422

    @patch("api.routes.chat.get_answer", side_effect=Exception("LLM unavailable"))
    def test_ask_handles_llm_error_gracefully(self, mock_get_answer):
        """If the RAG engine raises an exception, return HTTP 500."""
        response = client.post("/chat/ask", json={"question": "Test?"})
        assert response.status_code == 500
        assert "detail" in response.json()


# ---------------------------------------------------------------------------
# Tests: DELETE /chat/reset
# ---------------------------------------------------------------------------

class TestChatReset:
    """Integration tests for the DELETE /chat/reset endpoint."""

    @patch("api.routes.chat.clear_document_db")
    def test_reset_returns_200(self, mock_clear):
        """DELETE /chat/reset should return HTTP 200."""
        response = client.delete("/chat/reset")
        assert response.status_code == 200

    @patch("api.routes.chat.clear_document_db")
    def test_reset_response_has_message(self, mock_clear):
        """Reset response should contain a 'message' field."""
        response = client.delete("/chat/reset")
        data = response.json()
        assert "message" in data


# ---------------------------------------------------------------------------
# Tests: GET /documents/list
# ---------------------------------------------------------------------------

class TestDocumentList:
    """Integration tests for the GET /documents/list endpoint."""

    def test_list_returns_200(self):
        """GET /documents/list should return HTTP 200."""
        response = client.get("/documents/list")
        assert response.status_code == 200

    def test_list_response_structure(self):
        """Response should have 'uploaded_files' and 'total_files' fields."""
        response = client.get("/documents/list")
        data = response.json()
        assert "uploaded_files" in data
        assert "total_files" in data
        assert isinstance(data["uploaded_files"], list)
        assert isinstance(data["total_files"], int)


# ---------------------------------------------------------------------------
# Tests: POST /documents/upload
# ---------------------------------------------------------------------------

class TestDocumentUpload:
    """Integration tests for the POST /documents/upload endpoint."""

    def test_upload_unsupported_file_type_returns_400(self):
        """Uploading a .exe file should return HTTP 400 Bad Request."""
        response = client.post(
            "/documents/upload", 
            files={"file": ("malware.exe", b"fake binary content", "application/octet-stream")}
        )
        assert response.status_code == 400
        assert "not supported" in response.json()["detail"]

    @patch("api.routes.documents.add_document_to_db", return_value=15)
    @patch("api.routes.documents._save_file", return_value="/tmp/test.txt")
    def test_upload_valid_txt_returns_200(self, mock_save, mock_add):
        """Uploading a valid .txt file should return HTTP 200."""
        # Fix: 'files' value should be (filename, content, content_type)
        response = client.post(
            "/documents/upload", 
            files={"file": ("notes.txt", b"Some document content.", "text/plain")}
        )
        assert response.status_code == 200

    @patch("api.routes.documents.add_document_to_db", return_value=10)
    @patch("api.routes.documents._save_file", return_value="/tmp/test.pdf")
    def test_upload_response_contains_chunk_count(self, mock_save, mock_add):
        """Upload response should include 'chunks_added' count."""
        response = client.post(
            "/documents/upload", 
            files={"file": ("report.pdf", b"%PDF fake content", "application/pdf")}
        )
        data = response.json()
        assert "chunks_added" in data
        assert data["chunks_added"] == 10
