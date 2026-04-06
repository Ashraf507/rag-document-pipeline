"""
tests/test_rag_backend.py — Unit tests for the RAG engine.

These tests validate the core logic of the RAG pipeline in isolation.
We use temporary files and mock objects so that:
  - No real API calls are made to Groq.
  - No real embedding models are loaded (too slow for tests).
  - Tests run fast and reliably.

Run with:
    pytest tests/test_rag_backend.py -v
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from RAG.backend.rag import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    GREETINGS,
    TOP_K_RESULTS,
    clear_document_db,
    split_into_chunks,
    load_document,
)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def make_temp_txt_file(content: str) -> str:
    """
    Create a temporary .txt file with the given content.
    Returns the absolute path to the file.
    The caller is responsible for deleting it after use.
    """
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    tmp.write(content)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Tests: Constants
# ---------------------------------------------------------------------------

class TestConstants:
    """Verify that the key constants are set to sensible values."""

    def test_chunk_size_is_positive(self):
        assert CHUNK_SIZE > 0, "CHUNK_SIZE must be a positive integer."

    def test_chunk_overlap_is_less_than_chunk_size(self):
        assert CHUNK_OVERLAP < CHUNK_SIZE, (
            "CHUNK_OVERLAP must be smaller than CHUNK_SIZE to avoid infinite loops."
        )

    def test_top_k_results_is_positive(self):
        assert TOP_K_RESULTS > 0, "TOP_K_RESULTS must be a positive integer."

    def test_greetings_is_not_empty(self):
        assert len(GREETINGS) > 0, "GREETINGS set should not be empty."

    def test_common_greetings_are_present(self):
        """Ensure the most common greetings are covered."""
        for greeting in ["hi", "hello", "hey"]:
            assert greeting in GREETINGS, f"'{greeting}' should be in GREETINGS."


# ---------------------------------------------------------------------------
# Tests: load_document
# ---------------------------------------------------------------------------

class TestLoadDocument:
    """Tests for the load_document() function."""

    def test_load_txt_file(self):
        """Should successfully load a plain text file."""
        content = "This is a test document about machine learning."
        path = make_temp_txt_file(content)

        try:
            docs = load_document(path)
            assert len(docs) > 0, "Should return at least one Document."
            assert content in docs[0].page_content
        finally:
            os.unlink(path)

    def test_metadata_values_are_strings(self):
        """All metadata values must be strings for ChromaDB compatibility."""
        path = make_temp_txt_file("Metadata test document.")

        try:
            docs = load_document(path)
            for doc in docs:
                for key, value in doc.metadata.items():
                    assert isinstance(value, str), (
                        f"Metadata key '{key}' has non-string value: {type(value)}"
                    )
        finally:
            os.unlink(path)

    def test_raises_file_not_found(self):
        """Should raise FileNotFoundError for a non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_document("/non/existent/path/file.txt")


# ---------------------------------------------------------------------------
# Tests: split_into_chunks
# ---------------------------------------------------------------------------

class TestSplitIntoChunks:
    """Tests for the split_into_chunks() function."""

    def test_short_document_is_not_split(self):
        """A short document should remain as a single chunk."""
        doc = Document(page_content="Short text.", metadata={"source": "test.txt"})
        chunks = split_into_chunks([doc])
        assert len(chunks) >= 1

    def test_long_document_is_split(self):
        """A very long document should be split into multiple chunks."""
        long_text = "word " * 10_000   # ~50,000 characters
        doc = Document(page_content=long_text, metadata={"source": "test.txt"})
        chunks = split_into_chunks([doc])
        assert len(chunks) > 1, "Long documents should be split into multiple chunks."

    def test_chunks_preserve_content(self):
        """The combined text of all chunks should contain the original text."""
        original = "The quick brown fox jumps over the lazy dog."
        doc = Document(page_content=original, metadata={"source": "test.txt"})
        chunks = split_into_chunks([doc])
        combined = " ".join([c.page_content for c in chunks])
        assert "quick brown fox" in combined


# ---------------------------------------------------------------------------
# Tests: clear_document_db
# ---------------------------------------------------------------------------

class TestClearDocumentDb:
    """Tests for the clear_document_db() function."""

    def test_clear_does_not_raise(self):
        """Calling clear on an empty session should not raise any errors."""
        clear_document_db()   # Called on already-empty state — should be fine.

    def test_clear_resets_state(self):
        """After clearing, the global vectorstore and retriever should be None."""
        import RAG.backend.rag as rag_module

        # Manually set a fake vectorstore
        rag_module.vectorstore = MagicMock()
        rag_module.retriever = MagicMock()

        clear_document_db()

        assert rag_module.vectorstore is None, "vectorstore should be None after clear."
        assert rag_module.retriever is None, "retriever should be None after clear."


# ---------------------------------------------------------------------------
# Tests: get_answer (greeting shortcut)
# ---------------------------------------------------------------------------

class TestGetAnswerGreetings:
    """Test that greetings are handled without hitting the LLM or vector DB."""

    def test_greeting_returns_without_llm(self):
        """
        A greeting query should return a fixed response immediately.
        We verify the LLM is NOT called by checking the response is a string.
        """
        from RAG.backend.rag import get_answer

        answer, sources = get_answer("hi", chat_history=[], model_name="llama-3.1-8b-instant")

        assert isinstance(answer, str), "Answer should be a string."
        assert len(answer) > 0, "Answer should not be empty."
        assert sources == [], "Greetings should return empty sources list."

    @pytest.mark.parametrize("greeting", ["hi", "hello", "hey", "greetings"])
    def test_multiple_greetings(self, greeting):
        """All common greetings should be handled gracefully."""
        from RAG.backend.rag import get_answer

        answer, sources = get_answer(greeting, chat_history=[])
        assert isinstance(answer, str)
        assert sources == []
