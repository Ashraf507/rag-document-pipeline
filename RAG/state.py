import reflex as rx
from typing import List, Dict
from RAG.utils.rag_pipeline import get_answer

class ChatState(rx.State):
    messages: List[Dict[str, str]] = []

    def ask(self, question: str):
        if not question:
            return

        # Add user message
        self.messages.append({"role": "user", "content": question})

        # Show loading
        self.messages.append({"role": "assistant", "content": "Thinking..."})

        # Get answer
        answer, sources = get_answer(question, self.messages)

        response = answer + "\n\nSources:\n" + "\n".join(sources)

        # Replace loading message
        self.messages[-1] = {"role": "assistant", "content": response}