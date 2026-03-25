import os
import reflex as rx
from typing import List, Dict
from RAG.backend.rag import get_answer, add_documents_to_db, clear_db

class ChatState(rx.State):
    question: str = ""
    history: List[Dict[str, str]] = []
    uploaded_files: List[str] = []

    def set_question(self, question: str):
        self.question = question

    async def handle_upload(self, files: List[rx.UploadFile]):
        upload_dir = rx.get_upload_dir()
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)

        new_files = []
        for file in files:
            upload_data = await file.read()
            outfile_path = os.path.join(upload_dir, file.filename)
            with open(outfile_path, "wb") as file_object:
                file_object.write(upload_data)
            
            # process document logic
            add_documents_to_db(outfile_path)
            new_files.append(file.filename)
            
        self.uploaded_files = self.uploaded_files + new_files
        return rx.redirect("/chat")

    def ask(self, form_data: dict = None):
        if not self.question:
            return

        q = self.question
        # Add user message correctly by creating a new list
        self.history = self.history + [{"role": "user", "content": q}]
        self.question = ""
        yield  # Yield to update UI immediately

        self.history = self.history + [{"role": "assistant", "content": "Thinking..."}]
        yield

        # Get answer
        # The prompt says: "We modified corrective_rag(query, history). Now LLM sees Past Q/A. Final prompt: History + Context + Question"
        answer, sources = get_answer(q, self.history[:-2]) # passing without the "Thinking..." and the new user message (wait, user message SHOULD be passed, so `[:-1]`)

        response = answer
        if sources:
            response += "\n\n**Sources:**\n" + "\n".join([f"- {s}" for s in sources])

        # Replace loading message by replacing the last item via new list slicing
        self.history = self.history[:-1] + [{"role": "assistant", "content": response}]

    def reset_session(self):
        self.history = []
        self.uploaded_files = []
        clear_db()
