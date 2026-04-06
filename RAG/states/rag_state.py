import os
import reflex as rx
from typing import List, Dict
from RAG.backend.rag import get_answer, add_document_to_db, clear_document_db

# Descriptive labels for the UI selection
MODEL_OPTIONS = [
    "⚡ Fast (Llama 8B) - For basic chat & quick summaries",
    "🧠 Smart (Llama 70B) - For deep reasoning & research",
    "🧬 Advanced (Qwen 32B) - For complex technical tasks"
]

class ChatState(rx.State):
    question: str = ""
    history: List[Dict[str, str]] = []
    uploaded_files: List[str] = []
    
    # Mapping labels to internal model IDs
    model_mapping: Dict[str, str] = {
        MODEL_OPTIONS[0]: "llama-3.1-8b-instant",
        MODEL_OPTIONS[1]: "llama-3.3-70b-versatile",
        MODEL_OPTIONS[2]: "qwen/qwen3-32b"
    }
    selected_model_label: str = MODEL_OPTIONS[0]
    show_upload_dialog: bool = False

    @rx.var
    def selected_model(self) -> str:
        return self.model_mapping.get(self.selected_model_label, "llama-3.1-8b-instant")

    def set_question(self, question: str):
        self.question = question

    def toggle_upload_dialog(self):
        self.show_upload_dialog = not self.show_upload_dialog

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
            add_document_to_db(outfile_path)
            new_files.append(file.filename)
            
        self.uploaded_files = self.uploaded_files + new_files
        self.show_upload_dialog = True

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

        # Get answer with selected model
        answer, sources = get_answer(q, self.history[:-2], model_name=self.selected_model)

        response = answer
        if sources:
            response += "\n\n**Sources:**\n" + "\n".join([f"- {s}" for s in sources])

        # Replace loading message by replacing the last item via new list slicing
        self.history = self.history[:-1] + [{"role": "assistant", "content": response}]

    def reset_session(self):
        self.history = []
        self.uploaded_files = []
        clear_document_db()
