import reflex as rx
from RAG.components.navbar import navbar
from RAG.components.footer import footer
from RAG.states.rag_state import ChatState

def index():
    return rx.vstack(
        navbar(),
        rx.center(
            rx.vstack(
                rx.heading("Upload Documents", margin_bottom="20px", color="black", font_weight="bold"),
                rx.text("Upload your PDF and TXT files for the RAG AI to read.", margin_bottom="20px", color="black", font_weight="bold"),
                rx.upload(
                    rx.vstack(
                        rx.button("Select Files", bg="black", color="white", border_radius="10px"),
                        rx.text("Drag and drop files here or click to select files", color="black", font_weight="bold"),
                    ),
                    id="doc_upload",
                    multiple=True,
                    padding="2em",
                    border="2px dashed #ccc",
                    border_radius="10px",
                    on_drop=ChatState.handle_upload(rx.upload_files(upload_id="doc_upload")),
                ),
                rx.text("Uploaded Files:", font_weight="bold", margin_top="20px", color="black"),
                rx.foreach(ChatState.uploaded_files, lambda file: rx.text(file, color="black", font_weight="bold")),
                width="50%",
                padding="20px",
                box_shadow="lg",
                border_radius="lg",
                bg="white"
            ),
            width="100%",
            height="80vh",
        ),
        footer(),
        min_height="100vh",
        spacing="0",
        background="#f5f5f5"
    )
