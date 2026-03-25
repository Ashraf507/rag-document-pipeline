import reflex as rx

def footer():
    return rx.center(
        rx.text("© 2026 RAG Chatbot Application. Built with Reflex + LangChain. All rights reserved.", color="black", font_weight="bold"),
        padding="20px",
        width="100%",
        border_top="1px solid #eaeaea",
        margin_top="auto"
    )
