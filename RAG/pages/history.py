import reflex as rx
from RAG.components.navbar import navbar
from RAG.components.footer import footer
from RAG.states.rag_state import ChatState
from RAG.pages.chat import message_bubble

def index():
    return rx.vstack(
        navbar(),
        rx.center(
            rx.vstack(
                rx.hstack(
                    rx.heading("Session History", color="black", font_weight="bold"),
                    rx.spacer(),
                    rx.button("Clear Session", on_click=ChatState.reset_session, bg="red", color="white")
                ),
                rx.divider(),
                rx.box(
                    rx.foreach(ChatState.history, message_bubble),
                    height="70vh",
                    overflow="auto",
                    padding="20px",
                    width="100%"
                ),
                width="80%",
                bg="white",
                padding="20px",
                border_radius="10px",
                box_shadow="md",
                margin_top="20px"
            ),
            width="100%"
        ),
        footer(),
        min_height="100vh",
        spacing="0",
        background="#f5f5f5"
    )
