import reflex as rx

def hero():
    return rx.vstack(
        rx.heading("Welcome to RAG AI Assistant", font_size="2.5em", margin_bottom="20px", color="black", font_weight="bold"),
        rx.text("Upload your documents and chat intelligently.", font_size="1.2em", margin_bottom="30px", color="black", font_weight="bold"),
        rx.hstack(
            rx.link(
                rx.button("Get Started", bg="#007bff", color="white", padding="15px", border_radius="10px"),
                href="/upload"
            ),
            rx.link(
                rx.button("Start Chatting", bg="#28a745", color="white", padding="15px", border_radius="10px"),
                href="/chat"
            )
        ),
        align_items="center",
        justify_content="center",
        height="60vh",
        background="linear-gradient(to right bottom, #ebf8ff, #ffffff)",
        width="100%"
    )
