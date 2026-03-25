import reflex as rx

def navbar():
    return rx.hstack(
        rx.heading("🤖 RAG AI Assistant", color="white", size="6"),
        rx.spacer(),
        rx.hstack(
            rx.link("Home", href="/", color="white", font_weight="bold"),
            rx.link("Upload", href="/upload", color="white", font_weight="bold"),
            rx.link("Chat", href="/chat", color="white", font_weight="bold"),
            rx.link("History", href="/history", color="white", font_weight="bold"),
            spacing="4"
        ),
        bg="#1f1f1f",
        padding="15px",
        width="100%",
        align_items="center"
    )
