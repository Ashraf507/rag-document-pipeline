import reflex as rx
from RAG.components.navbar import navbar
from RAG.states.rag_state import ChatState

def message_bubble(msg):
    is_user = (msg["role"] == "user")
    return rx.hstack(
        rx.cond(is_user, rx.spacer(), rx.fragment()),
        rx.box(
            rx.markdown(msg["content"]),
            bg=rx.cond(is_user, "#007bff", "#e5e5ea"),
            color=rx.cond(is_user, "white", "black"),
            padding="15px",
            border_radius="15px",
            max_width="70%",
        ),
        rx.cond(~is_user, rx.spacer(), rx.fragment()),
        width="100%",
        padding_y="2",
    )

def index():
    return rx.vstack(
        navbar(),
        
        # Chat area
        rx.box(
            rx.foreach(ChatState.history, message_bubble),
            height="75vh",
            overflow="auto",
            padding="20px",
            width="80%",
            bg="white",
            border="2px solid black",
            border_radius="10px",
            box_shadow="md",
            margin_top="20px"
        ),
        
        # Input area
        rx.form.root(
            rx.hstack(
                rx.input(
                    placeholder="Ask a question about the uploaded documents...",
                    value=ChatState.question,
                    on_change=ChatState.set_question,
                    width="80%",
                    size="3",
                    style={
                        "height": "100%",
                        "borderRadius": "10px",
                        "border": "2px solid black",
                        "backgroundColor": "white",
                        "--text-field-selection-color": "black",
                        "& input": {
                            "color": "black !important",
                            "fontWeight": "bold !important",
                            "WebkitTextFillColor": "black !important",
                        },
                        "& input::placeholder": {
                            "color": "#4b5563 !important",
                        }
                    }
                ),
                rx.button(
                    "Send",
                    type="submit",
                    bg="#007bff",
                    color="white",
                    padding="15px 30px",
                    border_radius="10px",
                    font_weight="bold"
                ),
                width="80%",
                padding="10px",
            ),
            on_submit=ChatState.ask,
            width="100%",
            display="flex",
            justify_content="center",
        ),
        
        align_items="center",
        width="100%",
        height="100vh",
        background="#f5f5f5",
        spacing="0"
    )
