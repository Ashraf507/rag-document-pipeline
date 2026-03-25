import reflex as rx
from RAG.pages import home, upload, chat, history

# Add upload folder configuration 
app = rx.App()

# Register routes
app.add_page(home.index, route="/")
app.add_page(upload.index, route="/upload")
app.add_page(chat.index, route="/chat")
app.add_page(history.index, route="/history")