import streamlit as st
from utils.rag_pipeline import get_answer

st.title("📄 AI Document Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box
user_input = st.chat_input("Ask a question")

if user_input:
    # Store user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Temporary bot response
    response, sources = get_answer(user_input, st.session_state.messages)

    with st.chat_message("assistant"):
        st.markdown(response)

    st.markdown("**Sources:**")
    for src in sources:
        st.markdown(f"- {src}")

    st.session_state.messages.append({"role": "assistant", "content": response})