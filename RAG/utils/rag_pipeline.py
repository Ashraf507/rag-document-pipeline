try:
    from RAG.utils.loaders import load_documents, clean_metadata
    from RAG.utils.splitter import split_documents
    from RAG.utils.embeddings import get_embeddings
    from RAG.utils.vectordb import create_vector_store
    from RAG.utils.retriever import get_retriever
except ImportError:
    from utils.loaders import load_documents, clean_metadata
    from utils.splitter import split_documents
    from utils.embeddings import get_embeddings
    from utils.vectordb import create_vector_store
    from utils.retriever import get_retriever

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

# Initialize RAG
def initialize_rag():
    documents = load_documents()
    documents = clean_metadata(documents)
    chunks = split_documents(documents)
    embeddings = get_embeddings()
    vectordb = create_vector_store(chunks, embeddings)
    retriever = get_retriever(vectordb)
    return retriever


# Lazy loading (important)
retriever = None

def get_retriever_instance():
    global retriever
    if retriever is None:
        retriever = initialize_rag()
    return retriever


# Main function
def get_answer(query, chat_history):

    history_text = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in chat_history]
    )

    # Retrieve docs
    retrieved_docs = get_retriever_instance().invoke(query)

    # Context
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Prompt
    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful assistant.

        Use the conversation history and context to answer.

        If answer not found, say "Not available in documents."

        Conversation History:
        {history}

        Context:
        {context}

        Question:
        {question}
        """
    )

    formatted_prompt = prompt.format(
        history=history_text,
        context=context,
        question=query
    )

    # LLM response
    response = llm.invoke(formatted_prompt)

    # Sources
    sources = list(set([
        doc.metadata.get("source") for doc in retrieved_docs
    ]))

    return response.content, sources