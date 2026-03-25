from dotenv import load_dotenv
from utils.loaders import load_documents
from utils.loaders import load_documents, clean_metadata
from utils.embeddings import get_embeddings
from utils.vectordb import create_vector_store
from utils.splitter import split_documents
from utils.retriever import get_retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

import os
import shutil

# Load environment variables
load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

print("LLM connected successfully")

# Read API key
api_key = os.getenv("GROQ_API_KEY")

print("API Key loaded:", api_key[:10], "...")

# Startup cleanup function
def startup_cleanup():
    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db")
        print("Old Chroma DB deleted")
    else:
        print("No previous DB found")

# Run cleanup
startup_cleanup()

documents = load_documents()
print("Loaded documents:", len(documents))

documents = clean_metadata(documents)
print("Metadata cleaned")

chunks = split_documents(documents)
print("Number of chunks:", len(chunks))

embeddings = get_embeddings()
print("Embedding model loaded")

vectordb = create_vector_store(chunks, embeddings)
print("Vector database created successfully")

vectordb = create_vector_store(chunks, embeddings)
retriever = get_retriever(vectordb)
print("Retriever created successfully")

retriever = get_retriever(vectordb)

print("RAG system ready")

query = input("Enter your question: ")

# Step 1: Retrieve relevant chunks
retrieved_docs = retriever.invoke(query)

# Step 2: Combine context
context = "\n\n".join([doc.page_content for doc in retrieved_docs])

# Step 3: Create prompt template
prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant.
    Use ONLY the following context to answer the question.
    If the answer is not in the context, say "I don't know."

    Context:
    {context}

    Question:
    {question}
    """
)

# Step 4: Format prompt
formatted_prompt = prompt.format(
    context=context,
    question=query
)

# Step 5: Send to LLM
response = llm.invoke(formatted_prompt)

print("\nAnswer:\n")
print(response.content)

print("\nSources:\n")

unique_sources = set()

for idx, doc in enumerate(retrieved_docs, start=1):
    source = doc.metadata.get("source")
    if source not in unique_sources:
        print(f"[{idx}] {source}")
        unique_sources.add(source)
