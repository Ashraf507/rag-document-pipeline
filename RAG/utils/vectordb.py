from langchain_community.vectorstores import Chroma
import uuid

def create_vector_store(documents, embeddings):
    collection_name = f"collection_{uuid.uuid4()}"

    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_db",
        collection_name=collection_name
    )

    

    return vectordb
