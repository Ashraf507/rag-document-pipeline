import os
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Global memory vectorstore
vectorstore = None
retriever = None

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    return splitter.split_documents(documents)

def add_documents_to_db(file_path):
    global vectorstore, retriever
    
    if file_path.endswith('.pdf'):
        loader = PyMuPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
        
    docs = loader.load()
    for doc in docs:
        doc.metadata = {key: str(value) for key, value in doc.metadata.items()}
        
    chunks = split_documents(docs)
    
    if vectorstore is None:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name="rag_session"
        )
    else:
        vectorstore.add_documents(chunks)
        
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

def clear_db():
    global vectorstore, retriever
    vectorstore = None
    retriever = None

def corrective_rag(query, history_text, retrieved_docs):
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # 1. Relevance check (YES/NO) - checks if retrieved context can answer query
    relevance_prompt = ChatPromptTemplate.from_template(
        "Does the following context contain information relevant to answering the query? "
        "Answer strictly 'YES' or 'NO'.\n\nContext: {context}\n\nQuery: {query}"
    )
    
    relevance_response = llm.invoke(
        relevance_prompt.format(context=context, query=query)
    ).content.strip().upper()
    
    if "YES" in relevance_response:
        return retrieved_docs, context, query
        
    # 2. Query rewriting if NO
    rewrite_prompt = ChatPromptTemplate.from_template(
        "Rewrite the following query to be better optimized for a search engine, using the conversation history for context.\n\n"
        "History:\n{history}\n\n"
        "Original Query: {query}\n\n"
        "Just provide the rewritten query."
    )
    new_query = llm.invoke(rewrite_prompt.format(history=history_text, query=query)).content.strip()
    
    # Fetch again with rewritten query
    new_docs = retriever.invoke(new_query)
    new_context = "\n\n".join([doc.page_content for doc in new_docs])
    
    return new_docs, new_context, new_query

def get_answer(query, chat_history):
    global retriever
    if retriever is None:
        return "Please upload documents first.", []
        
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
        
    retrieved_docs = retriever.invoke(query)
    
    # Corrective RAG
    final_docs, context, final_query = corrective_rag(query, history_text, retrieved_docs)
    
    # Generation
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
        question=final_query
    )
    
    response = llm.invoke(formatted_prompt)
    sources = list(set([doc.metadata.get("source") for doc in final_docs]))
    
    return response.content, sources
