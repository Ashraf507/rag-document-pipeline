from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )

    return splitter.split_documents(documents)
