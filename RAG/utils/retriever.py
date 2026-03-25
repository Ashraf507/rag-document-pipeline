def get_retriever(vectordb):
    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    return retriever