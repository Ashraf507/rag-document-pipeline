from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyMuPDFLoader

def load_documents():
    # Load PDFs using PyMuPDF (no OCR dependency)
    pdf_loader = DirectoryLoader(
        "data",
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader
    )

    # Load text files
    text_loader = DirectoryLoader(
        "data",
        glob="**/*.txt",
        loader_cls=TextLoader
    )

    pdf_docs = pdf_loader.load()
    text_docs = text_loader.load()

    return pdf_docs + text_docs

def clean_metadata(documents):
    for doc in documents:
        doc.metadata = {
            key: str(value)
            for key, value in doc.metadata.items()
        }
    return documents

