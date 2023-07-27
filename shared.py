import os
from typing import List
from dotenv import load_dotenv
from langchain.document_loaders import (
    CSVLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
)
from chromadb.config import Settings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
# Define the folder for storing database
PERSIST_DIR = os.environ.get('PERSIST_DIRECTORY')
FILE_DIR = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
EMBED_MODEL_NAME = os.environ.get('EMBEDDINGS_MODEL_NAME')
CHUNK_SIZE = os.environ.get('CHUNK_SIZE', 500)
CHUNK_OVERLAP = os.environ.get('CHUNK_OVERLAP', 50)
CHROMA_SETTINGS = Settings(
    persist_directory=PERSIST_DIR,
    anonymized_telemetry=False,
    is_persistent=True
)
gr = {}

db = {
    'db': None,
    'files': set()
}


# Map file extensions to document loaders and their arguments
FILE_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}


def load_file_and_split_chunks(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in FILE_MAPPING:
        loader_class, loader_args = FILE_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        docs = loader.load()
        if file_path not in db['files']:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            texts = text_splitter.split_documents(docs)
            # create embeddings for the CSV
            db['db'].add_documents(texts)
        return docs

    raise ValueError(f"Unsupported file extension '{ext}'")

