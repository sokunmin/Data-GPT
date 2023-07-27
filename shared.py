import os
import gradio as gr
import shutil
from os.path import basename, exists, join
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
PERSIST_DB_DIR = os.environ.get('PERSIST_DB_DIR')
PERSIST_FILE_DIR = os.environ.get('PERSIST_FILE_DIR')
FILE_DIR = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
EMBED_MODEL_NAME = os.environ.get('EMBEDDINGS_MODEL_NAME')
CHUNK_SIZE = os.environ.get('CHUNK_SIZE', 500)
CHUNK_OVERLAP = os.environ.get('CHUNK_OVERLAP', 50)
CHROMA_SETTINGS = Settings(
    persist_directory=PERSIST_DB_DIR,
    anonymized_telemetry=False,
    is_persistent=True
)
gr = {}

db = {
    'db': None,
    'files': dict(),
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


def update_file_list():
    if db['db']:
        collections = set([metadata['source'] for metadata in db['db'].get()['metadatas']])
        db['files'] = {basename(f): f for f in collections}
        # TODO: categorize to various extensions


def load_file_and_split_chunks(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in FILE_MAPPING:
        loader_class, loader_args = FILE_MAPPING[ext]
        # move to `db/files` folder
        if not exists(PERSIST_FILE_DIR):
            os.makedirs(PERSIST_FILE_DIR)
        if file_path not in db['files']:
            shutil.move(file_path, PERSIST_FILE_DIR)
            file_path = join(PERSIST_FILE_DIR, basename(file_path))
        loader = loader_class(file_path, **loader_args)
        docs = loader.load()
        if file_path not in db['files']:
            # split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            texts = text_splitter.split_documents(docs)
            # create embeddings for the CSV
            db['db'].add_documents(texts)
            update_file_list()
        return docs

    raise gr.Error(f"Unsupported file extension '{ext}'")

