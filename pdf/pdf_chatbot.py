import re
from typing import Any

import chromadb
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

from llms.poe_llm import PoeLLM

SentenceTransformerEmbeddings = HuggingFaceEmbeddings


class PDFChatBot:
    def __init__(self, llm) -> None:
        self.backend_llm = llm
        self.chain = None
        self.chat_history = []
        self.num_page = 0
        self.count = 0

    def __call__(self, file) -> Any:
        if self.count == 0:
            self.build_chain(file)
            self.count += 1

        return self.chain

    def chroma_client(self):
        """https://python.langchain.com/docs/modules/data_connection/vectorstores/integrations/chroma#basic-example-using-the-docker-container"""
        client = chromadb.Client()
        collections = client.get_or_create_collection(name="pdf-collection")
        return client

    def load(self, file):
        loader = PyPDFLoader(file.name)
        documents = loader.load()
        # get filename using regex
        pattern = r"/([^/]+)$"
        match = re.search(pattern, file.name)
        filename = match.group(1)
        return documents, filename

    def build_chain(self, file):
        documents, filename = self.load(file)
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        embedding_function = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
        search_db = Chroma.from_documents(docs, embedding_function)
        chain = ConversationalRetrievalChain.from_llm(
            self.backend_llm,
            retriever=search_db.as_retriever(search_kwargs={"k": 1}),
            return_source_documents=True
        )
        return chain
