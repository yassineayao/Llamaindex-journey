import os

from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.query_engine import BaseQueryEngine


class RAG:
    DATA_FOLDER_PATH: str
    PERSIST_DIR: str
    query_engine: BaseQueryEngine

    def __init__(
        self,
        data_folder_path: str = os.path.join(".", "day0", "data"),
        persist_dir: str = os.path.join(".", "day0", "storage"),
    ) -> None:
        # set base paths
        self.DATA_FOLDER_PATH = data_folder_path
        self.PERSIST_DIR = persist_dir

        # Create/load indeces
        if not os.path.exists(self.PERSIST_DIR):
            self.create_index()
        else:
            self.load_existing_index()

    def create_index(self):
        # Loading data
        documents = SimpleDirectoryReader(self.DATA_FOLDER_PATH).load_data()

        # creating index
        index = VectorStoreIndex.from_documents(documents)

        # Storing index
        index.storage_context.persist(persist_dir=self.PERSIST_DIR)
        self.query_engine = index.as_query_engine()

    def load_existing_index(self):
        storage_context = StorageContext.from_defaults(persist_dir=self.PERSIST_DIR)
        index = load_index_from_storage(storage_context)
        self.query_engine = index.as_query_engine()

    def run(self, query):
        return self.query_engine.query(query)


if __name__ == "__main__":
    rag = RAG()
    rag.run("What did the author do growing up?")
