import os

import chromadb
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.vector_stores.types import VectorStore
from llama_index.vector_stores.chroma import ChromaVectorStore

# Global settings
Settings.chunk_size = 1000
Settings.chunk_overlap = 200


class RAG:
    """
    Retrieval Augmented Generation (RAG)

    Attributes:
        DATA_FOLDER_PATH (str): The path to the data folder.
        PERSIST_DIR (str): The folder for persisting Chroma db.
        CHROMA_COLLECTION_NAME (str): The name of the Chroma collection.
        query_engine (BaseQueryEngine): The query engine for performing searches.
    """

    DATA_FOLDER_PATH: str
    PERSIST_DIR: str
    CHROMA_COLLECTION_NAME: str
    query_engine: BaseQueryEngine

    def __init__(
        self,
        data_folder_path: str = os.path.join(".", "day1", "data"),
        persist_dir: str = os.path.join(".", "day1", "storage"),
        chroma_collection_name: str = "collection",
        similarity_top_k: int = 3,
        streaming: bool = True,
    ) -> None:
        """
        Initializes the RAG with specified parameters.

        Args:
            data_folder_path (str, optional): The path to the data folder. Defaults to "./day1/data".
            persist_dir (str, optional): The directory for persisting Chroma db. Defaults to "./day1/storage".
            chroma_collection_name (str, optional): The name of the Chroma collection. Defaults to "collection".
            similarity_top_k (int, optional): The number of top similar nodes to retrieve. Defaults to 3.
            streaming (bool, optional): Whether to stream the LLM response or not. Defaults to True.
        """
        # set base paths
        self.DATA_FOLDER_PATH = data_folder_path
        self.PERSIST_DIR = persist_dir
        # Set Chroma collection name
        self.CHROMA_COLLECTION_NAME = chroma_collection_name

        # Create/load indices
        vector_store = self.init_chroma()
        if not os.path.exists(self.PERSIST_DIR):
            index = self.create_index(vector_store)
        else:
            index = self.load_existing_index(vector_store)

        self.query_engine = index.as_query_engine(
            similarity_top_k=similarity_top_k,
            streaming=streaming,
        )

    def init_chroma(self):
        """
        Initializes the Chroma Vector Store and returns it.

        Returns:
            ChromaVectorStore: The initialized Chroma Vector Store.
        """
        chroma_client = chromadb.PersistentClient(self.PERSIST_DIR)
        chroma_collection = chroma_client.get_or_create_collection(
            self.CHROMA_COLLECTION_NAME
        )
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        return vector_store

    def create_index(self, vector_store: VectorStore):
        """
        Creates a new index using the provided vector store and returns it.

        Args:
            vector_store (VectorStore): The vector store for creating the index.

        Returns:
            VectorStoreIndex: The created index.
        """
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        # Loading data
        documents = SimpleDirectoryReader(self.DATA_FOLDER_PATH).load_data()

        # Creating index
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )
        return index

    def load_existing_index(self, vector_store: VectorStore):
        """
        Loads an existing index from the provided vector store and returns it.

        Args:
            vector_store (VectorStore): The vector store containing the index.

        Returns:
            VectorStoreIndex: The loaded index.
        """
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        return index

    def run(self, query):
        """
        Executes a query using the query engine and returns the results.

        Args:
            query: The query to be executed.

        Returns:
            QueryResult: The response from the LLM
        """
        return self.query_engine.query(query)

