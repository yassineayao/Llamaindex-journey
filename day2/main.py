import os
from pathlib import Path
from typing import Dict, List

import chromadb
import nest_asyncio
from llama_index.agent.openai import OpenAIAgent
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.query_engine import BaseQueryEngine, SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.vector_stores.types import VectorStore
from llama_index.readers.file import UnstructuredReader
from llama_index.vector_stores.chroma import ChromaVectorStore

nest_asyncio.apply()

# Global settings
Settings.chunk_size = 1000
Settings.chunk_overlap = 200


class Chatbot:
    """A simple chatbot that interacts with users and provides responses based on indexed documents."""

    DATA_FOLDER_PATH: str
    PERSIST_DIR: str
    CHROMA_COLLECTION_NAME: str
    query_engine: BaseQueryEngine
    years: List[str] = ["2022", "2021", "2020", "2019"]
    doc_set: Dict = {}

    def __init__(
        self,
        data_folder_path: str = os.path.join(".", "day2", "data"),
        persist_dir: str = os.path.join(".", "day2", "storage"),
        chroma_collection_name: str = "collection",
        similarity_top_k: int = 3,
        streaming: bool = True,
    ) -> None:
        """
        Initialize the chatbot.

        Args:
            data_folder_path (str): Path to the folder containing the document data.
            persist_dir (str): Path to the directory for storing persistent data.
            chroma_collection_name (str): Name of the Chroma collection.
            similarity_top_k (int): Number of top similar documents to retrieve.
            streaming (bool): Whether to enable streaming mode for document retrieval.
        """
        # set base paths
        self.DATA_FOLDER_PATH = data_folder_path
        self.PERSIST_DIR = persist_dir
        # Set Chroma collection name
        self.CHROMA_COLLECTION_NAME = chroma_collection_name

        # Create/load indices
        index_set = {}
        if not os.path.exists(self.PERSIST_DIR):
            index_set = self.init_chroma()
        else:
            index_set = self.load_existing_index()

        print(index_set.keys())

        individual_query_ingine_tools = [
            QueryEngineTool(
                query_engine=index_set[year].as_query_engine(
                    similarity_top_k=similarity_top_k,
                    streaming=streaming,
                ),
                metadata=ToolMetadata(
                    name=f"vector_index_{year}",
                    description=f"useful for when you want to answer queries about the {year} SEC 10-K for Uber",
                ),
            )
            for year in self.years
        ]

        query_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=individual_query_ingine_tools
        )

        query_engine_tool = QueryEngineTool(
            query_engine=query_engine,
            metadata=ToolMetadata(
                name="sub_question_query_engine",
                description="useful for when you want to answer queries that require analyzing multiple SEC 10-K documents for Uber",
            ),
        )

        tools = individual_query_ingine_tools + [query_engine_tool]

        self.agent = OpenAIAgent.from_tools(tools, verbose=True)  # type: ignore

    def init_chroma(self):
        """
        Initialize Chroma indices for each year.

        Returns:
            Dict[str, VectorStoreIndex]: A dictionary mapping years to their respective VectorStoreIndex objects.
        """
        loader = UnstructuredReader()
        all_docs = []
        index_set = {}
        for year in self.years:
            year_docs = loader.load_data(
                file=Path(f"./day2/data/UBER/UBER_{year}.html"), split_documents=False
            )
            # insert year metadata into each year
            for doc in year_docs:
                doc.metadata = {"year": year}
            self.doc_set[year] = year_docs

            chroma_client = chromadb.PersistentClient(f"{self.PERSIST_DIR}")
            chroma_collection = chroma_client.get_or_create_collection(
                f"{self.CHROMA_COLLECTION_NAME}-{year}"
            )
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            index = self.create_index(vector_store)
            index_set[year] = index

        all_docs.extend(year_docs)

        return index_set

    def create_index(self, vector_store: VectorStore):
        """
        Create a VectorStoreIndex.

        Args:
            vector_store (VectorStore): The VectorStore object to create the index from.

        Returns:
            VectorStoreIndex: The created VectorStoreIndex object.
        """
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        # Loading data
        documents = SimpleDirectoryReader(self.DATA_FOLDER_PATH).load_data()

        # Creating index
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )
        return index

    def load_existing_index(self):
        """
        Load existing Chroma indices.

        Returns:
            Dict[str, VectorStoreIndex]: A dictionary mapping years to their respective VectorStoreIndex objects.
        """
        index_set = {}
        for year in self.years:
            chroma_client = chromadb.PersistentClient(f"{self.PERSIST_DIR}")
            chroma_collection = chroma_client.get_or_create_collection(
                f"{self.CHROMA_COLLECTION_NAME}-{year}"
            )
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
            index_set[year] = index

        return index_set

    def run(self):
        """Run the chatbot."""
        while True:
            query = input("User: ")
            if query == "q":
                break
            response = self.agent.chat(query)
            print("Agent:", response)

