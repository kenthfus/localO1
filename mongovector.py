# vector_db/mongovector.py
from langchain_core.documents import Document
from pymongo import MongoClient, errors
from openai import OpenAI
import logging
import os
from typing import List, Optional, Dict, Union
from abc import ABC, abstractmethod
from langchain_core.vectorstores import VectorStore
from vector_db.vector_db import VectorDB  # Import your base class

logger = logging.getLogger(__name__)

class MongoDBVectorDB(VectorDB):
    """Concrete implementation of VectorDB using MongoDB Atlas Text Search"""

    def __init__(
        self,
        embeddings,
        collection_name: str,
        connection: str,
        database_name: str = "vector_db",
        index_name: str = "default_text_index"
    ):
        self.client = MongoClient(connection)
        self.db = self.client[database_name]
        self.collection_name = collection_name
        self.index_name = index_name

        # Ensure database and collection exist
        if collection_name not in self.db.list_collection_names():
            self.db.create_collection(collection_name)

        self.collection = self.db[collection_name]

        # Initialize DashScope-compatible embedding client (optional)
        self.embeddings_client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        # Wrap embedding functions if embeddings are provided
        self.embeddings = embeddings
        if embeddings:
            embeddings.embed_documents = self._wrap_embed_documents(embeddings.embed_documents)
            embeddings.embed_query = self._wrap_embed_query(embeddings.embed_query)

    def _wrap_embed_documents(self, original_embed_documents):
        def wrapped(texts: List[str]) -> List[List[float]]:
            response = self.embeddings_client.embeddings.create(
                model="text-embedding-v3",
                input=texts,
                dimensions=1024,
                encoding_format="float"
            )
            return [e.embedding for e in response.data]
        return wrapped

    def _wrap_embed_query(self, original_embed_query):
        def wrapped(text: str) -> List[float]:
            response = self.embeddings_client.embeddings.create(
                model="text-embedding-v3",
                input=text,
                dimensions=1024,
                encoding_format="float"
            )
            return response.data[0].embedding
        return wrapped

    def add_documents(self, documents: List[Document]) -> None:
        """Insert LangChain Documents into the collection"""
        docs = []
        for doc in documents:
            docs.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            })

        if docs:
            result = self.collection.insert_many(docs)
            logger.info(f"Inserted {len(result.inserted_ids)} documents")
        else:
            logger.warning("No documents to insert")

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None
    ) -> List:
        """Perform Atlas Text Search using $text operator"""
        pipeline = [
            {
                "$match": {
                    "$text": {"$search": query}
                }
            },
            {
                "$project": {
                    "page_content": 1,
                    "metadata": 1,
                    "score": {"$meta": "textScore"}
                }
            },
            {
                "$sort": {"score": {"$meta": "textScore"}}
            },
            {
                "$limit": k
            }
        ]

        if filter:
            pipeline.insert(0, {"$match": filter})

        results = list(self.collection.aggregate(pipeline))
        return [(Document(page_content=r["page_content"], metadata=r.get("metadata")), r["score"]) for r in results]

    def max_marginal_relevance_search_with_score(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict] = None
    ) -> List:
        """Fallback to simple text search for MMR"""
        return self.similarity_search_with_score(query, k=k, filter=filter)

    def query_search(self, query: dict, filter: dict = {}) -> List:
        """Run a direct MongoDB query on the collection"""
        return list(self.collection.find(query, filter))

    def check_documents_exist(self, checksum: List[str]) -> bool:
        """Check if documents with given checksums exist in the collection"""
        count = self.collection.count_documents({"metadata.checksum": {"$in": checksum}})
        return count > 0

    def remove_embeddings(self, document_id: str) -> None:
        """Remove document by document_id from metadata"""
        self.collection.delete_many({"metadata.document_id": document_id})

    def delete_collection(self) -> None:
        """Drop the entire collection"""
        self.db.drop_collection(self.collection_name)

    def get_vector_store(self) -> VectorStore:
        raise NotImplementedError("Not applicable for text-based search")

    def create_index(self) -> None:
        """Create a standard Atlas Text Search index on 'page_content'"""
        try:
            # Create a text index on page_content field
            self.collection.create_index([("page_content", "text")])
            logger.info(f"Atlas Text Search index created on '{self.collection_name}.page_content'")
        except errors.OperationFailure as e:
            logger.error(f"Failed to create index: {str(e)}")
            raise
