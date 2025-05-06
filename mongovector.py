from langchain_core.documents import Document
import logging
import os
from typing import List, Optional
from pymongo import MongoClient
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from .vector_db import VectorDB
from openai import OpenAI

logger = logging.getLogger(__name__)

class MongoDBVectorDB(VectorDB):
    """MongoDB Atlas vector search implementation of VectorDB with DashScope embeddings."""
    
    def __init__(self, embeddings, collection_name: str, connection: str, 
                 database_name: str = "vector_db", index_name: str = "vector_index"):
        # Initialize MongoDB client and collection
        self.client = MongoClient(connection)
        self.db = self.client[database_name]
        self.collection = self.db[collection_name]
        
        # Initialize OpenAI-compatible client for DashScope
        self.embeddings_client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        # Configure vector store with custom embeddings
        self.vector_store = MongoDBAtlasVectorSearch(
            collection=self.collection,
            embedding=self._get_dashscope_embeddings(embeddings),
            index_name=index_name
        )
        self.collection_name = collection_name
        self.database_name = database_name
        self.index_name = index_name

    def _get_dashscope_embeddings(self, embeddings):
        """Wrap the embeddings model with DashScope-compatible execution"""
        original_embed_documents = embeddings.embed_documents
        original_embed_query = embeddings.embed_query
        
        def dashscope_embed_documents(texts: List[str]) -> List[List[float]]:
            response = self.embeddings_client.embeddings.create(
                model="text-embedding-v3",
                input=texts,
                dimensions=1024,  # Match your model configuration
                encoding_format="float"
            )
            return [e.embedding for e in response.data]
        
        def dashscope_embed_query(text: str) -> List[float]:
            response = self.embeddings_client.embeddings.create(
                model="text-embedding-v3",
                input=text,
                dimensions=1024,  # Match your model configuration
                encoding_format="float"
            )
            return response.data[0].embedding
        
        # Monkey-patch the embedding methods
        embeddings.embed_documents = dashscope_embed_documents
        embeddings.embed_query = dashscope_embed_query
        return embeddings

    # Keep all original method implementations unchanged below
    def add_documents(self, documents: List[Document]) -> None:
        try:
            result = self.vector_store.add_documents(documents)
            logger.info("Added %d documents to collection %s", len(result), self.collection_name)
        except Exception as e:
            logger.error("Error adding documents to MongoDB: %s", str(e))
            raise
    
    def similarity_search_with_score(self, query: str, k: int = 4, filter: Optional[dict] = None) -> List:
        return self.vector_store.similarity_search_with_score(query, k=k, pre_filter=filter)
    
    def max_marginal_relevance_search_with_score(
        self, query: str, k: int = 4, fetch_k: int = 20, 
        lambda_mult: float = 0.5, filter: Optional[dict] = None
    ) -> List:
        return self.vector_store.max_marginal_relevance_search(
            query, 
            k=k, 
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            pre_filter=filter
        )
    
    def query_search(self, query: dict, projection: dict = None) -> List:
        try:
            if projection:
                cursor = self.collection.find(query, projection)
            else:
                cursor = self.collection.find(query)
            return list(cursor)
        except Exception as e:
            logger.error("Error executing query: %s", str(e))
            raise
            
    def check_documents_exist(self, checksum: List[str]) -> bool:
        try:
            count = self.collection.count_documents({
                "metadata.checksum": {"$in": checksum}
            })
            return count > 0
        except Exception as e:
            logger.error("Error checking documents: %s", str(e))
            return False

    def remove_embeddings(self, document_id: str) -> None:
        try:
            result = self.collection.delete_many({
                "metadata.document_id": document_id
            })
            logger.info("Removed %d existing embeddings for document: %s", 
                       result.deleted_count, document_id)
        except Exception as e:
            logger.error("Error removing embeddings: %s", str(e))
    
    def delete_collection(self) -> None:
        try:
            self.db.drop_collection(self.collection_name)
            logger.info("Deleted collection %s", self.collection_name)
        except Exception as e:
            logger.error("Error deleting collection: %s", str(e))
            raise
    
    def get_vector_store(self) -> MongoDBAtlasVectorSearch:
        return self.vector_store

    def create_index(self, dimensions: int = 1024) -> None:  # Updated default to match DashScope
        index = {
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": dimensions,
                    "similarity": "cosine"
                },
                {
                    "type": "filter",
                    "path": "metadata"
                }
            ]
        }
        try:
            self.db.command("createSearchIndexes", self.collection_name, indexes=[index])
            logger.info("Created vector search index %s", self.index_name)
        except Exception as e:
            logger.error("Error creating index: %s", str(e))
