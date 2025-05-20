from langchain_core.documents import Document
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from langchain_core.vectorstores import VectorStore
from openai import OpenAI
from typing import Optional, List
import os
import logging
import time

logger = logging.getLogger(__name__)

class MongoDBVectorDB:
    _client = None  # Class-level shared client

    def __init__(
        self,
        embeddings,
        collection_name: str,
        connection: str,
        database_name: str = "vector_db",
        index_name: str = "default_vector_index",
        client: Optional[MongoClient] = None
    ):
        """
        Initialize MongoDBVectorDB with optional shared MongoClient.
        If no client is provided, one will be created and reused for future instantiations.
        """
        if MongoDBVectorDB._client is None or client is not None:
            MongoDBVectorDB._client = client if client else MongoClient(connection)

        self.client = MongoDBVectorDB._client
        self.db = self.client[database_name]
        self.collection_name = collection_name
        self.index_name = index_name
        self.collection = self.db[collection_name]

        # Ensure collection exists
        if collection_name not in self.db.list_collection_names():
            self.db.create_collection(collection_name)

        # Initialize DashScope-compatible embedding client
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
        docs = []
        texts = [doc.page_content for doc in documents]
        embeddings = self.embeddings.embed_documents(texts)

        for doc, embedding in zip(documents, embeddings):
            docs.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata,
                "embedding": embedding
            })

        if docs:
            result = self.collection.insert_many(docs)
            logger.info(f"Inserted {len(result.inserted_ids)} documents")
        else:
            logger.warning("No documents to insert")

    def similarity_search_with_score(self, query: str, k: int = 4, filter: Optional[dict] = None):
        query_embedding = self.embeddings.embed_query(query)

        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.index_name,
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 100,
                    "limit": k
                }
            },
            {
                "$project": {
                    "page_content": 1,
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]

        if filter:
            pipeline.insert(0, {"$match": filter})

        results = list(self.collection.aggregate(pipeline))
        return [(Document(page_content=r["page_content"], metadata=r.get("metadata")), r["score"]) for r in results]

    def create_index(self) -> None:
        try:
            index_definition = {
                "fields": [{
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": 1024,
                    "similarity": "dotProduct",
                    "quantization": "scalar"
                }]
            }

            search_index_model = SearchIndexModel(
                definition=index_definition,
                name=self.index_name,
                type="vectorSearch"
            )

            result = self.collection.create_search_index(model=search_index_model)
            logger.info(f"Vector index '{self.index_name}' created successfully. Index ID: {result}")

            # Wait for index to become queryable
            while True:
                indices = list(self.collection.list_search_indexes(self.index_name))
                if indices and indices[0].get("queryable") is True:
                    break
                time.sleep(5)
            logger.info(f"Vector index '{self.index_name}' is now queryable.")

        except Exception as e:
            logger.error(f"Failed to create vector index: {str(e)}")
            raise
    
    def max_marginal_relevance_search_with_score(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict] = None
    ) -> List:
        """Perform MMR search using similarity search"""
        return self.similarity_search_with_score(query, k=k, filter=filter)

    def check_documents_exist(self, checksum: List[str]) -> bool:
        count = self.collection.count_documents({"metadata.checksum": {"$in": checksum}})
        return count > 0
    
    def remove_embeddings(self, document_id: str) -> None:
        self.collection.delete_many({"metadata.document_id": document_id})

    def query_search(self, query: dict, filter: dict = {}) -> List:
        return list(self.collection.find(query, filter))

    def delete_collection(self) -> None:
        self.db.drop_collection(self.collection_name)

    def get_vector_store(self) -> VectorStore:
        """Get a LangChain VectorStore instance (not supported in this implementation)"""
        raise NotImplementedError("Not applicable for this implementation")
    
    def close(self):
        """Close the shared MongoDB client"""
        if MongoDBVectorDB._client:
            MongoDBVectorDB._client.close()
            MongoDBVectorDB._client = None
