from typing import List
from .mongovector import MongoDBVectorDB
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import os
from openai import OpenAI

class DashScopeEmbeddings(Embeddings):
    """Native DashScope embedding implementation"""
    
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = "text-embedding-v3"
        self.dimensions = 1024
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dimensions,
            encoding_format="float"
        )
        return [embedding.embedding for embedding in response.data]
    
    def embed_query(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
            dimensions=self.dimensions,
            encoding_format="float"
        )
        return response.data[0].embedding

# Initialize vector store
# mongo_uri = "mongodb+srv://user:pass@cluster.mongodb.net/"
mongo_uri = "mongodb+srv://ehrehp20mongodev:QfvSb5Kh4cFXg5cz@cluster0.gcjlq1m.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
vector_db = MongoDBVectorDB(
    embeddings=DashScopeEmbeddings(),  # Use native implementation
    collection_name="documents",
    connection=mongo_uri,
    database_name="ai_docs"
)

# Create index
vector_db.create_index(dimensions=1024)

# Example usage
vector_db.add_documents([
    Document(page_content="Quality product...", metadata={"source": "dashscope"})
])

results = vector_db.similarity_search_with_score("search query", k=5)
vector_db.remove_embeddings("doc_123")
