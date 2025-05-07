
# vector_db/test-mongovector.py
import pytest
import os
import sys
import time
from langchain_core.documents import Document
from pymongo import MongoClient
from dotenv import load_dotenv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from vector_db.mongovector import MongoDBVectorDB
from vector_db.vector_db import VectorDB
from typing import List

load_dotenv()

class TestEmbeddings:
    """Deterministic test embeddings for consistent results"""
    def __init__(self, size=1024):
        self.size = size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[float(i % 1000) / 100 for i in range(self.size)] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [float(ord(c)) / 100 for c in text[:self.size]]

@pytest.fixture(scope="module")
def test_db():
    """MongoDB test database setup"""
    client = MongoClient("mongodb+srv://ehrehp20mongodev:QfvSb5Kh4cFXg5cz@cluster0.gcjlq1m.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
    db = client["test_vector_db"]
    
    # Clean up before tests
    db.drop_collection("test_collection")
    
    yield db
    
    # Clean up after tests
    client.drop_database("test_vector_db")
    client.close()

@pytest.fixture
def vector_store(request, test_db):
    """MongoDBVectorDB instance for testing"""
    # Use request.param to support multiple test configurations
    use_embeddings = getattr(request, 'param', True)

    return MongoDBVectorDB(
        embeddings=TestEmbeddings(size=1024) if use_embeddings else None,
        collection_name="test_collection",
        connection="mongodb+srv://ehrehp20mongodev:QfvSb5Kh4cFXg5cz@cluster0.gcjlq1m.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
        database_name="test_vector_db",
        index_name="default_text_index"
    )

@pytest.fixture
def sample_documents():
    """Test documents with consistent structure"""
    return [
        Document(page_content="Python is a popular programming language", metadata={"checksum": "doc1", "source": "test", "document_id": "doc1"}),
        Document(page_content="MongoDB is a NoSQL database", metadata={"checksum": "doc2", "source": "test", "document_id": "doc2"}),
        Document(page_content="LangChain is a framework for LLM applications", metadata={"checksum": "doc3", "source": "test", "document_id": "doc3"})
    ]

def test_add_documents(vector_store, sample_documents):
    """Test document insertion with count verification"""
    initial_count = vector_store.collection.count_documents({})
    vector_store.add_documents(sample_documents)
    new_count = vector_store.collection.count_documents({})
    assert new_count == initial_count + len(sample_documents)

def test_index_creation(vector_store):
    """Test search index creation with cleanup"""
    # Clean up any existing index
    try:
        vector_store.collection.drop_index("page_content_text")
    except Exception:
        pass

    # Create new index
    vector_store.create_index()
    
    # Verify index exists
    indexes = vector_store.collection.list_indexes()
    index_names = [idx.get('name') for idx in indexes]
    assert "page_content_text" in index_names

def test_similarity_search(vector_store, sample_documents):
    """Test text-based similarity search with deterministic results"""
    # vector_store.add_documents(sample_documents)
    
    # Wait for index to catch up
    time.sleep(5)
    
    results = vector_store.similarity_search_with_score("programming", k=1)
    assert len(results) == 1
    assert "Python" in results[0][0].page_content

def test_check_documents_exist(vector_store, sample_documents):
    """Test document existence verification"""
    # vector_store.add_documents(sample_documents)
    
    # Small delay for database consistency
    time.sleep(1)
    
    assert vector_store.check_documents_exist(["doc1", "doc3"]) is True
    assert vector_store.check_documents_exist(["doc4"]) is False

def test_query_search(vector_store, sample_documents):
    """Test direct MongoDB queries with field projection for 'NoSQL'"""
    # Add sample documents to the collection
    # vector_store.add_documents(sample_documents)

    # Create a text index on the 'page_content' field
    # vector_store.create_index()

    # Allow time for the index to be built and data to be searchable
    # time.sleep(5)

    # Perform the text search query for "NoSQL"
    results = vector_store.query_search(
        {"$text": {"$search": '"NoSQL"'}},  # Use quotes around search term for exact phrase match
        {"page_content": 1, "_id": 0}
    )

    # Validate the results
    assert len(results) == 1, "Expected exactly one document containing 'NoSQL'"
    assert "MongoDB" in results[0]["page_content"], "Result should contain 'MongoDB'"

def test_query_search_02(vector_store, sample_documents):
    """Test direct MongoDB queries with field projection"""
    # vector_store.add_documents(sample_documents)
    
    time.sleep(1)  # Allow indexing
    
    results = vector_store.query_search(
        {"metadata.source": "test"},
        {"page_content": 1, "_id": 0}
    )
    
    assert len(results) == 3
    assert all("Python" in doc["page_content"] or 
               "MongoDB" in doc["page_content"] or
               "LangChain" in doc["page_content"]
               for doc in results)

def test_remove_embeddings(vector_store, sample_documents):
    """Test document removal by document_id"""
    # vector_store.add_documents(sample_documents)
    
    time.sleep(1)  # Allow indexing
    
    vector_store.remove_embeddings("doc2")
    remaining = vector_store.collection.count_documents({})
    assert remaining == len(sample_documents) - 1

def test_delete_collection(vector_store, sample_documents):
    """Test full collection deletion"""
    vector_store.add_documents(sample_documents)
    vector_store.delete_collection()
    assert "test_collection" not in vector_store.db.list_collection_names()

if __name__ == "__main__":
    pytest.main(["-v", "--disable-warnings"])
