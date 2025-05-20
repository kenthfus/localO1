import os
import pytest
from pymongo import MongoClient
from langchain_core.documents import Document
from vector_db.mongodb_vector_db import MongoDBVectorDB
from dotenv import load_dotenv

load_dotenv()

# Configurable values
MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = "test_vector_db"
COLLECTION_NAME = "test_collection"
INDEX_NAME = "test_index"  

# Mock Embeddings
class MockEmbeddings:
    def __init__(self):
        self.embed_query = self._embed_query
        self.embed_documents = self._embed_documents

    def _embed_query(self, text):
        return [3.0] * 1536

    def _embed_documents(self, texts):
        return [[1.0 + i] * 1536 for i, _ in enumerate(texts)]

@pytest.fixture
def shared_client():
    client = MongoClient(MONGO_URI)
    yield client
    client.close()

@pytest.fixture
def vector_db(shared_client):
    """Reusable base setup for each test"""
    def _get_db(collection_name=COLLECTION_NAME):
        db = MongoDBVectorDB(
            embeddings=MockEmbeddings(),
            collection_name=collection_name,
            connection=MONGO_URI,
            database_name=DATABASE_NAME,
            index_name=INDEX_NAME,
            client=shared_client
        )
        return db
    return _get_db

# --- Individual Tests ---

def test_add_documents(vector_db):
    db = vector_db("add_docs_test")
    docs = [
        Document(page_content="Hello world", metadata={"source": "test1"}),
        Document(page_content="Goodbye world", metadata={"source": "test2"})
    ]
    db.add_documents(docs)
    assert db.collection.count_documents({}) == 2

def test_similarity_search_with_score(vector_db):
    db = vector_db("search_test")
    # Setup
    docs = [
        Document(page_content="Apple is a fruit", metadata={"source": "fruit"}),
        Document(page_content="Tesla is a car", metadata={"source": "car"})
    ]
    db.add_documents(docs)
    
    # Create index
    try:
        db.collection.drop_index(INDEX_NAME)
    except Exception:
        pass
    db.create_index()

    # Test search
    results = db.similarity_search_with_score("fruit", k=1)
    assert len(results) == 1
    doc, score = results[0]
    assert doc.page_content == "Apple is a fruit"
    assert score > 0.9

def test_check_documents_exist(vector_db):
    db = vector_db("check_exist_test")
    doc = Document(page_content="Test content", metadata={"checksum": "abc123"})
    db.add_documents([doc])
    assert db.check_documents_exist(["abc123"]) is True
    assert db.check_documents_exist(["xyz789"]) is False

def test_remove_embeddings(vector_db):
    db = vector_db("remove_embeddings_test")
    doc = Document(page_content="Test content", metadata={"document_id": "doc123"})
    db.add_documents([doc])
    db.remove_embeddings("doc123")
    assert db.collection.count_documents({}) == 0

def test_query_search(vector_db):
    db = vector_db("query_search_test")
    db.collection.insert_one({"name": "John", "age": 30})
    result = db.query_search({"name": "John"})
    assert len(result) == 1
    assert result[0]["name"] == "John"

def test_delete_collection(vector_db):
    db = vector_db("delete_collection_test")
    db.collection.insert_one({"name": "WillBeDeleted"})
    db.delete_collection()
    assert db.collection.count_documents({}) == 0

def test_create_index(vector_db):
    db = vector_db("create_index_test")
    db.create_index()
    indices = list(db.collection.list_search_indexes(db.index_name))
    assert any(i.get("queryable") is True for i in indices)

def test_max_marginal_relevance_search_with_score(vector_db):
    db = vector_db("mmr_search_test")

    # Ensure index is created before inserting data
    try:
        db.collection.drop_index(INDEX_NAME)
    except Exception:
        pass
    db.create_index()  # Create index before inserting

    # Add test documents
    docs = [
        Document(page_content="Apple is a fruit", metadata={"source": "fruit"}),
        Document(page_content="Banana is a fruit", metadata={"source": "fruit"}),
        Document(page_content="Tesla is a car", metadata={"source": "car"})
    ]
    db.add_documents(docs)

    # Perform search
    results = db.max_marginal_relevance_search_with_score("fruit", k=2)

    # Validate
    assert len(results) == 2
    assert "fruit" in results[0][0].page_content.lower()

def test_get_vector_store(vector_db):
    db = vector_db("get_vector_store_test")
    with pytest.raises(NotImplementedError):
        db.get_vector_store()
