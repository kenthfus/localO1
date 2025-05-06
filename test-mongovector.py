from .mongovector import MongoDBVectorDB
from langchain_core.documents import Document
from langchain.embeddings import OpenAIEmbeddings

# Initialize vector store
# mongo_uri = "mongodb+srv://user:pass@cluster.mongodb.net/"
mongo_uri = "mongodb+srv://ehrehp20mongodev:QfvSb5Kh4cFXg5cz@cluster0.gcjlq1m.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
vector_db = MongoDBVectorDB(
    embeddings=OpenAIEmbeddings(),
    collection_name="documents",
    connection=mongo_uri,
    database_name="ai_docs"
)

# Create vector index (once)
vector_db.create_index(dimensions=1536)  # Match embedding dimensions

# Add documents
vector_db.add_documents([Document(...)])

# Semantic search
results = vector_db.similarity_search_with_score("search query", k=5)

# Delete documents
vector_db.remove_embeddings("doc_123")
