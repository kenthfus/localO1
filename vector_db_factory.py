from .pgvector import PGVectorDB
from .vector_db import VectorDB
from typing import Optional

class VectorDBFactory:
    """Factory for creating VectorDB instances."""
    
    @staticmethod
    def create_vector_db(db_type: str, embeddings, collection_name: str, connection: Optional[str] = None) -> VectorDB:
        """Create a VectorDB instance based on the specified type."""
        if db_type.lower() == "pgvector":
            if not connection:
                raise ValueError("Connection string required for PGVector")
            return PGVectorDB(embeddings, collection_name, connection)
        # Add other vector store types (e.g., FAISS, Pinecone) here
        else:
            raise ValueError(f"Unsupported vector store type: {db_type}")
