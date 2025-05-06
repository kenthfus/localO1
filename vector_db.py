from abc import ABC, abstractmethod
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
import logging

logger = logging.getLogger(__name__)

class VectorDB(ABC):

    """Abstract base class for vector database operations."""
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        pass
    
    @abstractmethod
    def remove_embeddings(self, document_id: str) -> None:
        """Remove existing embeddings for a given document_id."""
        pass
    
    @abstractmethod
    def similarity_search_with_score(self, query: str, k: int = 4, filter: Optional[dict] = None) -> List:
        """Remove existing embeddings for a given file path."""
        pass

    @abstractmethod
    def max_marginal_relevance_search_with_score(self, query: str, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5, filter: Optional[dict] = None) -> List:
        """Remove existing embeddings for a given file path."""
        pass

    @abstractmethod
    def query_search(self, query: str, filter: dict={}) -> List:
        pass

    @abstractmethod
    def check_documents_exist(self, checksum: List[str]) -> None:
        """Delete the entire collection."""
        pass

    @abstractmethod
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        pass

    @abstractmethod
    def get_vector_store(self, collection_name: str) -> VectorStore:
        pass

