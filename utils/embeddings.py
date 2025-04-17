# utils/embeddings.py
import os
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever

class EmbeddingManager:
    """
    Manages document embeddings and vector retrieval.
    """
    def __init__(self, model_name: str = "text-embedding-ada-002"):
        # Try OpenAI embeddings first, fall back to local model if unavailable
        try:
            self.embeddings = OpenAIEmbeddings(model=model_name)
            self.embedding_type = "openai"
        except Exception:
            # Fall back to a local Hugging Face model
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self.embedding_type = "huggingface"
        
        self.vectorstore = None
        
    def create_vectorstore(self, documents: List[Document]) -> None:
        """Create a vector store from the provided documents"""
        if not documents:
            raise ValueError("No documents provided for vectorstore creation")
        
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
    
    def get_retriever(self, k: int = 4) -> Optional[VectorStoreRetriever]:
        """Get a retriever for the vectorstore"""
        if not self.vectorstore:
            return None
            
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents based on a query"""
        if not self.vectorstore:
            return []
            
        return self.vectorstore.similarity_search(query, k=k)
    
    def save_vectorstore(self, path: str) -> None:
        """Save the vector store to disk"""
        if not self.vectorstore:
            raise ValueError("No vectorstore to save")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        self.vectorstore.save_local(path)
    
    def load_vectorstore(self, path: str) -> None:
        """Load a vector store from disk"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vector store not found at {path}")
        
        self.vectorstore = FAISS.load_local(path, self.embeddings)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        if not self.vectorstore:
            return {"status": "empty", "document_count": 0}
            
        try:
            index_stats = self.vectorstore.index.ntotal
            return {
                "status": "loaded",
                "document_count": index_stats,
                "embedding_type": self.embedding_type
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}