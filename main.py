# main.py
import os
import asyncio
import nest_asyncio
from typing import Dict, List, Any, Optional, BinaryIO
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from utils.document_processor import DocumentProcessor
from utils.embeddings import EmbeddingManager
from utils.memory import ConversationManager
from agents.document_qa import DocumentQAAgent
from graph.workflow import DocumentQAWorkflow

# Apply nest_asyncio to prevent event loop errors
nest_asyncio.apply()

# Load environment variables
load_dotenv()

class DocumentQASystem:
    """
    Main system for the document question-answering application.
    """
    def __init__(self):
        # Initialize components
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.document_processor = DocumentProcessor()
        self.embedding_manager = EmbeddingManager()
        self.memory_manager = ConversationManager()
        
        # Initialize document QA agent
        self.document_qa_agent = DocumentQAAgent(llm=self.llm)
        
        # Initialize the workflow
        self.workflow = DocumentQAWorkflow(
            document_qa_agent=self.document_qa_agent,
            memory_manager=self.memory_manager
        )
        
        # Initialize document storage
        self.documents = []
        self.document_sources = []
    
    async def process_document(self, file_content: bytes, file_name: str) -> Dict[str, Any]:
        """
        Process an uploaded document and prepare it for Q&A.
        """
        try:
            # Process the document
            documents = self.document_processor.process_file(file_content, file_name)
            
            # Store documents
            self.documents.extend(documents)
            
            # Extract document sources
            source = file_name
            if source not in self.document_sources:
                self.document_sources.append(source)
            
            # Update document QA agent with sources
            self.document_qa_agent.set_document_sources(self.document_sources)
            
            # Create or update vector store
            self.embedding_manager.create_vectorstore(self.documents)
            
            # Set the retriever for the document QA agent
            retriever = self.embedding_manager.get_retriever()
            self.document_qa_agent.set_retriever(retriever)
            
            # Extract metadata for the response
            metadata = self.document_processor.extract_document_metadata(self.documents)
            
            return {
                "status": "success",
                "message": f"Successfully processed {file_name}",
                "document_count": len(self.document_sources),
                "chunk_count": len(self.documents),
                "metadata": metadata
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error processing document: {str(e)}"
            }
    
    async def ask_question(self, query: str) -> Dict[str, Any]:
        """
        Ask a question about the loaded documents.
        """
        # Check if documents are loaded
        if not self.documents:
            return {
                "status": "error",
                "message": "No documents loaded. Please upload documents first."
            }
        
        try:
            # Process the question through the workflow
            result = await self.workflow.run(query)
            
            # Check for errors
            if result.get("error"):
                return {
                    "status": "error",
                    "message": result["error"]
                }
            
            return {
                "status": "success",
                "answer": result["answer"],
                "sources": result["sources"],
                "followup_questions": result["followup_questions"]
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error processing question: {str(e)}"
            }
    
    def clear_documents(self) -> Dict[str, Any]:
        """
        Clear all loaded documents.
        """
        self.documents = []
        self.document_sources = []
        self.embedding_manager = EmbeddingManager()  # Reset the embedding manager
        
        # Reset the document QA agent
        self.document_qa_agent.set_document_sources([])
        self.document_qa_agent.set_retriever(None)
        
        return {
            "status": "success",
            "message": "All documents cleared"
        }
    
    def clear_chat_history(self) -> Dict[str, Any]:
        """
        Clear the chat history.
        """
        self.memory_manager.clear()
        
        return {
            "status": "success",
            "message": "Chat history cleared"
        }
    
    def get_document_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded documents.
        """
        if not self.documents:
            return {
                "status": "info",
                "message": "No documents loaded",
                "document_count": 0,
                "sources": []
            }
        
        # Get stats from document processor
        metadata = self.document_processor.extract_document_metadata(self.documents)
        
        # Get stats from embedding manager
        embedding_stats = self.embedding_manager.get_stats()
        
        return {
            "status": "success",
            "document_count": len(self.document_sources),
            "chunk_count": len(self.documents),
            "sources": self.document_sources,
            "metadata": metadata,
            "embeddings": embedding_stats
        }
    
    def get_chat_history(self) -> List[Dict[str, Any]]:
        """
        Get the current chat history.
        """
        return self.memory_manager.get_chat_history()

# Initialize the system
document_qa_system = DocumentQASystem()

async def process_document(file_content: bytes, file_name: str) -> Dict[str, Any]:
    """Process a document through the system"""
    return await document_qa_system.process_document(file_content, file_name)

async def ask_question(query: str) -> Dict[str, Any]:
    """Ask a question to the system"""
    return await document_qa_system.ask_question(query)

def clear_documents() -> Dict[str, Any]:
    """Clear all documents from the system"""
    return document_qa_system.clear_documents()

def clear_chat_history() -> Dict[str, Any]:
    """Clear chat history"""
    return document_qa_system.clear_chat_history()

def get_document_stats() -> Dict[str, Any]:
    """Get document statistics"""
    return document_qa_system.get_document_stats()

def get_chat_history() -> List[Dict[str, Any]]:
    """Get chat history"""
    return document_qa_system.get_chat_history()