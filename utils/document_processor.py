# utils/document_processor.py
import os
import pandas as pd
import io
from typing import List, Dict, Any, Union
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    TextLoader
)
from langchain_core.documents import Document

class DocumentProcessor:
    """
    Handles processing of various document types.
    Supports: PDF, DOCX, CSV, Excel, and text files.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
    def process_file(self, file_content: bytes, file_name: str) -> List[Document]:
        """Process a file and return langchain documents"""
        file_extension = os.path.splitext(file_name)[1].lower()
        
        # Create a temporary file for loaders that require file paths
        temp_file_path = f"temp_{file_name}"
        with open(temp_file_path, 'wb') as f:
            f.write(file_content)
        
        try:
            if file_extension == '.pdf':
                documents = self._process_pdf(temp_file_path)
            elif file_extension == '.docx':
                documents = self._process_docx(temp_file_path)
            elif file_extension == '.csv':
                documents = self._process_csv(temp_file_path)
            elif file_extension in ['.xlsx', '.xls']:
                documents = self._process_excel(temp_file_path)
            elif file_extension == '.txt':
                documents = self._process_text(temp_file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            # Add metadata about the source file
            for doc in documents:
                doc.metadata["source"] = file_name
                
            return documents
            
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    
    def _process_pdf(self, file_path: str) -> List[Document]:
        """Process PDF files"""
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return self.text_splitter.split_documents(documents)
    
    def _process_docx(self, file_path: str) -> List[Document]:
        """Process DOCX files"""
        loader = Docx2txtLoader(file_path)
        documents = loader.load()
        return self.text_splitter.split_documents(documents)
    
    def _process_csv(self, file_path: str) -> List[Document]:
        """Process CSV files with pandas for better handling"""
        try:
            df = pd.read_csv(file_path)
            # Convert DataFrame to a text representation
            text_chunks = []
            
            # Get column descriptions
            columns_desc = f"CSV Columns: {', '.join(df.columns)}\n\n"
            
            # Process in chunks to avoid memory issues with large files
            chunk_size = 100  # Adjust based on expected CSV size
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i+chunk_size]
                
                # Create a readable text representation of the chunk
                chunk_text = columns_desc + chunk.to_string(index=False)
                doc = Document(page_content=chunk_text, metadata={"source": file_path, "chunk": i//chunk_size})
                text_chunks.append(doc)
            
            return self.text_splitter.split_documents(text_chunks)
        except Exception as e:
            # Fallback to the standard CSVLoader if pandas approach fails
            loader = CSVLoader(file_path)
            documents = loader.load()
            return self.text_splitter.split_documents(documents)
    
    def _process_excel(self, file_path: str) -> List[Document]:
        """Process Excel files"""
        loader = UnstructuredExcelLoader(file_path, mode="elements")
        documents = loader.load()
        return self.text_splitter.split_documents(documents)
    
    def _process_text(self, file_path: str) -> List[Document]:
        """Process plain text files"""
        loader = TextLoader(file_path)
        documents = loader.load()
        return self.text_splitter.split_documents(documents)

    def extract_document_metadata(self, documents: List[Document]) -> Dict[str, Any]:
        """Extract and summarize metadata from a list of documents"""
        if not documents:
            return {}
            
        unique_sources = set()
        for doc in documents:
            if "source" in doc.metadata:
                unique_sources.add(doc.metadata["source"])
        
        total_chunks = len(documents)
        avg_chunk_size = sum(len(doc.page_content) for doc in documents) / total_chunks if total_chunks > 0 else 0
        
        return {
            "document_count": len(unique_sources),
            "sources": list(unique_sources),
            "chunk_count": total_chunks,
            "avg_chunk_size": round(avg_chunk_size, 2)
        }