# agents/document_qa.py
import os
from typing import Dict, List, Any, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from utils.prompts import DOCUMENT_QA_PROMPT, DOCUMENT_QA_SYSTEM_TEMPLATE, FOLLOW_UP_QUESTIONS_TEMPLATE

class DocumentQAAgent:
    """
    Agent for answering questions based on document content.
    """
    def __init__(self, retriever=None, llm=None):
        self.retriever = retriever
        self.llm = llm or ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.document_sources = []
    
    def set_retriever(self, retriever) -> None:
        """Set the document retriever"""
        self.retriever = retriever
    
    def set_document_sources(self, sources: List[str]) -> None:
        """Set the list of document sources for reference in prompts"""
        self.document_sources = sources
    
    async def answer_question(self, query: str, chat_history: str = "", generate_followups: bool = False) -> Dict[str, Any]:
        """
        Answer a question based on retrieved document content.
        """
        if not self.retriever:
            return {
                "answer": "I don't have any documents to search. Please upload documents first.",
                "sources": [],
                "followup_questions": []
            }
        
        # Format system message with document sources
        system_message = DOCUMENT_QA_SYSTEM_TEMPLATE.format(
            document_sources=", ".join(self.document_sources) or "No documents specified"
        )
        
        # Retrieve relevant documents
        docs = self.retriever.get_relevant_documents(query)
        context = self._format_docs(docs)
        
        # Format inputs for the chain
        inputs = {
            "system_message": system_message,
            "context": context,
            "chat_history": chat_history or "No previous conversation.",
            "query": query
        }
        
        # Create a simple chain
        chain = (
            RunnablePassthrough()
            | self.llm.bind(stop=["\nHuman:"])
            | StrOutputParser()
        )
        
        # Process the question
        answer = chain.invoke(DOCUMENT_QA_PROMPT.format(**inputs))
        
        # Extract sources from the answer
        sources = self._extract_sources(answer, docs)
        
        result = {
            "answer": answer,
            "sources": sources,
            "followup_questions": []
        }
        
        # Generate follow-up questions if requested
        if generate_followups:
            followups = await self._generate_followup_questions(query, answer)
            result["followup_questions"] = followups
            
        return result
    
    async def _generate_followup_questions(self, query: str, answer: str) -> List[str]:
        """Generate follow-up questions based on the answer"""
        try:
            followup_input = {
                "query": query,
                "answer": answer
            }
            
            # Create a simple chain for follow-up questions
            followup_chain = (
                RunnablePassthrough()
                | self.llm.bind(temperature=0.7)
                | StrOutputParser()
            )
            
            # Generate and parse follow-up questions
            result = followup_chain.invoke(FOLLOW_UP_QUESTIONS_TEMPLATE.format(**followup_input))
            
            # Process the result into a list of questions
            questions = []
            for line in result.strip().split("\n"):
                line = line.strip()
                if line and (line[0].isdigit() or line[0] == "-"):
                    # Remove numbering/bullets and clean up the question
                    clean_q = line.split(".", 1)[-1].split(")", 1)[-1].strip() if "." in line or ")" in line else line[1:].strip()
                    questions.append(clean_q)
            
            return questions[:3]  # Limit to 3 questions
        except Exception as e:
            print(f"Error generating follow-up questions: {e}")
            return []
    
    def _format_docs(self, docs: List[Document]) -> str:
        """Format documents for inclusion in the prompt"""
        formatted_docs = []
        
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", f"Document {i+1}")
            page = doc.metadata.get("page", "")
            page_info = f", Page {page}" if page else ""
            
            formatted_docs.append(
                f"[DOCUMENT: {source}{page_info}]\n{doc.page_content}\n"
            )
        
        return "\n\n".join(formatted_docs)
    
    def _extract_sources(self, answer: str, docs: List[Document]) -> List[Dict[str, Any]]:
        """Extract and format sources from the answer and docs"""
        sources = []
        unique_sources = set()
        
        for doc in docs:
            source = doc.metadata.get("source", "")
            if source and source not in unique_sources:
                unique_sources.add(source)
                sources.append({
                    "name": source,
                    "type": self._get_document_type(source),
                    "metadata": doc.metadata
                })
        
        return sources
    
    def _get_document_type(self, source: str) -> str:
        """Determine document type from filename"""
        if not source:
            return "unknown"
            
        extension = os.path.splitext(source)[1].lower()
        
        if extension == '.pdf':
            return "pdf"
        elif extension == '.docx':
            return "docx"
        elif extension == '.csv':
            return "csv"
        elif extension in ['.xlsx', '.xls']:
            return "excel"
        elif extension == '.txt':
            return "text"
        else:
            return "document"