# graph/workflow.py
from typing import Dict, List, Any, TypedDict, Optional
from langgraph.graph import StateGraph, END
from agents.document_qa import DocumentQAAgent
from utils.memory import ConversationManager

# Define state for our workflow
class DocumentQAState(TypedDict):
    query: str
    documents: List
    chat_history: str
    answer: Optional[str]
    sources: Optional[List]
    followup_questions: Optional[List]
    error: Optional[str]

class DocumentQAWorkflow:
    """
    Workflow for document question answering.
    """
    def __init__(self, document_qa_agent: DocumentQAAgent, memory_manager: ConversationManager):
        self.document_qa_agent = document_qa_agent
        self.memory_manager = memory_manager
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the workflow graph"""
        # Initialize the state graph
        workflow = StateGraph(DocumentQAState)
        
        # Define the nodes
        workflow.add_node("process_query", self._process_query)
        workflow.add_node("generate_answer", self._generate_answer)
        workflow.add_node("update_memory", self._update_memory)
        
        # Define the edges
        workflow.add_edge("process_query", "generate_answer")
        workflow.add_edge("generate_answer", "update_memory")
        workflow.add_edge("update_memory", END)
        
        # Set entry point
        workflow.set_entry_point("process_query")
        
        return workflow.compile()
    
    async def _process_query(self, state: DocumentQAState) -> DocumentQAState:
        """Process the user query"""
        try:
            # Validate inputs
            if not state.get("query"):
                return {
                    **state,
                    "error": "No query provided"
                }
                
            # Get chat history from memory manager
            chat_history = self.memory_manager.get_memory_variable_dict().get("chat_history", "")
            
            return {
                **state,
                "chat_history": chat_history
            }
        except Exception as e:
            return {
                **state,
                "error": f"Error processing query: {str(e)}"
            }
    
    async def _generate_answer(self, state: DocumentQAState) -> DocumentQAState:
        """Generate an answer using the document QA agent"""
        try:
            if state.get("error"):
                return state
                
            # Get answer from document QA agent
            result = await self.document_qa_agent.answer_question(
                query=state["query"],
                chat_history=state.get("chat_history", ""),
                generate_followups=True
            )
            
            return {
                **state,
                "answer": result["answer"],
                "sources": result["sources"],
                "followup_questions": result["followup_questions"]
            }
        except Exception as e:
            return {
                **state,
                "error": f"Error generating answer: {str(e)}",
                "answer": "I encountered an error while trying to answer your question."
            }
    
    async def _update_memory(self, state: DocumentQAState) -> DocumentQAState:
        """Update conversation memory with the new Q&A pair"""
        try:
            if state.get("error"):
                return state
                
            # Add the query and answer to memory
            self.memory_manager.add_user_message(state["query"])
            self.memory_manager.add_ai_message(state["answer"])
            
            # Trim memory if needed
            self.memory_manager.trim_if_needed()
            
            return state
        except Exception as e:
            return {
                **state,
                "error": f"Error updating memory: {str(e)}"
            }
    
    async def run(self, query: str) -> Dict[str, Any]:
        """Execute the workflow with a given query"""
        initial_state = {
            "query": query,
            "documents": [],
            "chat_history": "",
            "answer": None,
            "sources": None,
            "followup_questions": None,
            "error": None
        }
        
        result = await self.graph.ainvoke(initial_state)
        return result