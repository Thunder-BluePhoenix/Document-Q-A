# utils/memory.py
from typing import List, Dict, Any, Optional
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

class ConversationManager:
    """
    Manages conversation history for document Q&A sessions.
    """
    def __init__(self, memory_key: str = "chat_history", max_token_limit: int = 4000):
        self.memory = ConversationBufferMemory(
            memory_key=memory_key,
            return_messages=True,
            input_key="query",
            output_key="answer"
        )
        self.max_token_limit = max_token_limit
        self.system_message = None
    
    def add_system_message(self, content: str) -> None:
        """Add a system message to the conversation"""
        self.system_message = SystemMessage(content=content)
    
    def add_user_message(self, message: str) -> None:
        """Add a user message to the conversation"""
        self.memory.chat_memory.add_user_message(message)
    
    def add_ai_message(self, message: str) -> None:
        """Add an AI message to the conversation"""
        self.memory.chat_memory.add_ai_message(message)
    
    def get_chat_history(self) -> List[Dict[str, Any]]:
        """Get formatted chat history for display"""
        messages = self.memory.chat_memory.messages
        
        history = []
        for message in messages:
            if isinstance(message, HumanMessage):
                history.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                history.append({"role": "assistant", "content": message.content})
            elif isinstance(message, SystemMessage):
                history.append({"role": "system", "content": message.content})
        
        return history
    
    def get_memory_variable_dict(self) -> Dict[str, Any]:
        """Get the memory variables as a dict for LangChain"""
        return self.memory.load_memory_variables({})
    
    def clear(self) -> None:
        """Clear the conversation history"""
        self.memory.clear()
        
    def _approximate_token_count(self, text: str) -> int:
        """Approximate token count - very rough estimate"""
        return len(text) // 4
    
    def trim_if_needed(self) -> None:
        """Trim conversation history if it exceeds token limit"""
        # Simplified token management - in production you'd want to use tiktoken
        messages = self.memory.chat_memory.messages
        
        # Calculate total tokens (approximate)
        total_tokens = sum(self._approximate_token_count(m.content) for m in messages)
        
        # If we're under the limit, no need to trim
        if total_tokens <= self.max_token_limit:
            return
            
        # Remove oldest messages until we're under the limit
        while total_tokens > self.max_token_limit and len(messages) > 2:
            # Remove the two oldest messages (user and assistant pair)
            removed1 = messages.pop(0)
            total_tokens -= self._approximate_token_count(removed1.content)
            
            # Make sure we have another message to remove
            if messages:
                removed2 = messages.pop(0)
                total_tokens -= self._approximate_token_count(removed2.content)
        
        # Update the memory with trimmed messages
        self.memory.chat_memory.messages = messages