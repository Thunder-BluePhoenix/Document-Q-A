# app.py
import streamlit as st
import asyncio
import os
import tempfile
from typing import Dict, List, Any
import plotly.express as px
import pandas as pd

from main import (
    process_document,
    ask_question,
    clear_documents,
    clear_chat_history,
    get_document_stats,
    get_chat_history
)

# Page configuration
st.set_page_config(
    page_title="Document Q&A System",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set up session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "document_stats" not in st.session_state:
    st.session_state.document_stats = {"document_count": 0, "sources": []}

if "processing" not in st.session_state:
    st.session_state.processing = False

# Custom styling
st.markdown("""
<style>
    .chat-message {
        padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem;
        display: flex; align-items: center;
    }
    .chat-message.user {
        background-color: #2b313e;
    }
    .chat-message.assistant {
        background-color: #475063;
    }
    .chat-message .avatar {
        width: 20%;
    }
    .chat-message .avatar img {
        max-width: 78px;
        max-height: 78px;
        border-radius: 50%;
        object-fit: cover;
    }
    .chat-message .message {
        width: 80%;
        padding-left: 1rem;
    }
    .source-item {
        padding: 0.5rem;
        border-radius: 0.3rem;
        background-color: #4a5568;
        margin-bottom: 0.5rem;
    }
    .stButton > button {
        width: 100%;
    }
    .followup-button {
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("📑 Document Q&A System")
st.markdown("Upload your documents and start asking questions about them.")

# Sidebar
with st.sidebar:
    st.header("📂 Document Management")
    
    # Document upload
    uploaded_files = st.file_uploader(
        "Upload your documents",
        type=["pdf", "docx", "csv", "xlsx", "xls", "txt"],
        accept_multiple_files=True
    )
    
    # Upload button
    upload_pressed = st.button("📤 Process Documents", type="primary")
    
    # Clear documents button
    clear_docs_pressed = st.button("🗑️ Clear All Documents")
    
    # Display document stats
    st.header("📊 Document Statistics")
    doc_count = st.session_state.document_stats.get("document_count", 0)
    sources = st.session_state.document_stats.get("sources", [])
    
    st.metric("Documents Loaded", doc_count)
    
    if doc_count > 0:
        st.subheader("Loaded Documents:")
        for source in sources:
            st.markdown(f"- {source}")
    
    # Clear chat history
    st.header("💬 Chat Options")
    clear_chat_pressed = st.button("🧹 Clear Chat History")
    
    # About section
    st.header("ℹ️ About")
    st.markdown("""
    This application allows you to:
    1. Upload multiple document types (PDF, DOCX, CSV, Excel, etc.)
    2. Ask questions about your documents
    3. Get accurate answers with source citations
    4. Follow up with additional questions
    """)
    
    st.markdown("---")
    st.markdown("Built with Streamlit, LangChain, and OpenAI")

# Main area - divided into statistics/visualization and chat
col1, col2 = st.columns([1, 3])

with col1:
    if doc_count > 0:
        st.subheader("Document Insights")
        
        # Extract stats for visualization
        metadata = st.session_state.document_stats.get("metadata", {})
        chunk_count = metadata.get("chunk_count", 0)
        
        # Display metrics
        st.metric("Total Chunks", chunk_count)
        
        # Create a pie chart for document types
        doc_extensions = [os.path.splitext(src)[1][1:].upper() for src in sources]
        if doc_extensions:
            doc_types = pd.DataFrame({
                "Type": doc_extensions,
                "Count": [1] * len(doc_extensions)
            })
            doc_types = doc_types.groupby("Type").sum().reset_index()
            
            fig = px.pie(
                doc_types,
                values="Count",
                names="Type",
                title="Document Types",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Sample questions
    st.subheader("🔍 Sample Questions")
    st.markdown("""
    Once documents are loaded, try asking:
    - "What are the main topics in these documents?"
    - "Summarize the key points from [document name]"
    - "What data trends can you identify in the CSV files?"
    - "Compare the information between documents"
    """)

with col2:
    # Process document upload
    if upload_pressed and uploaded_files:
        st.session_state.processing = True
        process_results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, file in enumerate(uploaded_files):
            # Update progress
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing {file.name}... ({i+1}/{len(uploaded_files)})")
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Process the document
                result = asyncio.run(process_document(file.getvalue(), file.name))
                process_results.append(result)
            except Exception as e:
                process_results.append({
                    "status": "error",
                    "message": f"Error processing {file.name}: {str(e)}"
                })
            finally:
                # Remove temp file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        success_count = sum(1 for r in process_results if r["status"] == "success")
        error_count = len(process_results) - success_count
        
        if success_count > 0:
            st.success(f"✅ Successfully processed {success_count} document(s)")
        
        if error_count > 0:
            st.error(f"❌ Failed to process {error_count} document(s)")
            for result in process_results:
                if result["status"] == "error":
                    st.warning(result["message"])
        
        # Update document stats
        st.session_state.document_stats = get_document_stats()
        st.session_state.processing = False
        
        # Force a rerun to update the UI
        st.rerun()
    
    # Clear documents
    if clear_docs_pressed:
        clear_result = clear_documents()
        st.session_state.document_stats = get_document_stats()
        st.success(clear_result["message"])
        
        # Force a rerun to update the UI
        st.rerun()
    
    # Clear chat history
    if clear_chat_pressed:
        clear_result = clear_chat_history()
        st.session_state.chat_history = []
        st.success(clear_result["message"])
        
        # Force a rerun to update the UI
        st.rerun()
    
    # Chat interface
    st.subheader("💬 Ask about your documents")
    
    # Function to display chat messages
    def display_chat_message(role, content, sources=None, followup_questions=None):
        avatar_img = "👤" if role == "user" else "🤖"
        message_alignment = "flex-start" if role == "assistant" else "flex-end"
        
        message_html = f"""
        <div class="chat-message {role}" style="justify-content: {message_alignment}">
            <div class="avatar">
                <div style="font-size: 3rem; text-align: center;">{avatar_img}</div>
            </div>
            <div class="message">
                {content}
        """
        
        # Add sources if available
        if role == "assistant" and sources:
            message_html += "<div style='margin-top: 1rem; font-size: 0.9rem;'><strong>Sources:</strong><br>"
            for source in sources:
                source_name = source.get("name", "Unknown")
                source_type = source.get("type", "document").upper()
                message_html += f"<div class='source-item'>{source_name} ({source_type})</div>"
            message_html += "</div>"
        
        message_html += "</div></div>"
        
        st.markdown(message_html, unsafe_allow_html=True)
        
        # Display follow-up question buttons if available
        if role == "assistant" and followup_questions:
            cols = st.columns(len(followup_questions))
            for i, question in enumerate(followup_questions):
                if cols[i].button(question, key=f"followup_{i}_{hash(question)}"):
                    return question
        
        return None
    
    # Display chat history
    for message in st.session_state.chat_history:
        followup = display_chat_message(
            message["role"],
            message["content"],
            message.get("sources"),
            message.get("followup_questions")
        )
        
        # If a follow-up button was clicked, use it as the next query
        if followup:
            st.session_state.user_input = followup
            st.rerun()
    
    # Query input
    user_input = st.chat_input("Ask a question about your documents...")
    
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Display user message
        display_chat_message("user", user_input)
        
        # Check if documents are loaded
        if st.session_state.document_stats.get("document_count", 0) == 0:
            answer = "Please upload documents first before asking questions."
            sources = []
            followup_questions = []
        else:
            # Process the query
            with st.spinner("Thinking..."):
                try:
                    response = asyncio.run(ask_question(user_input))
                    
                    if response["status"] == "success":
                        answer = response["answer"]
                        sources = response["sources"]
                        followup_questions = response["followup_questions"]
                    else:
                        answer = response["message"]
                        sources = []
                        followup_questions = []
                except Exception as e:
                    answer = f"Error processing question: {str(e)}"
                    sources = []
                    followup_questions = []
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "followup_questions": followup_questions
        })
        
        # Display assistant response
        display_chat_message("assistant", answer, sources, followup_questions)