import streamlit as st
import os
import tempfile
from pathlib import Path

from data_processor import DataProcessor
from vector_store_faiss import FAISSVectorStore
from chatbot import RAGChatbot
import config

# Page configuration
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout="wide"
)

# Initialize session state
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = RAGChatbot()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store_ready' not in st.session_state:
    st.session_state.vector_store_ready = False

def check_vector_store_status():
    """Check if vector store has data"""
    info = st.session_state.chatbot.get_vector_store_info()
    st.session_state.vector_store_ready = info['exists'] and info['count'] > 0
    return info

def process_files(excel_file, zip_file):
    """Process uploaded files and create vector store"""
    
    processor = DataProcessor()
    vector_store = FAISSVectorStore()
    
    with st.spinner("Processing files..."):
        all_documents = []
        
        # Process Excel file (websites)
        if excel_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_excel:
                tmp_excel.write(excel_file.getbuffer())
                tmp_excel_path = tmp_excel.name
            
            st.info("Extracting URLs from Excel and scraping websites...")
            web_docs = processor.process_excel_file(tmp_excel_path)
            all_documents.extend(web_docs)
            os.unlink(tmp_excel_path)
        
        # Process ZIP file (PDFs)
        if zip_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_zip:
                tmp_zip.write(zip_file.getbuffer())
                tmp_zip_path = tmp_zip.name
            
            st.info("Extracting and processing PDFs...")
            pdf_docs = processor.process_zip_file(tmp_zip_path)
            all_documents.extend(pdf_docs)
            os.unlink(tmp_zip_path)
        
        if all_documents:
            # Chunk documents
            st.info("Chunking documents...")
            chunked_docs = processor.chunk_documents(all_documents)
            
            # Create vector store
            st.info("Creating FAISS vector index...")
            vector_store.add_documents(chunked_docs)
            
            st.success(f"Successfully processed {len(all_documents)} documents into {len(chunked_docs)} chunks using FAISS!")
            st.session_state.vector_store_ready = True
            
        else:
            st.error("No documents were processed. Please check your files.")

def main():
    # Create two columns
    col1, col2 = st.columns([1, 6])  # Adjust ratio as needed

    with col1:
        st.image("piramal_logo.png", width=150)  # adjust width if needed

    with col2:
        st.markdown(
        """
        <h1 style='display: inline-block; vertical-align: middle; padding-top: 50px;'>
        PDFs and Wblinks QnA Chatbot based on Local model
        </h1>
        """,
        unsafe_allow_html=True
        )
    st.markdown("Upload your Excel file (with website links) and ZIP file (with PDFs) to start chatting!")
    
    # Sidebar for file management
    with st.sidebar:
        st.header("📁 Data Management")
        
        # Check current status
        info = check_vector_store_status()
        
        if st.session_state.vector_store_ready:
            st.success(f"✅ FAISS index ready with {info['count']} vectors")
            st.info(f"📊 Vector dimension: {info.get('dimension', 'N/A')}")
        else:
            st.warning("⚠️ No data loaded")
        
        st.subheader("Upload Files")
        
        excel_file = st.file_uploader(
            "Upload Excel file with website links",
            type=['xlsx', 'xls'],
            help="Excel file should contain a column with website URLs"
        )
        
        zip_file = st.file_uploader(
            "Upload ZIP file with PDFs",
            type=['zip'],
            help="ZIP file containing PDF documents"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Process Files", disabled=not (excel_file or zip_file)):
                process_files(excel_file, zip_file)
        
        with col2:
            if st.button("🗑️ Delete Data", disabled=not st.session_state.vector_store_ready):
                if st.session_state.chatbot.vector_store.delete_collection():
                    st.success("FAISS data deleted successfully!")
                    st.session_state.vector_store_ready = False
                    st.session_state.chat_history = []
                    st.rerun()
                else:
                    st.error("Error deleting data")
    
    # Main chat interface
    if st.session_state.vector_store_ready:
        st.subheader("💬 Chat with your documents")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.chatbot.generate_response(prompt)
                st.write(response)
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Clear chat button
        if st.button("🧹 Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    else:
        st.info("👆 Please upload and process your files using the sidebar to start chatting!")
        
        st.subheader("ℹ️ How to use:")
        st.markdown("""
        1. **Upload Excel file**: Should contain website URLs in a column named 'url', 'link', or 'website'
        2. **Upload ZIP file**: Should contain PDF documents
        3. **Click 'Process Files'**: This will extract content and create the FAISS knowledge base
        4. **Start chatting**: Ask questions about your documents
        5. **Delete and retrain**: Use the 'Delete Data' button to clear everything and start over
        
        **Note**: This version uses FAISS for vector storage, avoiding ChromaDB dependency issues.
        """)

if __name__ == "__main__":
    main()
