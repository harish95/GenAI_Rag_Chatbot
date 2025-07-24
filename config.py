import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
EXTRACTED_DIR = DATA_DIR / "extracted"
FAISS_DB_DIR = DATA_DIR / "faiss_db"

# Create directories if they don't exist
for dir_path in [DATA_DIR, UPLOADS_DIR, EXTRACTED_DIR, FAISS_DB_DIR]:
    dir_path.mkdir(exist_ok=True)

# Ollama settings
OLLAMA_MODEL = "llama3.2"  # Change to your preferred model
OLLAMA_BASE_URL = "http://localhost:11434"

# FAISS settings
FAISS_INDEX_FILE = FAISS_DB_DIR / "faiss_index.bin"
METADATA_FILE = FAISS_DB_DIR / "metadata.pkl"
EMBEDDING_DIM = 384
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Streamlit settings
PAGE_TITLE = "RAG Chatbot"
PAGE_ICON = "🤖"
