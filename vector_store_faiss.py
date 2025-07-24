import faiss
import numpy as np
import pickle
from typing import List, Dict, Optional
import config
import shutil
import hashlib
import re
import math
from collections import Counter

class SimpleEmbedding:
    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions
        self.vocab = {}
        self.vocab_size = 0
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Simple text preprocessing"""
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        words = text.split()
        words = [word for word in words if len(word) > 2]
        return words
    
    def _build_vocabulary(self, texts: List[str]):
        """Build vocabulary from texts"""
        all_words = []
        for text in texts:
            words = self._preprocess_text(text)
            all_words.extend(words)
        
        word_counts = Counter(all_words)
        most_common = word_counts.most_common(min(5000, len(word_counts)))
        
        self.vocab = {word: idx for idx, (word, _) in enumerate(most_common)}
        self.vocab_size = len(self.vocab)
    
    def _text_to_vector(self, text: str) -> np.ndarray:
        """Convert text to vector"""
        words = self._preprocess_text(text)
        word_counts = Counter(words)
        total_words = len(words)
        
        vector = np.zeros(self.dimensions, dtype=np.float32)
        
        for word, count in word_counts.items():
            if word in self.vocab:
                tf = count / total_words if total_words > 0 else 0
                vocab_idx = self.vocab[word]
                vector_idx = vocab_idx % self.dimensions
                vector[vector_idx] += tf
        
        # Normalize vector
        magnitude = np.linalg.norm(vector)
        if magnitude > 0:
            vector = vector / magnitude
        
        return vector
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Convert texts to embeddings"""
        if not self.vocab:
            self._build_vocabulary(texts)
        
        embeddings = []
        for text in texts:
            embedding = self._text_to_vector(text)
            embeddings.append(embedding)
        
        return np.array(embeddings)

class FAISSVectorStore:
    def __init__(self):
        self.embedding_model = SimpleEmbedding(config.EMBEDDING_DIM)
        self.index = None
        self.metadata = []
        self.is_trained = False
        
    def _initialize_index(self):
        """Initialize FAISS index"""
        # Use IndexFlatIP for cosine similarity
        self.index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
        
    def _save_index(self):
        """Save FAISS index and metadata to disk"""
        if self.index is not None:
            faiss.write_index(self.index, str(config.FAISS_INDEX_FILE))
        
        with open(config.METADATA_FILE, 'wb') as f:
            pickle.dump({
                'metadata': self.metadata,
                'vocab': self.embedding_model.vocab,
                'is_trained': self.is_trained
            }, f)
    
    def _load_index(self) -> bool:
        """Load FAISS index and metadata from disk"""
        try:
            if config.FAISS_INDEX_FILE.exists() and config.METADATA_FILE.exists():
                self.index = faiss.read_index(str(config.FAISS_INDEX_FILE))
                
                with open(config.METADATA_FILE, 'rb') as f:
                    data = pickle.load(f)
                    self.metadata = data['metadata']
                    self.embedding_model.vocab = data['vocab']
                    self.is_trained = data['is_trained']
                
                return True
        except Exception as e:
            print(f"Error loading index: {e}")
        
        return False
    
    def add_documents(self, documents: List[Dict]):
        """Add documents to FAISS vector store"""
        if not self._load_index():
            self._initialize_index()
        
        # Prepare texts and metadata
        texts = [doc['content'] for doc in documents]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store metadata
        for i, doc in enumerate(documents):
            self.metadata.append({
                'content': doc['content'],
                'source': doc['source'],
                'type': doc['type'],
                'chunk_id': doc.get('chunk_id', 0)
            })
        
        self.is_trained = True
        self._save_index()
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for relevant documents"""
        if not self._load_index() or self.index is None:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, min(n_results, len(self.metadata)))
        
        # Format results
        documents = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1 and idx < len(self.metadata):  # Valid index
                documents.append({
                    'content': self.metadata[idx]['content'],
                    'metadata': {
                        'source': self.metadata[idx]['source'],
                        'type': self.metadata[idx]['type'],
                        'chunk_id': self.metadata[idx]['chunk_id']
                    },
                    'distance': 1.0 - score,  # Convert similarity to distance
                    'similarity': score
                })
        
        return documents
    
    def get_collection_info(self) -> Dict:
        """Get information about the vector store"""
        if self._load_index() and self.index is not None:
            return {
                'count': self.index.ntotal,
                'exists': True,
                'dimension': config.EMBEDDING_DIM
            }
        return {'count': 0, 'exists': False, 'dimension': config.EMBEDDING_DIM}
    
    def delete_collection(self) -> bool:
        """Delete the entire vector store"""
        try:
            # Remove files
            if config.FAISS_INDEX_FILE.exists():
                config.FAISS_INDEX_FILE.unlink()
            if config.METADATA_FILE.exists():
                config.METADATA_FILE.unlink()
            
            # Reset state
            self.index = None
            self.metadata = []
            self.is_trained = False
            self.embedding_model = SimpleEmbedding(config.EMBEDDING_DIM)
            
            return True
        except Exception as e:
            print(f"Error deleting collection: {e}")
            return False
