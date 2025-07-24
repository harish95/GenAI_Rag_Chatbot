import ollama
from typing import List, Dict
from vector_store_faiss import FAISSVectorStore
import config

class RAGChatbot:
    def __init__(self):
        self.vector_store = FAISSVectorStore()
        self.client = ollama.Client(host=config.OLLAMA_BASE_URL)
    
    def generate_response(self, query: str, chat_history: List = None) -> str:
        """Generate response using RAG with FAISS"""
        
        # Search for relevant documents
        relevant_docs = self.vector_store.search(query, n_results=5)
        
        if not relevant_docs:
            return "I don't have any relevant information to answer your question. Please make sure you have uploaded and processed your documents."
        
        # Prepare context
        context_parts = []
        for doc in relevant_docs:
            source = doc['metadata']['source']
            content = doc['content']
            similarity = doc.get('similarity', 0.0)
            context_parts.append(f"Source: {source} (Relevance: {similarity:.2f})\nContent: {content}\n")
        
        context = "\n---\n".join(context_parts)
        
        # Create prompt
        prompt = f"""Based on the following context, please answer the user's question. If the answer is not available in the context, please say so.

Context:
{context}

Question: {query}

Answer:"""
        
        try:
            # Generate response using Ollama
            response = self.client.generate(
                model=config.OLLAMA_MODEL,
                prompt=prompt,
                stream=False
            )
            
            return response['response']
            
        except Exception as e:
            return f"Error generating response: {str(e)}. Please make sure Ollama is running and the model '{config.OLLAMA_MODEL}' is available."
    
    def get_vector_store_info(self) -> Dict:
        """Get vector store information"""
        return self.vector_store.get_collection_info()
