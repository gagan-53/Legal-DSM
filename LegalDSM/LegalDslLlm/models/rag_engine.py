"""
RAG (Retrieval-Augmented Generation) Engine
Provides grounded question answering with source attribution
"""

import numpy as np
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RAGEngine:
    """
    RAG system for legal document Q&A
    - Retrieves relevant chunks using vector search (TF-IDF for MVP, FAISS for production)
    - Generates answers grounded to specific clauses/chunks
    """
    
    def __init__(self, vector_dims: int = 384):
        self.vector_dims = vector_dims
        self.vectorizer = TfidfVectorizer(max_features=vector_dims, ngram_range=(1, 2))
        self.chunks = []
        self.chunk_vectors = None
        self.document_text = ""
    
    def index_document(self, text: str, chunks: List[Dict] = None):
        """
        Index document for retrieval
        
        Args:
            text: Full document text
            chunks: Pre-chunked document (optional)
        """
        self.document_text = text
        
        if chunks is None:
            chunks = self._create_chunks(text, chunk_size=512, overlap=50)
        
        self.chunks = chunks
        
        chunk_texts = [chunk['text'] for chunk in chunks]
        
        if chunk_texts:
            self.chunk_vectors = self.vectorizer.fit_transform(chunk_texts)
    
    def _create_chunks(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[Dict]:
        """Create overlapping chunks from text"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            if end < len(text):
                last_period = text.rfind('.', start, end)
                if last_period > start:
                    end = last_period + 1
            
            chunk_text = text[start:end]
            chunks.append({
                'text': chunk_text,
                'start_offset': start,
                'end_offset': end,
                'chunk_id': len(chunks)
            })
            
            start = end - overlap if end < len(text) else end
        
        return chunks
    
    def query(self, question: str, top_k: int = 3) -> Dict:
        """
        Answer a question about the document
        
        Args:
            question: User's question
            top_k: Number of relevant chunks to retrieve
            
        Returns:
            Dict with answer, source chunks, and confidence
        """
        if not self.chunks or self.chunk_vectors is None:
            return {
                'answer': 'No document indexed. Please upload a document first.',
                'sources': [],
                'confidence': 0.0
            }
        
        query_vector = self.vectorizer.transform([question])
        
        similarities = cosine_similarity(query_vector, self.chunk_vectors)[0]
        
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        relevant_chunks = []
        for idx in top_indices:
            if similarities[idx] > 0.1:
                relevant_chunks.append({
                    'text': self.chunks[idx]['text'],
                    'chunk_id': self.chunks[idx]['chunk_id'],
                    'start_offset': self.chunks[idx]['start_offset'],
                    'end_offset': self.chunks[idx]['end_offset'],
                    'similarity_score': float(similarities[idx])
                })
        
        if not relevant_chunks:
            return {
                'answer': 'No relevant information found in the document.',
                'sources': [],
                'confidence': 0.0
            }
        
        answer = self._generate_answer(question, relevant_chunks)
        
        confidence = float(np.mean([chunk['similarity_score'] for chunk in relevant_chunks]))
        
        return {
            'answer': answer,
            'sources': relevant_chunks,
            'confidence': confidence,
            'num_sources_used': len(relevant_chunks)
        }
    
    def _generate_answer(self, question: str, chunks: List[Dict]) -> str:
        """
        Generate answer from retrieved chunks
        For production: Replace with fine-tuned LLM (Llama 3, GPT-4, etc.)
        """
        if not chunks:
            return "No relevant information found."
        
        best_chunk = chunks[0]
        
        answer = f"Based on the document: {best_chunk['text'][:300]}"
        
        if len(best_chunk['text']) > 300:
            answer += "..."
        
        if len(chunks) > 1:
            answer += f"\n\n(Additional relevant context found in {len(chunks)-1} other sections)"
        
        return answer
    
    def get_chunk_by_offset(self, offset: int) -> Dict:
        """Get chunk containing a specific character offset"""
        for chunk in self.chunks:
            if chunk['start_offset'] <= offset <= chunk['end_offset']:
                return chunk
        return None
    
    def highlight_answer_location(self, answer_data: Dict) -> List[Dict]:
        """
        Get highlightable locations for answer sources
        
        Returns:
            List of highlighting spans with offsets
        """
        highlights = []
        
        for source in answer_data.get('sources', []):
            highlights.append({
                'start_offset': source['start_offset'],
                'end_offset': source['end_offset'],
                'chunk_id': source['chunk_id'],
                'score': source['similarity_score'],
                'text_preview': source['text'][:100] + '...'
            })
        
        return highlights
