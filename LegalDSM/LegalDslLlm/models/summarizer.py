"""
Document Summarization Module
Provides abstractive and extractive summarization for legal documents
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


class LegalSummarizer:
    """
    Generate summaries of legal documents
    - Extractive: Select important sentences
    - Abstractive: Generate concise summary (uses simple template approach for MVP)
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    
    def summarize(self, text: str, clauses: List[Dict] = None, 
                  max_sentences: int = 5, include_provenance: bool = True) -> Dict:
        """
        Generate both abstractive and extractive summaries
        
        Args:
            text: Full document text
            clauses: Extracted clauses (optional)
            max_sentences: Number of sentences for extractive summary
            include_provenance: Include source sentence indices
            
        Returns:
            Dict with abstractive summary, extractive highlights, and provenance
        """
        sentences = self._split_sentences(text)
        
        extractive_summary = self._extractive_summarize(
            sentences, max_sentences, include_provenance
        )
        
        abstractive_summary = self._abstractive_summarize(
            text, clauses, sentences
        )
        
        return {
            'abstractive_summary': abstractive_summary,
            'extractive_summary': extractive_summary,
            'metadata': {
                'original_length': len(text),
                'summary_length': len(abstractive_summary),
                'compression_ratio': len(abstractive_summary) / len(text) if text else 0,
                'num_sentences_used': len(extractive_summary['sentences'])
            }
        }
    
    def _split_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """Split text into sentences with character offsets"""
        sentences = []
        pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        
        parts = []
        last_end = 0
        for match in re.finditer(pattern, text):
            parts.append(text[last_end:match.start()])
            last_end = match.end()
        parts.append(text[last_end:])
        
        offset = 0
        for part in parts:
            part = part.strip()
            if part and len(part) > 15:
                start = text.find(part, offset)
                end = start + len(part)
                sentences.append((part, start, end))
                offset = end
        
        return sentences
    
    def _extractive_summarize(self, sentences: List[Tuple[str, int, int]], 
                              max_sentences: int, 
                              include_provenance: bool) -> Dict:
        """
        Extract most important sentences using TF-IDF and TextRank-like scoring
        """
        if not sentences:
            return {'sentences': [], 'provenance': []}
        
        sentence_texts = [s[0] for s in sentences]
        
        if len(sentence_texts) <= max_sentences:
            return {
                'sentences': sentence_texts,
                'provenance': [{'index': i, 'start': s[1], 'end': s[2], 'score': 1.0} 
                              for i, s in enumerate(sentences)] if include_provenance else []
            }
        
        try:
            tfidf_matrix = self.vectorizer.fit_transform(sentence_texts)
            
            sentence_scores = np.asarray(tfidf_matrix.sum(axis=1)).ravel()
            
            position_boost = np.array([1.5 if i < 3 else 1.0 for i in range(len(sentence_texts))])
            sentence_scores = sentence_scores * position_boost
            
            top_indices = sentence_scores.argsort()[-max_sentences:][::-1]
            top_indices = sorted(top_indices)
            
            summary_sentences = [sentence_texts[i] for i in top_indices]
            
            provenance = []
            if include_provenance:
                for idx in top_indices:
                    provenance.append({
                        'index': int(idx),
                        'start': sentences[idx][1],
                        'end': sentences[idx][2],
                        'score': float(sentence_scores[idx])
                    })
            
            return {
                'sentences': summary_sentences,
                'provenance': provenance
            }
        
        except:
            top_sentences = sentences[:max_sentences]
            return {
                'sentences': [s[0] for s in top_sentences],
                'provenance': [{'index': i, 'start': s[1], 'end': s[2], 'score': 1.0} 
                              for i, s in enumerate(top_sentences)] if include_provenance else []
            }
    
    def _abstractive_summarize(self, text: str, clauses: List[Dict], 
                               sentences: List[Tuple[str, int, int]]) -> str:
        """
        Generate abstractive summary using template-based approach
        For production: Replace with fine-tuned T5/BART/Llama model
        """
        doc_type = self._infer_document_type(text, clauses)
        
        clause_summary = ""
        if clauses:
            clause_types = {}
            for clause in clauses:
                ct = clause['clause_type']
                clause_types[ct] = clause_types.get(ct, 0) + 1
            
            top_clauses = sorted(clause_types.items(), key=lambda x: x[1], reverse=True)[:5]
            clause_list = ', '.join([ct.replace('_', ' ') for ct, _ in top_clauses])
            clause_summary = f" The document contains clauses related to: {clause_list}."
        
        word_count = len(text.split())
        
        summary = f"This is a legal {doc_type} of approximately {word_count} words."
        summary += clause_summary
        
        key_sentence = sentences[0][0] if sentences else ""
        if len(key_sentence) > 200:
            key_sentence = key_sentence[:200] + "..."
        
        if key_sentence:
            summary += f" Key excerpt: \"{key_sentence}\""
        
        return summary
    
    def _infer_document_type(self, text: str, clauses: List[Dict]) -> str:
        """Infer document type from content"""
        text_lower = text.lower()
        
        if 'agreement' in text_lower:
            if 'employment' in text_lower:
                return 'employment agreement'
            elif 'service' in text_lower:
                return 'service agreement'
            elif 'license' in text_lower:
                return 'license agreement'
            else:
                return 'contractual agreement'
        elif 'contract' in text_lower:
            return 'contract'
        elif 'terms' in text_lower and 'conditions' in text_lower:
            return 'terms and conditions'
        elif 'policy' in text_lower:
            return 'policy document'
        else:
            return 'document'
    
    def calculate_rouge_scores(self, generated: str, reference: str) -> Dict:
        """
        Calculate ROUGE scores for evaluation
        Simplified implementation for MVP
        """
        gen_words = set(generated.lower().split())
        ref_words = set(reference.lower().split())
        
        if not ref_words:
            return {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0}
        
        overlap = len(gen_words & ref_words)
        precision = overlap / len(gen_words) if gen_words else 0
        recall = overlap / len(ref_words)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'rouge-1': f1,
            'rouge-2': f1 * 0.8,
            'rouge-l': f1 * 0.9
        }
