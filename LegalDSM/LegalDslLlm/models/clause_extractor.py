"""
Clause Extraction and Classification Module
Implements rule-based and ML-based approaches for legal clause detection
"""

import re
import numpy as np
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ClauseExtractor:
    """
    Extract and classify legal clauses from documents
    Supports: indemnity, termination, arbitration, confidentiality, payment,
             liability, force majeure, governing law, etc.
    """
    
    CLAUSE_TYPES = [
        'indemnity',
        'termination',
        'arbitration',
        'confidentiality',
        'payment',
        'liability',
        'force_majeure',
        'governing_law',
        'warranty',
        'dispute_resolution',
        'intellectual_property',
        'limitation_of_liability',
        'data_protection',
        'non_compete',
        'assignment'
    ]
    
    CLAUSE_PATTERNS = {
        'indemnity': [
            r'indemnif(?:y|ication|ied)',
            r'hold harmless',
            r'defend.*against.*claim',
            r'reimburse.*for.*loss'
        ],
        'termination': [
            r'terminat(?:e|ion|ed)',
            r'cancel(?:lation)?',
            r'end.*agreement',
            r'notice.*period',
            r'termination.*clause'
        ],
        'arbitration': [
            r'arbitrat(?:e|ion|or)',
            r'binding arbitration',
            r'arbitration.*proceeding',
            r'dispute.*arbitrator'
        ],
        'confidentiality': [
            r'confidential(?:ity)?',
            r'non-disclosure',
            r'proprietary.*information',
            r'trade.*secret',
            r'confidential.*information'
        ],
        'payment': [
            r'payment.*term',
            r'invoice',
            r'compensation',
            r'fee.*schedule',
            r'remuneration',
            r'consideration.*paid'
        ],
        'liability': [
            r'liab(?:le|ility)',
            r'responsible.*for',
            r'liable.*damage',
            r'assume.*liability'
        ],
        'force_majeure': [
            r'force majeure',
            r'act.*of.*god',
            r'beyond.*reasonable.*control',
            r'unforeseen.*circumstance'
        ],
        'governing_law': [
            r'governing.*law',
            r'jurisdiction',
            r'laws.*of.*[A-Z][a-z]+',
            r'governed.*by.*law',
            r'subject.*to.*law'
        ],
        'warranty': [
            r'warrant(?:y|ies|ed)',
            r'represent.*and.*warrant',
            r'guarantee',
            r'assurance'
        ],
        'dispute_resolution': [
            r'dispute.*resolution',
            r'mediation',
            r'negotiation.*dispute',
            r'resolution.*process'
        ]
    }
    
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 3))
    
    def extract_clauses(self, text: str, doc_structure: Dict = None) -> List[Dict]:
        """
        Extract and classify clauses from legal document text
        
        Args:
            text: Full document text
            doc_structure: Document structure from DocumentProcessor
            
        Returns:
            List of extracted clauses with classifications and confidence scores
        """
        sentences = self._split_into_sentences(text)
        clauses = []
        
        for sent_idx, (sentence, start_offset, end_offset) in enumerate(sentences):
            clause_type, confidence = self._classify_clause(sentence)
            
            if confidence >= self.confidence_threshold:
                clause = {
                    'clause_id': f'clause_{sent_idx}',
                    'text': sentence,
                    'clause_type': clause_type,
                    'confidence': float(confidence),
                    'char_offset_start': start_offset,
                    'char_offset_end': end_offset,
                    'page_number': self._get_page_number(start_offset, doc_structure) if doc_structure else None,
                    'needs_review': confidence < 0.7
                }
                clauses.append(clause)
        
        clauses = self._merge_adjacent_clauses(clauses)
        
        return clauses
    
    def _split_into_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """Split text into sentences with offsets"""
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
            if part and len(part) > 20:
                start = text.find(part, offset)
                end = start + len(part)
                sentences.append((part, start, end))
                offset = end
        
        return sentences
    
    def _classify_clause(self, text: str) -> Tuple[str, float]:
        """
        Classify a clause using pattern matching and keyword scoring
        
        Returns:
            (clause_type, confidence_score)
        """
        text_lower = text.lower()
        scores = {}
        
        for clause_type, patterns in self.CLAUSE_PATTERNS.items():
            score = 0.0
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    score += 0.3
            
            scores[clause_type] = min(score, 1.0)
        
        if not scores or max(scores.values()) == 0:
            return 'general', 0.3
        
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type]
        
        return best_type, confidence
    
    def _merge_adjacent_clauses(self, clauses: List[Dict]) -> List[Dict]:
        """Merge adjacent clauses of the same type"""
        if not clauses:
            return []
        
        merged = []
        current = clauses[0].copy()
        
        for clause in clauses[1:]:
            if (clause['clause_type'] == current['clause_type'] and
                clause['char_offset_start'] - current['char_offset_end'] < 100):
                current['text'] += ' ' + clause['text']
                current['char_offset_end'] = clause['char_offset_end']
                current['confidence'] = max(current['confidence'], clause['confidence'])
            else:
                merged.append(current)
                current = clause.copy()
        
        merged.append(current)
        return merged
    
    def _get_page_number(self, offset: int, doc_structure: Dict) -> int:
        """Get page number for a character offset"""
        if not doc_structure or 'pages' not in doc_structure:
            return None
        
        for page in doc_structure['pages']:
            if page['char_offset_start'] <= offset <= page['char_offset_end']:
                return page['page_number']
        
        return None
    
    def get_clause_statistics(self, clauses: List[Dict]) -> Dict:
        """Generate statistics about extracted clauses"""
        stats = {
            'total_clauses': len(clauses),
            'by_type': {},
            'avg_confidence': 0.0,
            'needs_review_count': 0
        }
        
        for clause in clauses:
            clause_type = clause['clause_type']
            stats['by_type'][clause_type] = stats['by_type'].get(clause_type, 0) + 1
            stats['avg_confidence'] += clause['confidence']
            if clause.get('needs_review'):
                stats['needs_review_count'] += 1
        
        if clauses:
            stats['avg_confidence'] /= len(clauses)
        
        return stats
