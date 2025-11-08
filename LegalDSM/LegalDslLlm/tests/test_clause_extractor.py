"""
Unit tests for Clause Extractor
Run with: pytest tests/test_clause_extractor.py -v
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from models.clause_extractor import ClauseExtractor


class TestClauseExtractor:
    """Test suite for clause extraction and classification"""
    
    @pytest.fixture
    def extractor(self):
        """Create extractor instance"""
        return ClauseExtractor(confidence_threshold=0.5)
    
    @pytest.fixture
    def sample_indemnity_text(self):
        """Sample indemnity clause"""
        return """The Contractor shall indemnify and hold harmless the Company 
        from and against any and all claims, damages, liabilities, costs and 
        expenses arising from or related to the Contractor's performance under 
        this Agreement."""
    
    @pytest.fixture
    def sample_termination_text(self):
        """Sample termination clause"""
        return """Either party may terminate this Agreement upon thirty (30) 
        days' written notice to the other party. Termination shall be effective 
        upon expiration of the notice period."""
    
    def test_extractor_initialization(self, extractor):
        """Test extractor initializes correctly"""
        assert extractor.confidence_threshold == 0.5
        assert len(extractor.CLAUSE_TYPES) == 15
        assert 'indemnity' in extractor.CLAUSE_TYPES
        assert 'termination' in extractor.CLAUSE_TYPES
    
    def test_classify_indemnity_clause(self, extractor, sample_indemnity_text):
        """Test classification of indemnity clause"""
        clause_type, confidence = extractor._classify_clause(sample_indemnity_text)
        assert clause_type == 'indemnity'
        assert confidence > 0.5
    
    def test_classify_termination_clause(self, extractor, sample_termination_text):
        """Test classification of termination clause"""
        clause_type, confidence = extractor._classify_clause(sample_termination_text)
        assert clause_type == 'termination'
        assert confidence > 0.5
    
    def test_extract_clauses_full_document(self, extractor):
        """Test extraction from full document"""
        document = """
        This Agreement is made on January 1, 2024.
        
        The Contractor shall indemnify the Company from all claims.
        
        Either party may terminate this Agreement with 30 days notice.
        
        All disputes shall be resolved through binding arbitration.
        """
        
        clauses = extractor.extract_clauses(document)
        
        assert len(clauses) > 0
        clause_types = [c['clause_type'] for c in clauses]
        assert 'indemnity' in clause_types or 'termination' in clause_types
    
    def test_clause_with_offsets(self, extractor, sample_indemnity_text):
        """Test that clauses include character offsets"""
        clauses = extractor.extract_clauses(sample_indemnity_text)
        
        if clauses:
            clause = clauses[0]
            assert 'char_offset_start' in clause
            assert 'char_offset_end' in clause
            assert clause['char_offset_start'] >= 0
            assert clause['char_offset_end'] > clause['char_offset_start']
    
    def test_confidence_threshold_filtering(self):
        """Test that low-confidence clauses are filtered"""
        extractor = ClauseExtractor(confidence_threshold=0.9)
        
        generic_text = "This is a generic sentence without legal terms."
        clauses = extractor.extract_clauses(generic_text)
        
        for clause in clauses:
            assert clause['confidence'] >= 0.5
    
    def test_clause_statistics(self, extractor):
        """Test clause statistics generation"""
        document = """
        The Contractor shall indemnify the Company.
        Either party may terminate this Agreement.
        Disputes shall be resolved through arbitration.
        """
        
        clauses = extractor.extract_clauses(document)
        stats = extractor.get_clause_statistics(clauses)
        
        assert 'total_clauses' in stats
        assert 'by_type' in stats
        assert 'avg_confidence' in stats
        assert stats['total_clauses'] == len(clauses)
    
    def test_empty_document(self, extractor):
        """Test handling of empty document"""
        clauses = extractor.extract_clauses("")
        assert clauses == []
    
    def test_very_short_text(self, extractor):
        """Test handling of very short text"""
        clauses = extractor.extract_clauses("Short.")
        assert isinstance(clauses, list)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
