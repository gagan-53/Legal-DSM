"""
Unit tests for NER Extractor
Run with: pytest tests/test_ner_extractor.py -v
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from models.ner_extractor import LegalNER


class TestLegalNER:
    """Test suite for legal named entity recognition"""
    
    @pytest.fixture
    def ner(self):
        """Create NER instance"""
        return LegalNER()
    
    @pytest.fixture
    def sample_contract_text(self):
        """Sample contract text with entities"""
        return """This Agreement is entered into on January 15, 2024 between 
        Acme Corporation ("the Company") and Beta LLC ("the Contractor"). 
        The contract value is $1,000,000 USD. This Agreement shall be governed 
        by the laws of the State of Delaware. Contact: john@acme.com, 
        phone: 555-123-4567."""
    
    def test_ner_initialization(self, ner):
        """Test NER initializes correctly"""
        assert len(ner.ENTITY_TYPES) > 0
        assert 'PARTY' in ner.ENTITY_TYPES
        assert 'DATE' in ner.ENTITY_TYPES
        assert 'AMOUNT' in ner.ENTITY_TYPES
    
    def test_extract_dates(self, ner):
        """Test date extraction"""
        text = "This Agreement is effective January 15, 2024."
        entities = ner.extract_entities(text)
        
        date_entities = [e for e in entities if e['entity_type'] == 'DATE']
        assert len(date_entities) > 0
        assert 'January 15, 2024' in [e['text'] for e in date_entities]
    
    def test_extract_amounts(self, ner):
        """Test monetary amount extraction"""
        text = "The fee is $50,000 USD paid monthly."
        entities = ner.extract_entities(text)
        
        amount_entities = [e for e in entities if e['entity_type'] == 'AMOUNT']
        assert len(amount_entities) > 0
    
    def test_extract_email(self, ner):
        """Test email extraction"""
        text = "Contact us at legal@company.com for questions."
        entities = ner.extract_entities(text)
        
        email_entities = [e for e in entities if e['entity_type'] == 'EMAIL']
        assert len(email_entities) > 0
        assert 'legal@company.com' in [e['text'] for e in email_entities]
    
    def test_extract_phone(self, ner):
        """Test phone number extraction"""
        text = "Call us at 555-123-4567 or (555) 987-6543."
        entities = ner.extract_entities(text)
        
        phone_entities = [e for e in entities if e['entity_type'] == 'PHONE']
        assert len(phone_entities) > 0
    
    def test_extract_jurisdiction(self, ner):
        """Test jurisdiction extraction"""
        text = "Governed by the laws of California and the State of New York."
        entities = ner.extract_entities(text)
        
        jurisdiction_entities = [e for e in entities if e['entity_type'] == 'JURISDICTION']
        assert len(jurisdiction_entities) > 0
    
    def test_extract_all_entities(self, ner, sample_contract_text):
        """Test extraction of all entity types"""
        entities = ner.extract_entities(sample_contract_text)
        
        assert len(entities) > 0
        entity_types = set(e['entity_type'] for e in entities)
        
        assert 'DATE' in entity_types
        assert 'AMOUNT' in entity_types
    
    def test_entity_offsets(self, ner, sample_contract_text):
        """Test that entities have correct offsets"""
        entities = ner.extract_entities(sample_contract_text)
        
        for entity in entities:
            assert 'start_offset' in entity
            assert 'end_offset' in entity
            assert entity['start_offset'] < entity['end_offset']
            
            extracted_text = sample_contract_text[entity['start_offset']:entity['end_offset']]
            assert entity['text'] in extracted_text or extracted_text in entity['text']
    
    def test_entity_summary(self, ner, sample_contract_text):
        """Test entity summary generation"""
        entities = ner.extract_entities(sample_contract_text)
        summary = ner.get_entity_summary(entities)
        
        assert 'total_entities' in summary
        assert 'by_type' in summary
        assert 'unique_entities' in summary
        assert summary['total_entities'] == len(entities)
    
    def test_empty_text(self, ner):
        """Test handling of empty text"""
        entities = ner.extract_entities("")
        assert entities == []
    
    def test_confidence_scores(self, ner, sample_contract_text):
        """Test that entities have confidence scores"""
        entities = ner.extract_entities(sample_contract_text)
        
        for entity in entities:
            assert 'confidence' in entity
            assert 0.0 <= entity['confidence'] <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
