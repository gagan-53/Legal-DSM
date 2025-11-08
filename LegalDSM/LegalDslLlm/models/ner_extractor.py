"""
Named Entity Recognition for Legal Documents
Extracts: parties, dates, amounts, jurisdictions
"""

import re
from typing import List, Dict
from datetime import datetime


class LegalNER:
    """Extract legal entities from documents"""
    
    ENTITY_TYPES = [
        'PARTY',
        'DATE',
        'AMOUNT',
        'JURISDICTION',
        'PERSON',
        'ORGANIZATION',
        'ADDRESS',
        'EMAIL',
        'PHONE'
    ]
    
    def __init__(self):
        self.patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> Dict:
        """Compile regex patterns for entity extraction"""
        return {
            'AMOUNT': [
                r'\$\s*[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|thousand|USD|EUR|GBP))?',
                r'(?:USD|EUR|GBP)\s*[\d,]+(?:\.\d{2})?',
                r'[\d,]+(?:\.\d{2})?\s*dollars?'
            ],
            'DATE': [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
                r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
                r'\b\d{4}-\d{2}-\d{2}\b'
            ],
            'EMAIL': [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            'PHONE': [
                r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
                r'\(\d{3}\)\s*\d{3}[-.\s]?\d{4}\b',
                r'\+\d{1,3}\s*\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'
            ],
            'JURISDICTION': [
                r'\b(?:State of|Commonwealth of)\s+[A-Z][a-z]+',
                r'\b(?:United States|USA|UK|European Union|EU)\b',
                r'\blaws?\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
                r'\b(?:California|New York|Texas|Delaware|Florida|Illinois)\b'
            ]
        }
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract all entities from text
        
        Returns:
            List of entity dictionaries with type, text, and offsets
        """
        entities = []
        
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entity = {
                        'entity_type': entity_type,
                        'text': match.group(),
                        'start_offset': match.start(),
                        'end_offset': match.end(),
                        'confidence': 0.85
                    }
                    entities.append(entity)
        
        entities.extend(self._extract_parties(text))
        entities.extend(self._extract_organizations(text))
        
        entities.sort(key=lambda x: x['start_offset'])
        
        return entities
    
    def _extract_parties(self, text: str) -> List[Dict]:
        """Extract party names (contract parties)"""
        parties = []
        
        party_patterns = [
            r'(?:between|by and among)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:and|,)',
            r'(?:Party|Parties):\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+\("(?:the\s+)?(?:Company|Client|Vendor|Contractor)"\)'
        ]
        
        for pattern in party_patterns:
            for match in re.finditer(pattern, text):
                parties.append({
                    'entity_type': 'PARTY',
                    'text': match.group(1),
                    'start_offset': match.start(1),
                    'end_offset': match.end(1),
                    'confidence': 0.8
                })
        
        return parties
    
    def _extract_organizations(self, text: str) -> List[Dict]:
        """Extract organization names"""
        orgs = []
        
        org_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Inc\.|LLC|Corp\.|Corporation|Ltd\.|Limited|LLP)\b',
            r'\b([A-Z][A-Z]+)\b(?:\s+Inc\.|\s+LLC|\s+Corp\.)?'
        ]
        
        for pattern in org_patterns:
            for match in re.finditer(pattern, text):
                orgs.append({
                    'entity_type': 'ORGANIZATION',
                    'text': match.group(),
                    'start_offset': match.start(),
                    'end_offset': match.end(),
                    'confidence': 0.75
                })
        
        return orgs
    
    def get_entity_summary(self, entities: List[Dict]) -> Dict:
        """Generate summary statistics for extracted entities"""
        summary = {
            'total_entities': len(entities),
            'by_type': {},
            'unique_entities': {}
        }
        
        for entity in entities:
            entity_type = entity['entity_type']
            summary['by_type'][entity_type] = summary['by_type'].get(entity_type, 0) + 1
            
            if entity_type not in summary['unique_entities']:
                summary['unique_entities'][entity_type] = set()
            summary['unique_entities'][entity_type].add(entity['text'])
        
        for entity_type in summary['unique_entities']:
            summary['unique_entities'][entity_type] = list(summary['unique_entities'][entity_type])
        
        return summary
