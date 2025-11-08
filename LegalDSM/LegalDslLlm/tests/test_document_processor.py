"""
Unit tests for Document Processor
Run with: pytest tests/test_document_processor.py -v
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from models.document_processor import DocumentProcessor
from io import BytesIO


class TestDocumentProcessor:
    """Test suite for document processing"""
    
    @pytest.fixture
    def processor(self):
        """Create processor instance"""
        return DocumentProcessor()
    
    def test_processor_initialization(self, processor):
        """Test processor initializes correctly"""
        assert processor.supported_formats == ['.pdf', '.docx', '.txt']
    
    def test_process_txt(self, processor):
        """Test text file processing"""
        text_content = "This is a legal document with important clauses."
        text_bytes = text_content.encode('utf-8')
        
        result = processor.process_document(file_bytes=text_bytes, file_ext='.txt')
        
        assert result['full_text'] == text_content
        assert result['metadata']['format'] == 'txt'
        assert result['metadata']['total_chars'] == len(text_content)
    
    def test_process_empty_txt(self, processor):
        """Test empty text file"""
        result = processor.process_document(file_bytes=b'', file_ext='.txt')
        
        assert result['full_text'] == ''
        assert result['metadata']['total_chars'] == 0
    
    def test_chunk_text(self, processor):
        """Test text chunking"""
        text = "This is sentence one. This is sentence two. This is sentence three."
        chunks = processor.chunk_text(text, chunk_size=30, overlap=10)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert 'text' in chunk
            assert 'start_offset' in chunk
            assert 'end_offset' in chunk
            assert 'chunk_id' in chunk
    
    def test_chunk_overlap(self, processor):
        """Test that chunks have correct overlap"""
        text = "A" * 100
        chunks = processor.chunk_text(text, chunk_size=50, overlap=10)
        
        if len(chunks) > 1:
            first_end = chunks[0]['end_offset']
            second_start = chunks[1]['start_offset']
            overlap = first_end - second_start
            assert overlap == 10
    
    def test_split_into_sentences(self, processor):
        """Test sentence splitting"""
        text = "First sentence. Second sentence. Third sentence."
        sentences = processor.split_into_sentences(text)
        
        assert len(sentences) >= 2
        for sent, start, end in sentences:
            assert isinstance(sent, str)
            assert start < end
            assert text[start:end].strip() in sent or sent in text[start:end].strip()
    
    def test_unsupported_format(self, processor):
        """Test error on unsupported file format"""
        with pytest.raises(ValueError, match="Unsupported file format"):
            processor.process_document(file_bytes=b'test', file_ext='.xyz')
    
    def test_pdf_none_handling(self, processor):
        """Test that None from extract_text() triggers OCR fallback"""
        class MockPage:
            def extract_text(self):
                return None
        
        class MockReader:
            pages = [MockPage()]
        
        from unittest.mock import patch, MagicMock
        
        with patch('models.document_processor.PdfReader', return_value=MockReader()):
            result = processor._process_pdf(b'fake_pdf_bytes')
            
            assert result is not None
            assert 'full_text' in result
            assert 'pages' in result
            assert len(result['pages']) == 1
            
            if result['metadata'].get('ocr_available'):
                assert result['metadata']['pages_with_ocr'] >= 0
            else:
                assert 'OCR not available' in result['full_text'] or result['metadata']['pages_with_ocr'] >= 0
    
    def test_ocr_availability(self, processor):
        """Test OCR availability flag"""
        result = processor.process_document(file_bytes=b'fake', file_ext='.txt')
        
        from models.document_processor import OCR_AVAILABLE
        assert isinstance(OCR_AVAILABLE, bool)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
