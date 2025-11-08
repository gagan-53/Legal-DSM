"""
Document Processing Module
Handles PDF, DOCX parsing and OCR for scanned documents
"""

import os
import re
from typing import Dict, List, Tuple
from PyPDF2 import PdfReader
from docx import Document
import io

try:
    import pytesseract
    from pdf2image import convert_from_bytes
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


class DocumentProcessor:
    """Process legal documents (PDF, DOCX) and extract text"""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.txt']
    
    def process_document(self, file_path: str = None, file_bytes: bytes = None, file_ext: str = None) -> Dict:
        """
        Process a document and extract structured text with metadata
        
        Args:
            file_path: Path to document file
            file_bytes: Raw bytes of document
            file_ext: File extension if using bytes
            
        Returns:
            Dict with text, metadata, and page information
        """
        if file_path:
            ext = os.path.splitext(file_path)[1].lower()
            with open(file_path, 'rb') as f:
                file_bytes = f.read()
        else:
            ext = file_ext
        
        if ext == '.pdf':
            return self._process_pdf(file_bytes)
        elif ext == '.docx':
            return self._process_docx(file_bytes)
        elif ext == '.txt':
            return self._process_txt(file_bytes)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    def _process_pdf(self, file_bytes: bytes) -> Dict:
        """Extract text from PDF with page numbers and offsets, using OCR fallback"""
        reader = PdfReader(io.BytesIO(file_bytes))
        
        pages = []
        full_text = ""
        current_offset = 0
        pdf_images = None
        ocr_used_count = 0
        
        for page_num, page in enumerate(reader.pages, 1):
            page_text = page.extract_text()
            ocr_applied = False
            
            if page_text is None or not page_text.strip():
                if OCR_AVAILABLE:
                    try:
                        if pdf_images is None:
                            pdf_images = convert_from_bytes(
                                file_bytes,
                                dpi=200,
                                fmt='jpeg',
                                thread_count=2
                            )
                        
                        if page_num <= len(pdf_images):
                            page_image = pdf_images[page_num - 1]
                            page_text = pytesseract.image_to_string(page_image, lang='eng')
                            ocr_applied = True
                            ocr_used_count += 1
                            
                            if not page_text.strip():
                                page_text = f"[Page {page_num}: No text detected even after OCR]"
                        else:
                            page_text = f"[Page {page_num}: Image conversion failed]"
                    except Exception as e:
                        page_text = f"[Page {page_num}: OCR error - {str(e)[:50]}]"
                else:
                    page_text = f"[Page {page_num}: No text extracted - OCR not available. Install pytesseract and pdf2image.]"
            
            pages.append({
                'page_number': page_num,
                'text': page_text,
                'char_offset_start': current_offset,
                'char_offset_end': current_offset + len(page_text),
                'ocr_applied': ocr_applied
            })
            full_text += page_text + "\n"
            current_offset += len(page_text) + 1
        
        return {
            'full_text': full_text,
            'pages': pages,
            'num_pages': len(pages),
            'metadata': {
                'format': 'pdf',
                'total_chars': len(full_text),
                'pages_with_ocr': ocr_used_count,
                'ocr_available': OCR_AVAILABLE
            }
        }
    
    def _process_docx(self, file_bytes: bytes) -> Dict:
        """Extract text from DOCX with paragraph tracking"""
        doc = Document(io.BytesIO(file_bytes))
        
        paragraphs = []
        full_text = ""
        current_offset = 0
        
        for para_num, para in enumerate(doc.paragraphs, 1):
            para_text = para.text
            if para_text.strip():
                paragraphs.append({
                    'paragraph_number': para_num,
                    'text': para_text,
                    'char_offset_start': current_offset,
                    'char_offset_end': current_offset + len(para_text)
                })
                full_text += para_text + "\n"
                current_offset += len(para_text) + 1
        
        return {
            'full_text': full_text,
            'paragraphs': paragraphs,
            'num_paragraphs': len(paragraphs),
            'metadata': {
                'format': 'docx',
                'total_chars': len(full_text)
            }
        }
    
    def _process_txt(self, file_bytes: bytes) -> Dict:
        """Extract text from plain text file"""
        text = file_bytes.decode('utf-8', errors='ignore')
        
        return {
            'full_text': text,
            'metadata': {
                'format': 'txt',
                'total_chars': len(text)
            }
        }
    
    def split_into_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Split text into sentences with character offsets
        
        Returns:
            List of (sentence, start_offset, end_offset)
        """
        sentences = []
        pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        parts = re.split(pattern, text)
        
        offset = 0
        for part in parts:
            part = part.strip()
            if part:
                start = text.find(part, offset)
                end = start + len(part)
                sentences.append((part, start, end))
                offset = end
        
        return sentences
    
    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[Dict]:
        """
        Chunk text into overlapping segments for RAG
        
        Args:
            text: Full document text
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks
            
        Returns:
            List of chunk dictionaries with text and offsets
        """
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
