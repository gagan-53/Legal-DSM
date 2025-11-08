# OCR Implementation Guide

This guide explains how to add OCR support to the Legal-DSL LLM system for processing scanned/image-only PDFs.

## Current Status

The system currently:
- ✅ Detects pages where text extraction fails (returns None)
- ✅ Flags pages that require OCR in metadata
- ✅ Prevents crashes when processing scanned PDFs
- ❌ Does not automatically perform OCR

## Implementation Options

### Option 1: Tesseract + pdf2image (Recommended for Production)

**Install Dependencies:**
```bash
# System packages
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils

# Python packages
pip install pytesseract pdf2image Pillow
```

**Update `models/document_processor.py`:**

```python
import pytesseract
from pdf2image import convert_from_bytes

def _process_pdf_with_ocr(self, file_bytes: bytes) -> Dict:
    """Extract text from PDF with OCR fallback"""
    reader = PdfReader(io.BytesIO(file_bytes))
    
    pages = []
    full_text = ""
    current_offset = 0
    
    images = None  # Lazy load if needed
    
    for page_num, page in enumerate(reader.pages, 1):
        page_text = page.extract_text()
        
        # Try OCR if text extraction failed
        if page_text is None or not page_text.strip():
            try:
                # Convert this specific page to image and OCR
                if images is None:
                    images = convert_from_bytes(file_bytes, dpi=300)
                
                page_image = images[page_num - 1]
                page_text = pytesseract.image_to_string(page_image)
                
                if not page_text.strip():
                    page_text = f"[Page {page_num}: No text detected even after OCR]"
            except Exception as e:
                page_text = f"[Page {page_num}: OCR failed - {str(e)}]"
        
        pages.append({
            'page_number': page_num,
            'text': page_text,
            'char_offset_start': current_offset,
            'char_offset_end': current_offset + len(page_text),
            'ocr_applied': bool('[Page' not in page_text)
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
            'pages_with_ocr': sum(1 for p in pages if p.get('ocr_applied', False))
        }
    }
```

**Update Dockerfile.web:**
```dockerfile
# Add tesseract to Docker image
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir pytesseract pdf2image
```

### Option 2: Cloud OCR Services

**Google Cloud Vision:**
```python
from google.cloud import vision

def ocr_with_gcloud(image_bytes):
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)
    response = client.text_detection(image=image)
    return response.full_text_annotation.text
```

**AWS Textract:**
```python
import boto3

def ocr_with_textract(image_bytes):
    client = boto3.client('textract')
    response = client.detect_document_text(Document={'Bytes': image_bytes})
    return '\n'.join([block['Text'] for block in response['Blocks'] if block['BlockType'] == 'LINE'])
```

### Option 3: Pre-processing Pipeline

Instead of OCR at runtime, pre-process documents:

```bash
# Batch OCR script
for pdf in data/raw/*.pdf; do
    pdf2image "$pdf" | tesseract - "data/processed/$(basename $pdf .pdf)"
done
```

## Performance Considerations

### Tesseract Settings
```python
# Fast but lower quality
pytesseract.image_to_string(image, config='--psm 1 --oem 1')

# High quality but slower
pytesseract.image_to_string(image, config='--psm 1 --oem 3')
```

### Parallel Processing
```python
from concurrent.futures import ThreadPoolExecutor

def parallel_ocr(pages):
    with ThreadPoolExecutor(max_workers=4) as executor:
        return list(executor.map(pytesseract.image_to_string, pages))
```

### Caching
```python
import hashlib

def ocr_with_cache(image_bytes):
    cache_key = hashlib.md5(image_bytes).hexdigest()
    cache_file = f"cache/ocr/{cache_key}.txt"
    
    if os.path.exists(cache_file):
        return open(cache_file).read()
    
    text = pytesseract.image_to_string(image_bytes)
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, 'w') as f:
        f.write(text)
    
    return text
```

## Testing OCR

```python
# tests/test_ocr.py
def test_ocr_scanned_pdf():
    """Test OCR on scanned PDF"""
    processor = DocumentProcessor()
    
    # Use sample scanned PDF
    with open('tests/fixtures/scanned_contract.pdf', 'rb') as f:
        result = processor.process_document(file_bytes=f.read(), file_ext='.pdf')
    
    assert result['metadata']['pages_with_ocr'] > 0
    assert len(result['full_text']) > 100  # Should have extracted text
    assert 'No text detected' not in result['full_text']
```

## Production Checklist

- [ ] Install tesseract and dependencies
- [ ] Add OCR configuration to train_config.yaml
- [ ] Update DocumentProcessor with OCR fallback
- [ ] Add caching layer for OCR results
- [ ] Test with sample scanned PDFs
- [ ] Monitor OCR latency and accuracy
- [ ] Set up error handling and retry logic
- [ ] Document OCR quality settings
- [ ] Add OCR metrics to monitoring dashboard

## Monitoring

```python
# Track OCR usage
import prometheus_client

ocr_counter = prometheus_client.Counter(
    'ocr_pages_processed_total',
    'Total pages processed with OCR'
)

ocr_duration = prometheus_client.Histogram(
    'ocr_duration_seconds',
    'Time spent on OCR per page'
)
```

## Cost Estimation

| Service | Cost | Speed | Quality |
|---------|------|-------|---------|
| Tesseract (self-hosted) | Free | Medium | Good |
| Google Cloud Vision | $1.50/1000 pages | Fast | Excellent |
| AWS Textract | $1.50/1000 pages | Fast | Excellent |
| Azure Computer Vision | $1.00/1000 pages | Fast | Excellent |

## References

- [Tesseract Documentation](https://github.com/tesseract-ocr/tesseract)
- [pdf2image](https://github.com/Belval/pdf2image)
- [pytesseract](https://github.com/madmaze/pytesseract)
- [Google Cloud Vision](https://cloud.google.com/vision/docs)
- [AWS Textract](https://aws.amazon.com/textract/)
