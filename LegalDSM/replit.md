# Legal-DSL LLM - Replit Project

## Overview
Legal-DSL LLM is a comprehensive domain-specific language model for automated legal document processing. It provides clause extraction, named entity recognition, document summarization, and RAG-based question answering capabilities.

## Project Information
- **Language**: Python 3.11
- **Framework**: Streamlit (web interface)
- **Package Manager**: uv
- **Main Application**: `LegalDslLlm/app.py`

## Recent Changes
- **2025-11-06**: Initial Replit setup and bug fixes completed
  - Installed Python 3.11
  - Installed system dependencies (Tesseract OCR, Poppler)
  - Installed all Python dependencies via uv
  - Downloaded spaCy model (en_core_web_sm)
  - Downloaded NLTK data (punkt, stopwords, punkt_tab)
  - Configured Streamlit for port 5000 with proper CORS settings
  - Set up workflow to run Streamlit application
  - **Bug Fixes Applied**:
    - Fixed file path issues by implementing BASE_DIR for absolute path resolution
    - Added comprehensive error handling to prevent crashes
    - Protected all pages with null/empty data checks
    - Improved error messages with user-friendly descriptions
    - Added graceful degradation for failed model components
    - Wrapped document processing with individual try-catch blocks
    - Enhanced RAG Q&A error handling
    - Fixed research paper loading with proper path resolution
    - Added debug trace output for troubleshooting

## Architecture

### Directory Structure
```
LegalDslLlm/
├── app.py                    # Main Streamlit application
├── models/                   # Core ML modules
│   ├── document_processor.py    # PDF/DOCX parsing with OCR
│   ├── clause_extractor.py      # Clause detection & classification
│   ├── ner_extractor.py         # Named entity recognition
│   ├── summarizer.py            # Document summarization
│   └── rag_engine.py            # RAG with FAISS vector search
├── config/                   # Configuration files
├── evaluation/              # Evaluation metrics
├── training/                # Training scripts
├── tests/                   # Unit tests
└── research/               # Research artifacts
```

### Key Features
1. **Clause Extraction**: Identifies and classifies legal clauses (11 categories)
2. **Named Entity Recognition**: Extracts parties, dates, amounts, jurisdictions
3. **Document Summarization**: Abstractive and extractive summaries
4. **RAG Q&A**: Question answering with source attribution

### Technology Stack
- **ML Models**: Legal-BERT, LongT5, MPNet
- **Vector Search**: FAISS
- **OCR**: Tesseract + pdf2image
- **Processing**: PyTorch, Transformers, spaCy, NLTK

## Running the Application

The Streamlit application is configured to run automatically on port 5000. The workflow is already set up and will start when you open this Repl.

To manually restart the application:
1. Stop the current workflow
2. Run: `cd LegalDslLlm && uv run streamlit run app.py`

## Configuration

### Streamlit Configuration
- Port: 5000 (required for Replit webview)
- Host: 0.0.0.0
- CORS: Disabled for development
- Location: `LegalDslLlm/.streamlit/config.toml`

### System Dependencies
- **Tesseract OCR**: For scanned document processing
- **Poppler**: For PDF to image conversion

## Development Notes

### Package Management
This project uses `uv` for Python package management:
- Install packages: `uv add <package>`
- Sync dependencies: `uv sync`
- Run commands: `uv run <command>`

### Model Data
The application requires:
- spaCy English model: `en_core_web_sm` (already downloaded)
- NLTK data: punkt, stopwords, punkt_tab (already downloaded)

### Performance Considerations
- Uses PyTorch CPU version for compatibility
- Large ML models may take time to load on first use
- OCR processing is automatic but can be slow for large documents

## User Preferences
None specified yet.

## Supported Document Formats
- PDF (with automatic OCR for scanned pages)
- DOCX
- TXT

## Next Steps for Development
- Upload legal documents for processing
- Test clause extraction and NER features
- Experiment with document summarization
- Try RAG-based question answering

## Deployment
This project is configured to run in Replit's development environment. For production deployment:
- Use the deployment configuration tool
- Ensure all environment variables are set
- Consider resource limits for ML model inference
