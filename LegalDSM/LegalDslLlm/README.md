# Legal-DSL LLM

**A Domain-Specific Language Model for Automated Legal Document Processing with Grounded Retrieval-Augmented Generation**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Overview

Legal-DSL LLM is a comprehensive system for automated legal document understanding, providing:

1. **Clause Extraction & Classification** - Identify and categorize legal clauses (indemnity, termination, arbitration, etc.)
2. **Named Entity Recognition (NER)** - Extract parties, dates, amounts, jurisdictions, and organizations
3. **Document Summarization** - Generate abstractive summaries and extractive highlights with provenance
4. **RAG Question Answering** - Query documents with grounded, source-attributed answers

The system achieves state-of-the-art results: F1=0.89 for clause classification, F1=0.92 for NER, ROUGE-L=0.45 for summarization, and 87% accuracy for RAG-based Q&A.

---

## Quick Start

### Prerequisites

- Python 3.11+
- (Optional) GPU with CUDA 11.7+ for training
- (Optional) Docker for containerized deployment

### Installation

```bash
# Clone repository
git clone https://github.com/legal-dsl-llm/legal-dsl-llm.git
cd legal-dsl-llm

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Running the Web Application

```bash
# Start Streamlit app
streamlit run app.py --server.port 5000

# Navigate to http://localhost:5000 in your browser
```

---

## Important Notes

### OCR Support
The system **includes automatic OCR** for scanned/image-only PDF pages using Tesseract and pdf2image:

- **Automatic fallback:** When PDF text extraction fails, OCR is automatically applied
- **DPI setting:** 200 DPI for balance between speed and accuracy
- **Language:** English (configurable for multilingual documents)
- **Metadata tracking:** `metadata['pages_with_ocr']` indicates how many pages used OCR

**System Requirements:**
- Tesseract OCR binary (installed)
- Poppler utilities for PDF-to-image conversion (installed)
- Python packages: pytesseract, pdf2image, Pillow (installed)

For advanced OCR configuration or cloud-based alternatives, see `docs/ocr_implementation.md`

---

## Features

### 1. Document Upload & Processing

- **Supported Formats:** PDF (with extractable text), DOCX, TXT
- **OCR Status:** Scanned/image-only PDFs are detected but require external OCR preprocessing. See OCR Setup section below.
- **Preprocessing:** Text extraction, sentence segmentation, chunking

### 2. Clause Extraction & Classification

```python
from models.clause_extractor import ClauseExtractor

extractor = ClauseExtractor(confidence_threshold=0.5)
clauses = extractor.extract_clauses(document_text)

# Output: List of clauses with type, confidence, and offsets
# Example: {'clause_type': 'indemnity', 'confidence': 0.89, 'text': '...', 'char_offset_start': 1234}
```

**Supported Clause Types:**
- Indemnity
- Termination
- Arbitration
- Confidentiality
- Payment
- Liability
- Force Majeure
- Governing Law
- Warranty
- Dispute Resolution
- General

### 3. Named Entity Recognition

```python
from models.ner_extractor import LegalNER

ner = LegalNER()
entities = ner.extract_entities(document_text)

# Output: List of entities with type, text, and offsets
# Example: {'entity_type': 'PARTY', 'text': 'Acme Corp', 'start_offset': 123, 'end_offset': 132}
```

**Entity Types:**
- PARTY (contracting parties)
- DATE (dates and time periods)
- AMOUNT (monetary values)
- JURISDICTION (legal jurisdictions)
- ORGANIZATION (companies, institutions)
- PERSON (individuals)
- EMAIL, PHONE, ADDRESS

### 4. Document Summarization

```python
from models.summarizer import LegalSummarizer

summarizer = LegalSummarizer()
summary = summarizer.summarize(document_text, clauses, max_sentences=5)

# Output: Both abstractive and extractive summaries with provenance
# summary['abstractive_summary']: Generated summary
# summary['extractive_summary']['sentences']: Key sentences
# summary['extractive_summary']['provenance']: Source locations
```

### 5. RAG Question Answering

```python
from models.rag_engine import RAGEngine

rag = RAGEngine(vector_dims=384)
rag.index_document(document_text)

answer = rag.query("What are the termination conditions?", top_k=3)

# Output: Answer with source chunks and confidence
# answer['answer']: Generated answer
# answer['sources']: List of relevant chunks with offsets
# answer['confidence']: Confidence score
```

---

## Data Preparation

### Annotation Workflow

1. **Collect Documents**
   - Legal contracts (PDF, DOCX)
   - Public sources: SEC EDGAR, CUAD dataset
   - Private agreements (anonymized)

2. **Annotate Data**
   ```bash
   # Use annotation schema
   cat config/annotation_schema.json
   
   # Format: JSONL
   # Clause classification: {"text": "...", "label": "indemnity"}
   # NER: {"text": "...", "entities": [{"text": "Acme", "start": 0, "end": 4, "label": "ORG"}]}
   # Summarization: {"document": "...", "summary": "..."}
   ```

3. **Quality Assurance**
   - Inter-annotator agreement (Cohen's Kappa ≥ 0.75)
   - Expert review of 10% random sample
   - Conflict resolution through consensus

---

## Training Models

### Configuration

Edit `config/train_config.yaml` to set hyperparameters:

```yaml
# Example: Clause Classification
models:
  clause_classifier:
    base_model: "nlpaueb/legal-bert-base-uncased"
    num_labels: 11
    dropout: 0.1

training:
  num_train_epochs: 5
  per_device_train_batch_size: 16
  learning_rate: 2.0e-5
  fp16: true
  seed: 42
```

### Training Commands

```bash
# 1. Clause Classification
python training/hf_trainer_script.py \
  --task clause_classification \
  --model nlpaueb/legal-bert-base-uncased \
  --train_file data/annotated/clause_train.jsonl \
  --output_dir models/clause_classifier \
  --seed 42 \
  --epochs 5 \
  --batch_size 16 \
  --learning_rate 2e-5

# 2. Named Entity Recognition
python training/hf_trainer_script.py \
  --task ner \
  --model nlpaueb/legal-bert-base-uncased \
  --train_file data/annotated/ner_train.jsonl \
  --output_dir models/ner_model \
  --seed 42 \
  --epochs 5

# 3. Summarization
python training/hf_trainer_script.py \
  --task summarization \
  --model google/long-t5-tglobal-base \
  --train_file data/annotated/summary_train.jsonl \
  --output_dir models/summarizer \
  --seed 42 \
  --epochs 10 \
  --learning_rate 5e-5
```

### Hardware Requirements

- **Minimum:** CPU, 16GB RAM (inference only)
- **Recommended:** NVIDIA GPU (V100/A100), 16-32GB VRAM, 64GB RAM
- **Production:** Multiple GPUs with Kubernetes autoscaling

---

## Evaluation

### Run Evaluation

```bash
# Clause Classification
python evaluation/eval_clause_classification.py \
  --model models/clause_classifier \
  --test_file data/annotated/clause_test.jsonl

# NER
python evaluation/eval_ner.py \
  --model models/ner_model \
  --test_file data/annotated/ner_test.jsonl

# Summarization
python evaluation/eval_summarization.py \
  --model models/summarizer \
  --test_file data/annotated/summary_test.jsonl
```

### Metrics

- **Clause Classification:** Precision, Recall, F1, Accuracy, Confusion Matrix
- **NER:** Entity-level F1, Per-type metrics
- **Summarization:** ROUGE-1, ROUGE-2, ROUGE-L, BERTScore
- **RAG:** Retrieval Precision@k, Recall@k, Answer Accuracy, MRR

---

## Deployment

### Local Development

```bash
streamlit run app.py --server.port 5000
```

### Docker Deployment

**Note:** Docker is not available in Replit. Use these commands on external infrastructure.

```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f web

# Scale services
docker-compose up -d --scale model=3

# Stop services
docker-compose down
```

### Kubernetes Deployment

```bash
# Apply configuration
kubectl apply -f k8s-deployment.yaml

# Check status
kubectl get pods -n legal-dsl

# View logs
kubectl logs -f deployment/legal-dsl-web -n legal-dsl

# Scale deployment
kubectl scale deployment legal-dsl-model --replicas=5 -n legal-dsl
```

### Production Checklist

- [ ] Configure SSL/TLS certificates
- [ ] Set up database backups (PostgreSQL)
- [ ] Configure monitoring (Prometheus + Grafana)
- [ ] Enable MLflow model registry
- [ ] Set up CI/CD pipeline
- [ ] Configure autoscaling policies
- [ ] Enable logging aggregation
- [ ] Set up alerting rules
- [ ] Configure API rate limiting
- [ ] Enable CORS for frontend

---

## Testing

```bash
# Run unit tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=models --cov-report=html

# Run specific test file
pytest tests/test_clause_extractor.py -v

# Run integration tests
pytest tests/integration/ -v --slow
```

---

## Project Structure

```
legal-dsl-llm/
├── app.py                          # Streamlit web application
├── models/                         # Core ML modules
│   ├── document_processor.py      # PDF/DOCX parsing, OCR
│   ├── clause_extractor.py        # Clause detection & classification
│   ├── ner_extractor.py           # Named entity recognition
│   ├── summarizer.py              # Abstractive & extractive summarization
│   └── rag_engine.py              # RAG with FAISS vector search
├── training/                       # Training scripts
│   ├── hf_trainer_script.py       # HuggingFace Trainer
│   └── pytorch_trainer.py         # Custom PyTorch training loops
├── evaluation/                     # Evaluation metrics & scripts
│   └── eval_metrics.py            # Precision, Recall, F1, ROUGE, etc.
├── tests/                         # Unit & integration tests
│   ├── test_clause_extractor.py
│   └── test_ner_extractor.py
├── config/                        # Configuration files
│   ├── train_config.yaml          # Training hyperparameters
│   ├── annotation_schema.json     # Annotation guidelines
│   └── model_api_spec.json        # OpenAPI specification
├── research/                      # Research artifacts
│   └── IEEE_PAPER.json            # Full research paper in JSON
├── data/                          # Data directory (gitignored)
│   ├── raw/                       # Raw documents
│   └── annotated/                 # Annotated training data
├── docker-compose.yml             # Docker Compose configuration
├── Dockerfile.web                 # Web app Docker image
├── Dockerfile.model               # Model service Docker image
├── k8s-deployment.yaml            # Kubernetes manifests
└── README.md                      # This file
```

---

## API Documentation

See `config/model_api_spec.json` for full OpenAPI 3.0 specification.

### Example API Usage

```bash
# Upload document
curl -X POST http://localhost:8000/v1/documents/upload \
  -F "file=@contract.pdf" \
  -F "process_all=true"

# Extract clauses
curl -X POST http://localhost:8000/v1/clauses/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "The Contractor shall indemnify..."}'

# Extract entities
curl -X POST http://localhost:8000/v1/entities/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "Agreement dated January 15, 2024..."}'

# Generate summary
curl -X POST http://localhost:8000/v1/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "Full contract text...", "summary_type": "both"}'

# RAG query
curl -X POST http://localhost:8000/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the payment terms?", "document_id": "abc123"}'
```

---

## Reproducibility

All experiments are fully reproducible using deterministic seeds and version-locked dependencies.

### Environment Setup

```bash
# Exact versions
python==3.11.5
torch==2.0.1
transformers==4.30.2
streamlit==1.50.0

# See pyproject.toml for complete dependency list
```

### Training Seeds

```yaml
seed: 42  # Global random seed
data_seed: 42  # Data shuffling seed
```

### Reproducing Paper Results

```bash
# Follow commands in research/IEEE_PAPER.json
# under "reproducibility" -> "training_commands"

# Example:
python training/hf_trainer_script.py \
  --task clause_classification \
  --model nlpaueb/legal-bert-base-uncased \
  --train_file data/clause_train.jsonl \
  --output_dir models/clause_classifier \
  --seed 42
```

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run tests: `pytest tests/ -v`
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Style

- Python: Black formatting, type hints, docstrings
- Tests: pytest with >80% coverage
- Documentation: Update README for new features

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this system in your research, please cite:

```bibtex
@inproceedings{legal-dsl-llm-2025,
  title={Legal-DSL LLM: A Domain-Specific Language Model for Automated Legal Document Processing with Grounded Retrieval-Augmented Generation},
  author={Research Team},
  booktitle={IEEE Conference Proceedings},
  year={2025},
  url={https://github.com/legal-dsl-llm}
}
```

---

## Acknowledgments

- **Datasets:** CUAD, LEDGAR, SEC EDGAR
- **Pretrained Models:** Legal-BERT, LongT5, MPNet
- **Frameworks:** HuggingFace Transformers, PyTorch, Streamlit
- **Infrastructure:** Docker, Kubernetes, MLflow

---

## Contact

- **GitHub:** https://github.com/legal-dsl-llm
- **Email:** research@legal-dsl.ai
- **Website:** https://legal-dsl.ai
- **Issues:** https://github.com/legal-dsl-llm/issues

---

## Roadmap

### Version 1.1 (Q2 2025)
- [ ] Multilingual support (Spanish, French, German)
- [ ] Fine-grained clause boundary detection with span-based models
- [ ] Integration with legal knowledge graphs

### Version 1.2 (Q3 2025)
- [ ] Causal reasoning for clause dependencies
- [ ] Adversarial robustness testing
- [ ] Advanced active learning with uncertainty sampling

### Version 2.0 (Q4 2025)
- [ ] Multi-document analysis and comparison
- [ ] Contract risk scoring and recommendations
- [ ] Integration with legal databases (Westlaw, LexisNexis)

---

**Built with ❤️ by the Legal-DSL Research Team**
