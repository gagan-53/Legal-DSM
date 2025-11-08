# Next Actions: Quick Start Guide

This document provides the exact commands to go from **repository clone → train → serve → test**.

---

## Prerequisites

- Python 3.11+
- Git + Git LFS (for large model files)
- (Optional) NVIDIA GPU with CUDA 11.7+ for training
- (Optional) Docker for containerized deployment

---

## Step 1: Clone Repository & Setup Environment

```bash
# Clone the repository
git clone https://github.com/legal-dsl-llm/legal-dsl-llm.git
cd legal-dsl-llm

# (Optional) Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

---

## Step 2: Prepare Data (Optional - for training)

```bash
# Create data directories
mkdir -p data/raw data/processed data/annotated

# Download sample legal documents or use your own
# Example: Place PDF/DOCX files in data/raw/

# Process documents (extract text)
python utils/data_utils.py --input data/raw/ --output data/processed/

# Annotate data using annotation UI
# Follow guidelines in config/annotation_schema.json
# Output: data/annotated/{clause,ner,summary}_{train,val,test}.jsonl
```

---

## Step 3: Train Models (Optional - skip if using pre-trained)

### 3.1 Clause Classification

```bash
python training/hf_trainer_script.py \
  --task clause_classification \
  --model nlpaueb/legal-bert-base-uncased \
  --train_file data/annotated/clause_train.jsonl \
  --output_dir models_trained/clause_classifier \
  --seed 42 \
  --epochs 5 \
  --batch_size 16 \
  --learning_rate 2e-5
```

**Expected output:**
- Model checkpoints in `models_trained/clause_classifier/`
- Training logs in `logs/training/clause_classifier/`
- MLflow experiments tracked at http://localhost:5000

**Estimated time:** 2-3 hours on V100 GPU

---

### 3.2 Named Entity Recognition

```bash
python training/hf_trainer_script.py \
  --task ner \
  --model nlpaueb/legal-bert-base-uncased \
  --train_file data/annotated/ner_train.jsonl \
  --output_dir models_trained/ner_model \
  --seed 42 \
  --epochs 5 \
  --batch_size 16 \
  --learning_rate 2e-5
```

**Expected output:**
- Model checkpoints in `models_trained/ner_model/`
- Entity-level F1 scores in evaluation logs

**Estimated time:** 2-3 hours on V100 GPU

---

### 3.3 Summarization

```bash
python training/hf_trainer_script.py \
  --task summarization \
  --model google/long-t5-tglobal-base \
  --train_file data/annotated/summary_train.jsonl \
  --output_dir models_trained/summarizer \
  --seed 42 \
  --epochs 10 \
  --batch_size 8 \
  --learning_rate 5e-5
```

**Expected output:**
- Model checkpoints in `models_trained/summarizer/`
- ROUGE scores in evaluation logs

**Estimated time:** 8-10 hours on V100 GPU

---

## Step 4: Evaluate Models

```bash
# Evaluate clause classification
python evaluation/eval_clause_classification.py \
  --model models_trained/clause_classifier \
  --test_file data/annotated/clause_test.jsonl \
  --output results/clause_eval_results.json

# Evaluate NER
python evaluation/eval_ner.py \
  --model models_trained/ner_model \
  --test_file data/annotated/ner_test.jsonl \
  --output results/ner_eval_results.json

# Evaluate summarization
python evaluation/eval_summarization.py \
  --model models_trained/summarizer \
  --test_file data/annotated/summary_test.jsonl \
  --output results/summary_eval_results.json
```

**Expected metrics:**
- Clause Classification: F1 ≥ 0.85
- NER: Entity-level F1 ≥ 0.88
- Summarization: ROUGE-L ≥ 0.40

---

## Step 5: Serve Application Locally

```bash
# Start Streamlit web application
streamlit run app.py --server.port 5000

# Open browser to http://localhost:5000
```

**Features available:**
- Upload documents (PDF, DOCX, TXT)
- Extract and classify clauses
- Recognize named entities
- Generate summaries
- Ask questions with RAG

---

## Step 6: Test the System

### 6.1 Run Unit Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=models --cov-report=html

# Run specific test file
pytest tests/test_clause_extractor.py -v
```

**Expected:** All tests pass with >80% coverage

---

### 6.2 Integration Testing

```bash
# Run end-to-end integration tests
pytest tests/test_integration.py -v --slow

# Test API endpoints (if running FastAPI backend)
pytest tests/test_api.py -v
```

---

## Step 7: Deploy with Docker (Optional)

### 7.1 Build Docker Images

```bash
# Build web application image
docker build -f Dockerfile.web -t legal-dsl/web:latest .

# Build model service image
docker build -f Dockerfile.model -t legal-dsl/model:latest .
```

---

### 7.2 Run with Docker Compose

```bash
# Start all services (web, model, database, MLflow, monitoring)
docker-compose up -d

# View logs
docker-compose logs -f web

# Check service health
docker-compose ps

# Stop services
docker-compose down
```

**Services available:**
- Web UI: http://localhost:5000
- Model API: http://localhost:8000
- MLflow: http://localhost:5001
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090

---

## Step 8: Deploy to Kubernetes (Production)

```bash
# Create namespace
kubectl create namespace legal-dsl

# Apply configuration
kubectl apply -f k8s-deployment.yaml

# Check deployment status
kubectl get pods -n legal-dsl
kubectl get services -n legal-dsl

# View logs
kubectl logs -f deployment/legal-dsl-web -n legal-dsl

# Scale deployment
kubectl scale deployment legal-dsl-model --replicas=5 -n legal-dsl

# Delete deployment
kubectl delete namespace legal-dsl
```

---

## Troubleshooting

### Issue: Out of memory during training

**Solution:**
```bash
# Reduce batch size
--batch_size 8

# Enable gradient accumulation
--gradient_accumulation_steps 4

# Use mixed precision
--fp16 true
```

---

### Issue: Slow inference

**Solution:**
```bash
# Use GPU
export CUDA_VISIBLE_DEVICES=0

# Enable model compilation (PyTorch 2.0+)
--use_compile true

# Reduce max sequence length
--max_seq_length 256
```

---

### Issue: Low evaluation scores

**Solution:**
1. Check data quality and annotation consistency
2. Increase training epochs
3. Try different learning rates
4. Use domain-specific pretrained models (Legal-BERT)
5. Enable data augmentation

---

## Additional Resources

- **Documentation:** README.md
- **Research Paper:** research/IEEE_PAPER.json
- **API Specification:** config/model_api_spec.json
- **Annotation Guidelines:** config/annotation_schema.json
- **Training Config:** config/train_config.yaml

---

## Quick Reference: All Commands in Order

```bash
# 1. Setup
git clone https://github.com/legal-dsl-llm/legal-dsl-llm.git
cd legal-dsl-llm
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. (Optional) Train
python training/hf_trainer_script.py --task clause_classification ...
python training/hf_trainer_script.py --task ner ...
python training/hf_trainer_script.py --task summarization ...

# 3. Evaluate
python evaluation/eval_clause_classification.py ...
python evaluation/eval_ner.py ...
python evaluation/eval_summarization.py ...

# 4. Serve
streamlit run app.py --server.port 5000

# 5. Test
pytest tests/ -v --cov=models

# 6. (Optional) Deploy
docker-compose up -d
# OR
kubectl apply -f k8s-deployment.yaml
```

---

**Total time from clone to working system:**
- **With pre-trained models:** 15-30 minutes
- **With training:** 12-24 hours (depending on hardware)

**Questions?** See README.md or open an issue on GitHub.
