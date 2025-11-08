# Sample Legal Documents for Testing

This folder contains sample legal documents to test the Legal-DSL LLM application's various features.

## Available Sample Documents

### 1. **nda_agreement.txt** (2.9 KB, 61 lines)
**Type:** Non-Disclosure Agreement  
**Best for testing:**
- Confidentiality clause extraction
- Party identification (TechCorp Inc., Innovation Labs LLC)
- Date and term extraction
- Arbitration and dispute resolution clauses
- Governing law identification

**Expected Results:**
- Clauses: Confidentiality, arbitration, governing law, indemnification, intellectual property
- Entities: PARTY (TechCorp Inc., Innovation Labs LLC), DATE (March 1, 2024), JURISDICTION (Delaware)
- Sample Questions: "What are the confidentiality obligations?", "Who are the parties?", "What is the term of this agreement?"

---

### 2. **employment_agreement.txt** (2.9 KB, 70 lines)
**Type:** Employment Contract  
**Best for testing:**
- Compensation and payment terms
- Termination clauses
- Non-compete provisions
- Benefits extraction
- Multiple entity types

**Expected Results:**
- Clauses: Payment, termination, confidentiality, non-compete, intellectual property
- Entities: PARTY (Digital Solutions Corp., Jennifer Martinez), AMOUNT ($125,000), DATE (January 15, 2024), JURISDICTION (California)
- Sample Questions: "What is the salary?", "What are the termination conditions?", "What benefits are included?"

---

### 3. **license_agreement.txt** (4.4 KB, 99 lines)
**Type:** Software License Agreement  
**Best for testing:**
- License and intellectual property clauses
- Payment and fee terms
- Warranty and liability clauses
- Detailed restrictions
- International parties (US and UK)

**Expected Results:**
- Clauses: Payment, warranty, liability, intellectual property, termination, force majeure, governing law
- Entities: PARTY (CloudTech Software Inc., Enterprise Solutions Ltd.), AMOUNT ($50,000 USD), DATE (February 10, 2024), JURISDICTION (Delaware, UK, New York)
- Sample Questions: "What are the license restrictions?", "What is the license fee?", "What warranties are provided?"

---

### 4. **terms_of_service.txt** (5.0 KB, 146 lines)
**Type:** Terms of Service / User Agreement  
**Best for testing:**
- Comprehensive clause extraction (longest document)
- Multiple pricing tiers
- Warranty disclaimers and liability limitations
- User obligations and restrictions
- Complex legal language

**Expected Results:**
- Clauses: Payment, warranty, liability, termination, arbitration, governing law, indemnification
- Entities: ORGANIZATION (WebApp Pro Inc.), AMOUNT ($9.99, $29.99), ADDRESS (San Francisco, CA), EMAIL (legal@webapppro.com), PHONE, JURISDICTION (California)
- Sample Questions: "What are the subscription plans?", "What is the limitation of liability?", "How can I terminate my account?"

---

## How to Test

### Step 1: Access the Application
Open the Streamlit app in your browser (running on port 5000).

### Step 2: Upload a Document
1. Go to "Upload & Process" page
2. Click "Choose a document"
3. Navigate to `LegalDslLlm/sample_documents/`
4. Select one of the `.txt` files
5. Click "🚀 Process Document"

### Step 3: Explore Results

**Clause Extraction**
- Navigate to "Clause Extraction" page
- Filter by clause type
- Adjust confidence threshold
- Review extracted clauses with their types and confidence scores

**Named Entities**
- Navigate to "Named Entities" page
- See entity type distribution in pie chart
- Filter by entity type
- Export entities as JSON

**Summarization**
- Navigate to "Summarization" page
- View abstractive summary
- See extractive highlights with provenance
- Check compression ratio and metadata

**RAG Q&A**
- Navigate to "RAG Q&A" page
- Try the example questions or create your own
- See grounded answers with source attribution
- Adjust number of chunks to retrieve

---

## Testing Tips

### Progressive Testing
Start simple, then increase complexity:
1. **First:** Test with `nda_agreement.txt` (simplest)
2. **Then:** Try `employment_agreement.txt` (moderate)
3. **Next:** Test `license_agreement.txt` (complex)
4. **Finally:** Try `terms_of_service.txt` (most comprehensive)

### Feature-Specific Testing

**To test clause extraction:**
- Use `license_agreement.txt` (has 7+ different clause types)

**To test entity recognition:**
- Use `employment_agreement.txt` (has parties, amounts, dates, jurisdictions)

**To test summarization:**
- Use `terms_of_service.txt` (longest document, best compression ratio)

**To test RAG Q&A:**
- Use `nda_agreement.txt` (clear structure, easy to query)

### Expected Processing Time
- Small documents (< 3 KB): 2-5 seconds
- Medium documents (3-5 KB): 5-10 seconds
- Processing includes: clause extraction, NER, summarization, and RAG indexing

---

## Sample Questions by Document Type

### For NDA Agreement
- "What information is considered confidential?"
- "How long does this agreement last?"
- "What happens if someone violates the confidentiality?"
- "Which law governs this agreement?"

### For Employment Agreement
- "What is the employee's position and salary?"
- "How many vacation days are provided?"
- "What are the termination conditions?"
- "Are there any non-compete restrictions?"

### For License Agreement
- "What are the restrictions on using the software?"
- "What warranty is provided?"
- "How can this agreement be terminated?"
- "What is the limitation of liability?"

### For Terms of Service
- "What are the different subscription plans?"
- "What activities are prohibited?"
- "How can I cancel my subscription?"
- "What happens to my data if I terminate?"

---

## Troubleshooting

**If no clauses are extracted:**
- This is normal for very short or non-legal text
- Try a different sample document
- Legal documents work best

**If entity count seems low:**
- The system extracts high-confidence entities
- Some documents naturally have fewer entities
- Check the entity type distribution

**If RAG confidence is low:**
- Normal for short documents
- System performs better with longer, detailed text
- Try more specific questions

**If processing seems slow:**
- First document takes longer (model loading)
- Subsequent documents process faster
- OCR-enabled PDFs take additional time

---

## File Locations

All sample documents are located in:
```
LegalDslLlm/sample_documents/
├── nda_agreement.txt
├── employment_agreement.txt
├── license_agreement.txt
└── terms_of_service.txt
```

---

## Next Steps

After testing with these samples:
1. **Try your own documents** - Upload real legal contracts
2. **Compare results** - See how different document types perform
3. **Experiment with questions** - Test the RAG Q&A with complex queries
4. **Export data** - Download extracted entities as JSON
5. **Review research paper** - Check the "Research Paper" page for methodology

---

## Need Help?

- Check `BUG_FIXES_SUMMARY.md` for common issues
- Review `replit.md` for project documentation
- All features have built-in help and example questions

Happy testing! 🎉
