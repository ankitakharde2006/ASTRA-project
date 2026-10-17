# RAG Pipeline Evaluation Plan

## Overview

This document outlines a systematic evaluation of the Naive RAG pipeline performance across different chunk sizes (100 vs 500 characters). The evaluation will assess retrieval effectiveness while keeping all other parameters controlled:

- **Chunk Sizes to Compare**: 100 characters (small) vs 500 characters (large)
- **Controlled Settings**: 
  - Overlap policy: 25% of chunk size (25 chars for 100, 125 chars for 500)
  - Top-k: 5 chunks retrieved
  - Embedding model: nvidia/nemotron-3-embed-1b
  - Similarity metric: Cosine similarity
- **Test Dataset**: The AstraTech employee handbook (astratech_employee_handbook.txt)
- **Manual Evaluation**: Human judgment of retrieved chunk relevance

## Test Questions (10 Total)

For each test question, we need to:
1. Identify which source file should contain the answer
2. Define how to manually judge if retrieved chunks are relevant (hit rate)

### Question 1: What is the one-time remote home office setup stipend amount?
**Expected Source File**: `data/astratech_employee_handbook.txt`
**Relevance Criteria**: 
- ✅ **HIT**: Chunk contains the exact stipend amount (e.g., "$1,200", "$1,200.00")
- ✅ **HIT**: Chunk mentions "one-time remote home office setup stipend"
- ❌ **MISS**: Chunk contains only related but different compensation information
- ❌ **MISS**: Chunk contains placeholder text or examples without actual amounts

### Question 2: What is the Wi-Fi password for office visitors?
**Expected Source File**: `data/astratech_employee_handbook.txt`
**Relevance Criteria**:
- ✅ **HIT**: Chunk contains the exact Wi-Fi password (e.g., "CoffeeAndCode2026!")
- ✅ **HIT**: Chunk contains "office visitors Wi-Fi" or similar keywords
- ❌ **MISS**: Chunk contains unrelated network configuration
- ❌ **MISS**: Chunk contains examples without actual passwords

### Question 3: How many paid sick days are employees entitled to?
**Expected Source File**: `data/astratech_employee_handbook.txt`
**Relevance Criteria**:
- ✅ **HIT**: Chunk contains specific number (e.g., "15 paid sick days")
- ✅ **HIT**: Chunk contains policy details about sick leave entitlements
- ❌ **MISS**: Chunk contains other types of leave (vacation, holidays)
- ❌ **MISS**: Chunk contains guidelines without specific numbers

### Question 4: What is the expense submission deadline for remote work equipment?
**Expected Source File**: `data/astratech_employee_handbook.txt`
**Relevance Criteria**:
- ✅ **HIT**: Chunk contains specific deadline (e.g., "30 days", "within 30 days")
- ✅ **HIT**: Chunk contains policy language about submission timelines
- ❌ **MISS**: Chunk contains processing timelines instead of submission deadlines
- ❌ **MISS**: Chunk mentions other expenses or different deadlines

### Question 5: What is the required format for expense reports?
**Expected Source File**: `data/astratech_employee_handbook.txt`
**Relevance Criteria**:
- ✅ **HIT**: Chunk contains format requirements (e.g., "Excel spreadsheet", "Expensify")
- ✅ **HIT**: Chunk contains specific instructions for report structure
- ❌ **MISS**: Chunk contains general expense policies
- ❌ **MISS**: Chunk contains processing procedures without format details

### Question 6: Which programming languages are required for the development team?
**Expected Source File**: `data/astratech_employee_handbook.txt`
**Relevance Criteria**:
- ✅ **HIT**: Chunk lists specific programming languages (e.g., "Python, JavaScript, Go")
- ✅ **HIT**: Chunk contains skill requirements for developers
- ❌ **MISS**: Chunk contains technology stack descriptions without requirements
- ❌ **MISS**: Chunk contains project requirements without language specifics

### Question 7: What is the policy for remote work equipment returns?
**Expected Source File**: `data/astratech_employee_handbook.txt`
**Relevance Criteria**:
- ✅ **HIT**: Chunk contains return policy details (e.g., "30 days", "within 14 days")
- ✅ **HIT**: Chunk contains specific procedures for equipment returns
- ❌ **MISS**: Chunk contains acquisition or setup procedures
- ❌ **MISS**: Chunk contains related but different policy areas

### Question 8: What training resources are available for new hires?
**Expected Source File**: `data/astratech_employee_handbook.txt`
**Relevance Criteria**:
- ✅ **HIT**: Chunk lists specific training resources or programs
- ✅ **HIT**: Chunk contains training curriculum or modules
- ❌ **MISS**: Chunk contains general benefits information
- ❌ **MISS**: Chunk contains compliance requirements without resources

### Question 9: How is performance review frequency determined?
**Expected Source File**: `data/astratech_employee_handbook.txt`
**Relevance Criteria**:
- ✅ **HIT**: Chunk contains specific review intervals (e.g., "quarterly", "biannually")
- ✅ **HIT**: Chunk contains performance management schedule details
- ❌ **MISS**: Chunk contains review criteria without frequency
- ❌ **MISS**: Chunk contains manager responsibilities without timing

### Question 10: What are the travel reimbursement limits per mile?
**Expected Source File**: `data/astratech_employee_handbook.txt`
**Relevance Criteria**:
- ✅ **HIT**: Chunk contains specific mileage rates (e.g., "$0.65 per mile")
- ✅ **HIT**: Chunk contains travel expense policy details
- ❌ **MISS**: Chunk contains total travel budget without per-mile rates
- ❌ **MISS**: Chunk contains vehicle policies without reimbursement specifics

## Evaluation Methodology

### 1. Setup

**Before Testing:**
1. Ensure both chunk sizes are tested on the same dataset
2. Standardize overlap policy (25% of chunk size)
3. Use consistent top-k=5 retrieval
4. Document exact test parameters in a configuration file

**Test Configuration for Each Chunk Size:**
```
Test Configuration:
  Chunk Size: 100 characters
  Chunk Overlap: 25 characters (25%)
  Top-K Retrieved: 5
  Similarity Metric: Cosine
  Test Dataset: astratech_employee_handbook.txt
```

### 2. Manual Relevance Judgment Process

**For each question + chunk size combination:**

1. **Execute Query**: Run the RAG pipeline with the test question
2. **Retrieve Results**: Note the retrieved chunks and their similarity scores
3. **Manual Assessment**: For each retrieved chunk:
   - Read the chunk text and the original question
   - Apply the relevance criteria above
   - Mark as HIT if it meets criteria, MISS otherwise
   - Record similarity score for analysis
4. **Calculate Hit Rate**: Hit Rate = (Number of HITs) / (Total Chunks Retrieved)
5. **Document Findings**: Record all details in the results table

**Example Manual Assessment:**
```
Question: "What is the Wi-Fi password for office visitors?"
Chunk 1: "The office guest Wi-Fi network SSID is AstraTech-Guest with password 'CoffeeAndCode2026!'"
Manual Judgment: ✅ HIT (Contains exact password and context)
Similarity Score: 0.8765

Chunk 2: "Employees should use the corporate VPN when connecting to external networks."
Manual Judgment: ❌ MISS (No password information)
Similarity Score: 0.4521

Hit Rate: 1/2 = 50%
```

### 3. Retrieval Hit-Rate Calculation

**Formula:**
```
Hit Rate = (Number of Relevant Chunks) / (Total Chunks Retrieved) × 100%

Where a "Relevant Chunk" is one that meets the manual relevance criteria for the specific question.
```

**Quality Metrics to Track:**
1. **Hit Rate**: Primary relevance metric
2. **Average Similarity**: Quality of retrieved chunks
3. **First Chunk Hit Rate**: Whether the most similar chunk is relevant
4. **Top-1 Accuracy**: Whether the single best retrieval is relevant

## Results Recording Table

| Test ID | Chunk Size | Question | Source File | Chunk # | Text Preview | Similarity | Manual Judgment | HIT/MISS | Hit Rate (per test) |
|---------|------------|----------|-------------|---------|--------------|------------|-----------------|----------|---------------------|
| 1-A | 100 chars | Q1: Setup stipend | handbook.txt | 1 | "The one-time remote home office setup stipend..." | 0.8921 | ✅ Contains $1,200 | HIT | 1/5 (20%) |
| 1-A | 100 chars | Q1: Setup stipend | handbook.txt | 2 | "Employees may request equipment up to $1,200..." | 0.7654 | ✅ Mentions $1,200 context | HIT | 2/5 (40%) |
| 1-A | 100 chars | Q1: Setup stipend | handbook.txt | 3 | "Remote work policy requires pre-approval..." | 0.3421 | ❌ No stipend amount | MISS | 2/5 (40%) |
| 1-A | 100 chars | Q1: Setup stipend | handbook.txt | 4 | "Travel reimbursement at $0.65 per mile..." | 0.1234 | ❌ Different topic | MISS | 2/5 (40%) |
| 1-A | 100 chars | Q1: Setup stipend | handbook.txt | 5 | "Family medical leave policy..." | 0.0987 | ❌ Unrelated | MISS | 2/5 (40%) |
| 1-B | 500 chars | Q1: Setup stipend | handbook.txt | 1 | "The one-time remote home office setup stipend at AstraTech Solutions is $1,200 (to be submitted via Expensify within 30 days)..." | 0.9567 | ✅ Contains complete stipend policy | HIT | 1/5 (20%) |
| 1-B | 500 chars | Q1: Setup stipend | handbook.txt | 2 | "Equipment requests must be approved by direct manager and submitted through the HR portal..." | 0.8234 | ✅ Mentions $1,200 in context | HIT | 2/5 (40%) |
| 1-B | 500 chars | Q1: Setup stipend | handbook.txt | 3 | "All employees receive 15 paid sick days per year..." | 0.4567 | ❌ Different policy | MISS | 2/5 (40%) |
| 1-B | 500 chars | Q1: Setup stipend | handbook.txt | 4 | "Remote team members may work from any location..." | 0.2345 | ❌ No stipend details | MISS | 2/5 (40%) |
| 1-B | 500 chars | Q1: Setup stipend | handbook.txt | 5 | "The visitor Wi-Fi network requires guest badge verification..." | 0.1123 | ❌ Wi-Fi only, no stipend | MISS | 2/5 (40%) |

## Analysis Guidelines

### What to Look For:

1. **Chunk Size Impact**: Does 500-char chunks improve or reduce hit rates?
2. **Context Completeness**: Do larger chunks provide more complete information?
3. **Precision vs Recall**: Do smaller chunks be more precise, larger chunks more comprehensive?
4. **Similarity Scores**: Do higher similarity scores correlate with manual relevance?
5. **Noise Levels**: Do larger chunks introduce more irrelevant but detailed information?

### Expected Findings (Preliminary Hypotheses):

**Hypothesis 1**: 100-char chunks will have higher precision (fewer irrelevant chunks) but lower recall (misses contextual details)
**Hypothesis 2**: 500-char chunks will have higher recall (more context) but potentially lower precision (more noise)
**Hypothesis 3**: The relationship between similarity scores and manual relevance may differ by chunk size

### Reporting Structure:

1. **Raw Data**: Complete results table with all test runs
2. **Statistical Analysis**: Hit rate comparisons, similarity score distributions
3. **Qualitative Analysis**: Patterns in relevance, chunk quality assessment
4. **Recommendations**: Which chunk size performs better for different use cases

## Next Steps

1. **Execute Tests**: Run the evaluation using this framework
2. **Manual Review**: Apply relevance criteria to each result
3. **Data Collection**: Populate the results table with actual findings
4. **Analysis**: Draw conclusions about chunk size effectiveness
5. **Documentation**: Create final report with recommendations

**Important**: This evaluation requires manual judgment of retrieved chunks. The scoring system is designed to be consistent but may require adjustments based on the specific content of the AstraTech handbook and the test questions used.