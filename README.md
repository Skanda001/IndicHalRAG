# HALRAG — Hallucination Analysis in RAG for Indic Languages

> **Comparing hallucination rates of Gemini 2.5 Flash vs Sarvam-105B on Kannada QA, with a binary hallucination detector trained on mBERT.**

---

## Project Overview

This project investigates hallucination behaviour in large language models (LLMs) for **Kannada**, a low-resource Indic language. We compare:

- **Gemini 2.5 Flash** (Google, general-purpose frontier model)
- **Sarvam-105B** (Sarvam AI, Indic-specialized model)

across two retrieval settings:

| Phase | Context Source | Research Question |
|-------|---------------|-------------------|
| **Phase 1** | Gold context (IndicQA) | Do models hallucinate even with perfect context? |
| **Phase 2** | Real RAG (FAISS + bge-m3 + Kannada Wikipedia) | Does retrieval quality affect hallucination? |

A binary **hallucination detector** (mBERT fine-tuned) is trained on the annotated data.

---

## Pipeline

```
download_questions.py     → data/questions.json          (500 Kannada QA pairs)
retrieve_passages.py      → data/raw_triplets.json        (Phase 1 gold context)
generate_responses.py     → data/kannada_llm_outputs.json (Gemini + Sarvam answers)
prepare_annotation.py     → data/annotation/*.json        (Label Studio tasks)
  [Manual annotation in Label Studio]
merge_annotations.py      → data/annotation/annotated_outputs.json
iaa_kappa.py              → results/iaa_results.json      (Cohen's Kappa)
phase1_analysis.py        → results/phase1_*.json + charts
train_detector.py         → models/detector_*/            (mBERT classifiers)
build_faiss_index.py      → data/faiss_index/             (Phase 2 index)
retrieve_real.py          → data/real_llm_outputs.json    (Phase 2 RAG answers)
phase2_analysis.py        → results/phase2_*.json + charts
```

---

## Setup

### 1. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API keys
```bash
python setup.py
```
Or manually copy `.env.example` → `.env` and fill in your keys:
- **Gemini** → https://aistudio.google.com/app/apikey (free, 1500 req/day)
- **Sarvam** → https://dashboard.sarvam.ai (free ₹100 credits on signup)

---

## Running the Pipeline

### Phase 1 — Gold Context

```bash
# Step 1: Download 500 Kannada questions from IndicQA
python scripts/download_questions.py

# Step 2: Create gold-context triplets (with answer-in-context validation)
python scripts/retrieve_passages.py

# Step 3: Generate LLM responses (test with 10 first!)
python scripts/generate_responses.py --limit 10
python scripts/generate_responses.py

# Step 4: Prepare annotation files for Label Studio
python scripts/prepare_annotation.py

# [Upload to Label Studio, annotate, export as person1_export.json etc.]

# Step 5: Merge annotation exports
python scripts/merge_annotations.py

# Step 6: Calculate Inter-Annotator Agreement
python scripts/iaa_kappa.py

# Step 7: Phase 1 analysis + charts
python scripts/phase1_analysis.py
```

### Phase 2 — Real RAG

```bash
# Step 8: Build FAISS index (Kannada Wikipedia + IndicQA gold contexts)
python scripts/build_faiss_index.py

# Step 9: Retrieve + generate with real RAG
python scripts/retrieve_real.py --retrieve-only   # sanity check first
python scripts/retrieve_real.py --limit 10         # test
python scripts/retrieve_real.py                    # full run

# [Annotate real_llm_outputs.json, add gemini_label + sarvam_label fields]

# Step 10: Phase 2 analysis
python scripts/phase2_analysis.py
```

### Detector Training

```bash
# Step 11: Train hallucination detector (requires annotated data)
python scripts/train_detector.py
```

---

## Annotation Labels

Each LLM response is manually labeled as one of:

| Label | Definition |
|-------|-----------|
| `correct` | Answer is accurate and supported by the passage |
| `hallucinated` | Answer contains information NOT in the passage |
| `partial` | Partially correct but adds unsupported claims |
| `refused` | Model says "answer not found" (even if answer exists) |

Label Studio XML config is in `notebooks/label_studio_config.xml`.

---

## File Structure

```
HALRAG/
├── scripts/
│   ├── llm_clients.py              # Shared Gemini + Sarvam LLM wrappers
│   ├── download_questions.py       # Download IndicQA Kannada questions
│   ├── retrieve_passages.py        # Phase 1: gold context triplets
│   ├── generate_responses.py       # Call Gemini + Sarvam, save responses
│   ├── prepare_annotation.py       # Split into Label Studio tasks
│   ├── merge_annotations.py        # Merge 4 annotator exports
│   ├── iaa_kappa.py                # Inter-annotator agreement (Cohen's Kappa)
│   ├── check_label_distribution.py # Quick sanity check on labels
│   ├── phase1_analysis.py          # Phase 1 hallucination analysis + charts
│   ├── phase2_analysis.py          # Phase 2 RAG vs gold comparison
│   ├── build_faiss_index.py        # Build FAISS index from Kannada Wikipedia
│   ├── retrieve_real.py            # Phase 2: real RAG retrieval + generation
│   ├── train_detector.py           # Train mBERT hallucination detector
│   └── test_sarvam.py              # Quick API connectivity test
├── notebooks/                      # Jupyter notebooks for exploration
├── data/                           # Generated data (gitignored)
│   ├── questions.json
│   ├── raw_triplets.json
│   ├── kannada_llm_outputs.json
│   ├── annotation/
│   └── faiss_index/
├── results/                        # Analysis outputs (gitignored)
├── models/                         # Trained detector models (gitignored)
├── requirements.txt
├── setup.py
└── .env.example
```

---

## Rate Limits

| API | Free Tier | Our Usage |
|-----|-----------|-----------|
| Gemini 2.5 Flash | 15 RPM, 1500 req/day | ~12 RPM (5s delay) |
| Sarvam-105B | ₹100 free credits | ~5s delay |

If you hit `429 ResourceExhausted`, increase `DELAY_BETWEEN_REQUESTS` in `llm_clients.py`.
The pipeline auto-resumes from checkpoints — already-completed rows are skipped.

---

## Key Design Decisions

- **Stratified sampling**: max 4 questions per Wikipedia article (prevents topic bias)
- **Answer-in-context validation**: rows where `gold_answer ∉ gold_context` are flagged and excluded
- **Context windowing**: only the sentence containing the answer ± 1 sentence is sent to the LLM (realistic RAG chunk size)
- **mBERT detector**: multilingual BERT fine-tuned as binary classifier (`hallucinated=1`, else=0), trained with class-weighted loss to handle label imbalance
