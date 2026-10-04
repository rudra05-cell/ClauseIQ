# ClauseIQ

An NLP system that classifies contract clauses into legal categories and
flags their risk level. Built on the CUAD (Contract Understanding Atticus
Dataset), combining a trained multi-label classifier with rule-based risk
analysis.

## Features

- **Document analysis** — upload a full contract (PDF, DOCX, TXT); it is
  automatically split into clauses, each classified and risk-scored, with
  an overall risk score, a filterable findings list, and a highlighted
  in-document view.
- **Single clause analysis** — paste an individual clause for a quick check.
- **Batch analysis** — upload multiple contracts at once for a
  portfolio-level risk summary.
- **Clause and document comparison** — compare two clauses, two documents,
  or one of each (e.g. before/after negotiation, or two competing vendor
  contracts) to see which carries more risk and exactly which categories
  differ between them.
- **Exportable reports** — download findings as CSV or a formatted PDF.
- **Confidence scores** — each prediction shows the model's confidence.
- **Suggested rewrites** — High-risk clauses come with a rule-based
  suggestion for what a stronger version could require.
- **Feedback logging** — mark a prediction correct or incorrect; feedback
  is logged to `reports/feedback_log.csv` for future review.
- **REST API** — the same analysis is available over HTTP via FastAPI.

## Categories

Expanded from an initial 6 to 11, all chosen for having strong data support
in CUAD (100+ labeled examples) and a clear risk story:

- Termination
- Renewal Term
- Renewal Notice Period
- Liability Cap
- Uncapped Liability
- Non-Compete
- Audit Rights
- Insurance
- Exclusivity
- Warranty Duration
- Volume Restriction

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# 1. Download the CUAD dataset
python src/download_data.py

# 2. Build the training dataset
python src/preprocessing.py

# 3. Train the baseline model
python src/baseline_model.py

# 4. Launch the web application
python -m streamlit run app/streamlit_app.py
```

Run all commands from the project root.

### REST API

```bash
python -m uvicorn api.main:app --reload --port 8000
```

Interactive docs at `http://localhost:8000/docs`. Endpoints:
- `POST /analyze-clause` — `{"text": "..."}` → category, risk level, reason, confidence
- `POST /analyze-document` — multipart file upload → full document analysis

### Docker

```bash
docker compose up
```

Runs the Streamlit app on port 8501 and the API on port 8000.

## Enabling the transformer model (optional)

1. Open `notebooks/transformer_finetuning.ipynb` in Google Colab and select
   a GPU runtime.
2. Upload `data/processed/clauses_dataset.csv` when prompted, then run all
   cells.
3. Extract the downloaded model into `models/legalbert_finetuned/`.
4. Relaunch the app — an "Enhanced" model option will appear in the
   sidebar, selectable alongside the Standard model.

Note: the category set was expanded from 6 to 11. A transformer fine-tuned
on the earlier 6-category version is not compatible and needs to be
retrained with the current notebook.

## Testing

```bash
python -m pytest tests/ -v
```

27 tests cover risk scoring, document splitting, and feedback logging,
including regression tests for three bugs found by testing against a real
external contract not seen during training:
- a damages-exclusion clause misflagged as uncapped liability,
- a notice period stated in months (rather than days) that was missed, and
- a one-sided termination clause using a party role name the pattern
  didn't originally recognize.

### Real-world holdout check

```bash
python src/evaluate_holdout.py
```

Evaluates the model against a small, hand-labeled set of clauses from a
real, publicly released SEC contract that is not part of CUAD. This is an
honesty check on generalization, not a replacement for the CUAD test split
metrics below — see the script's docstring for details on its scope and
limitations.

## Classical ML vs. LLM comparison (optional)

`src/llm_comparison.py` compares the trained classifier's prediction
against prompting an LLM directly on the same clause. Requires an
`ANTHROPIC_API_KEY` environment variable; without one it fails gracefully
with a clear message and the rest of the app is unaffected.

## Architecture

```
Contract text
     │
     ▼
Preprocessing — clause segmentation and cleaning
     │
     ├── Standard model (TF-IDF + Logistic Regression)
     └── Enhanced model (fine-tuned transformer, optional)
     │
     ▼
Risk scoring — rule-based checks on predicted categories
     │
     ▼
Explainability, suggestions, confidence scoring
     │
     ▼
Streamlit app  /  REST API
```

## Results

| Model | Macro F1 | Micro F1 |
|---|---|---|
| Baseline (TF-IDF + Logistic Regression), 11 categories | see `reports/evaluation_metrics.csv` | |
| Transformer (fine-tuned) | pending — run the notebook | |

## Project structure

```
clauseiq/
├── api/
│   └── main.py
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   ├── processed/
│   │   └── clauses_dataset.csv
│   └── holdout/
│       └── holdout_labels.csv
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── baseline_model.pkl
│   └── legalbert_finetuned/
├── notebooks/
│   └── transformer_finetuning.ipynb
├── src/
│   ├── download_data.py
│   ├── preprocessing.py
│   ├── baseline_model.py
│   ├── transformer_model.py
│   ├── risk_scoring.py
│   ├── explainability.py
│   ├── document_processing.py
│   ├── report_generator.py
│   ├── suggestions.py
│   ├── feedback.py
│   ├── llm_comparison.py
│   └── evaluate_holdout.py
├── tests/
├── reports/
│   └── evaluation_metrics.csv
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Dataset

[CUAD](https://github.com/TheAtticusProject/cuad) — 510 commercial contracts
with over 13,000 expert-labeled clauses across 41 categories, filtered here
to 11. License: CC BY 4.0.

## Limitations

- Covers 11 of CUAD's 41 clause categories.
- Risk rules are hand-authored, not learned from labeled risk data.
- The CUAD test-split F1 score measures performance on documents from the
  same source as training data; the holdout script checks a small,
  separate, hand-labeled sample for a more honest generalization signal,
  but is not statistically powered at its current size.
- Trained on commercial contract language; may not generalize to informal
  or non-English text.
- Informational tool only — not a substitute for legal advice.
