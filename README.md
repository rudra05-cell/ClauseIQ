# Contract Clause Risk Classifier

An NLP system that classifies contract clauses into legal categories and flags
their risk level. Built on the CUAD (Contract Understanding Atticus Dataset),
combining a trained multi-label classifier with rule-based risk analysis.

## Features

- **Document analysis** — upload a full contract (PDF, DOCX, TXT); it is
  automatically split into clauses, each classified and risk-scored, with a
  summary report and filterable results.
- **Single clause analysis** — paste an individual clause for a quick check.
- **Risk scoring** — clauses are flagged Low, Medium, or High risk based on
  specific wording (missing notice periods, uncapped liability, broad
  non-compete scope, one-sided termination rights).
- **Explainability** — every result shows the terms that drove the
  classification.
- **Overall risk score** — an aggregate risk rating for the whole document,
  based on its highest individual clause risk.
- **Highlighted document view** — see the entire contract in its original
  order, with flagged sections color-coded by risk level.
- **Exportable reports** — download findings as CSV or a formatted PDF.
- **Two models** — a fast TF-IDF + Logistic Regression baseline (always
  available), and an optional fine-tuned transformer for improved accuracy.

## Categories

- Termination
- Renewal Term
- Renewal Notice Period
- Liability Cap
- Uncapped Liability
- Non-Compete

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

# 4. Launch the application
python -m streamlit run app/streamlit_app.py
```

Run all commands from the project root.

## Enabling the transformer model (optional)

1. Open `notebooks/transformer_finetuning.ipynb` in Google Colab and select a
   GPU runtime.
2. Upload `data/processed/clauses_dataset.csv` when prompted, then run all
   cells.
3. Extract the downloaded model into `models/legalbert_finetuned/`.
4. Relaunch the app — an enhanced model option will appear automatically.

## Architecture

```
Contract text
     │
     ▼
Preprocessing — clause segmentation and cleaning
     │
     ├── Baseline model (TF-IDF + Logistic Regression)
     └── Transformer model (fine-tuned, optional)
     │
     ▼
Risk scoring — rule-based checks on predicted categories
     │
     ▼
Explainability — key terms behind each classification
     │
     ▼
Web application — document and single-clause analysis
```

## Results

| Model | Macro F1 | Micro F1 |
|---|---|---|
| Baseline (TF-IDF + Logistic Regression) | 0.853 | 0.892 |

## Project structure

```
contract-clause-risk-classifier/
├── data/
│   ├── raw/
│   └── processed/
│       └── clauses_dataset.csv
├── notebooks/
│   └── transformer_finetuning.ipynb
├── src/
│   ├── download_data.py
│   ├── preprocessing.py
│   ├── baseline_model.py
│   ├── transformer_model.py
│   ├── risk_scoring.py
│   ├── explainability.py
│   └── document_processing.py
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── baseline_model.pkl
│   └── legalbert_finetuned/
├── app/
│   └── streamlit_app.py
├── requirements.txt
└── README.md
```

## Dataset

[CUAD](https://github.com/TheAtticusProject/cuad) — 510 commercial contracts
with over 13,000 expert-labeled clauses across 41 categories, filtered here
to six.

## Limitations

- Covers 6 of CUAD's 41 clause categories.
- Risk rules are hand-authored, not learned from labeled risk data.
- Trained on commercial contract language; may not generalize to informal or
  non-English text.
- Informational tool only — not a substitute for legal advice.
