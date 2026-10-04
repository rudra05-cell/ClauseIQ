"""ClauseIQ REST API — exposes clause and document analysis over HTTP.

Run with: python -m uvicorn api.main:app --reload --port 8000
Docs at:  http://localhost:8000/docs
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from src.baseline_model import load_model, predict as predict_baseline
from src import risk_scoring
from src import document_processing
from src import report_generator

app = FastAPI(
    title="ClauseIQ API",
    description="Contract clause classification and risk analysis.",
    version="1.0.0",
)

_vectorizer, _clf = load_model()


class ClauseRequest(BaseModel):
    text: str


class ClauseResult(BaseModel):
    category: str
    risk_level: str
    reason: str
    confidence: float


@app.get("/")
def root():
    return {"service": "ClauseIQ API", "status": "ok"}


@app.post("/analyze-clause", response_model=list[ClauseResult])
def analyze_clause(payload: ClauseRequest):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Clause text must not be empty.")

    preds = predict_baseline(payload.text, _vectorizer, _clf)
    categories = [c for c, _ in preds]
    confidence = {c: p for c, p in preds}

    results = []
    for r in risk_scoring.assess(payload.text, categories):
        results.append(ClauseResult(
            category=r["category"],
            risk_level=r["level"],
            reason=r["reason"],
            confidence=confidence.get(r["category"], 0.0),
        ))
    return results


@app.post("/analyze-document")
async def analyze_document(file: UploadFile = File(...)):
    class _FileWrapper:
        """Adapts FastAPI's UploadFile to the .name/.read() interface
        document_processing expects (originally written for Streamlit's
        UploadedFile)."""
        def __init__(self, upload: UploadFile, content: bytes):
            self.name = upload.filename
            self._content = content

        def read(self):
            return self._content

    content = await file.read()
    wrapped = _FileWrapper(file, content)

    try:
        text = document_processing.extract_text(wrapped)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    clauses = document_processing.split_into_clauses(text)

    results = []
    for clause in clauses:
        preds = predict_baseline(clause, _vectorizer, _clf)
        categories = [c for c, _ in preds]
        if categories:
            for r in risk_scoring.assess(clause, categories):
                results.append({
                    "category": r["category"],
                    "risk_level": r["level"],
                    "reason": r["reason"],
                    "clause_text": clause,
                })

    return {
        "filename": file.filename,
        "sections_found": len(clauses),
        "flagged_clauses": len(results),
        "overall_risk": report_generator.compute_overall_risk(
            [{"level": r["risk_level"]} for r in results]
        ),
        "results": results,
    }
