"""Logs user feedback on flagged clauses to a local CSV file.

This does not retrain the model -- it builds a record of corrections that
could seed a future active-learning pass. Kept intentionally simple: a
flat, append-only CSV rather than a database, since the volume expected
from a demo/small-scale tool doesn't need more than that.
"""

import os
import csv
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEEDBACK_PATH = os.path.join(PROJECT_ROOT, "reports", "feedback_log.csv")

_FIELDS = ["timestamp", "clause_text", "predicted_category", "predicted_risk", "feedback"]


def log_feedback(clause_text: str, category: str, risk_level: str, feedback: str):
    """feedback: 'correct' or 'incorrect', as marked by the user in the UI."""
    os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)
    file_exists = os.path.isfile(FEEDBACK_PATH)

    with open(FEEDBACK_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "clause_text": clause_text,
            "predicted_category": category,
            "predicted_risk": risk_level,
            "feedback": feedback,
        })


def load_feedback_summary() -> dict:
    """Returns counts of correct/incorrect feedback logged so far, or an
    empty summary if no feedback has been recorded yet."""
    if not os.path.isfile(FEEDBACK_PATH):
        return {"correct": 0, "incorrect": 0, "total": 0}

    correct = incorrect = 0
    with open(FEEDBACK_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["feedback"] == "correct":
                correct += 1
            elif row["feedback"] == "incorrect":
                incorrect += 1

    return {"correct": correct, "incorrect": incorrect, "total": correct + incorrect}
