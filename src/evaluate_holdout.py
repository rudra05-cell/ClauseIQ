"""Evaluates the model against a small, hand-labeled set of clauses from a
real external contract (SEC-filed, publicly released via FOIA) that is NOT
part of CUAD and was never seen during training.

This is an honesty check, not a benchmark: the CUAD test split measures
performance on documents the model's training data was drawn from. This
script measures something different -- does it still work on a contract
it has genuinely never seen.

N is small (5 hand-labeled clauses) because hand-labeling real contracts
is slow and this is illustrative, not statistically powered. Extend
data/holdout/holdout_labels.csv with more real, hand-labeled clauses to
strengthen this check.
"""

import os
import csv
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baseline_model import predict, load_model
from src import risk_scoring

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS_PATH = os.path.join(PROJECT_ROOT, "data", "holdout", "holdout_labels.csv")


def run_evaluation():
    vectorizer, clf = load_model()

    with open(LABELS_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    category_correct = 0
    risk_correct = 0

    print(f"Evaluating against {len(rows)} hand-labeled real-world clauses\n")

    for i, row in enumerate(rows, 1):
        text = row["clause_text"]
        expected_category = row["expected_category"]
        expected_risk = row["expected_risk"]

        preds = predict(text, vectorizer, clf)
        predicted_categories = [c for c, _ in preds]
        category_match = expected_category in predicted_categories

        risk_match = False
        if category_match:
            risk_results = risk_scoring.assess(text, [expected_category])
            actual_risk = risk_results[0]["level"] if risk_results else None
            risk_match = actual_risk == expected_risk
        else:
            actual_risk = None

        category_correct += int(category_match)
        risk_correct += int(risk_match)

        status = "PASS" if (category_match and risk_match) else "FAIL"
        print(f"[{status}] Clause {i}")
        print(f"  Expected: {expected_category} / {expected_risk}")
        print(f"  Predicted categories: {predicted_categories}")
        print(f"  Predicted risk (for expected category): {actual_risk}")
        print()

    n = len(rows)
    print("=" * 50)
    print(f"Category match: {category_correct}/{n} ({100*category_correct/n:.0f}%)")
    print(f"Risk level match (when category matched): {risk_correct}/{n} ({100*risk_correct/n:.0f}%)")
    print()
    print("Note: this is a small, illustrative real-world holdout check, "
          "not a statistically powered benchmark. See module docstring.")


if __name__ == "__main__":
    run_evaluation()
