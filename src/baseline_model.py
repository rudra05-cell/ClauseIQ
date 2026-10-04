"""TF-IDF + Logistic Regression multi-label classifier for contract clauses."""

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, f1_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "clauses_dataset.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

CATEGORIES = [
    "Termination For Convenience",
    "Renewal Term",
    "Notice Period To Terminate Renewal",
    "Cap On Liability",
    "Uncapped Liability",
    "Non-Compete",
    "Audit Rights",
    "Insurance",
    "Exclusivity",
    "Warranty Duration",
    "Volume Restriction",
]


def _load_data():
    df = pd.read_csv(DATA_PATH)
    X = df["clause_text"].astype(str).to_numpy()
    y = df[CATEGORIES].to_numpy(dtype=int)
    return X, y


def train() -> dict:
    X, y = _load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(
        max_features=6000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced"))
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_test, y_pred, average="micro", zero_division=0)
    report = classification_report(y_test, y_pred, target_names=CATEGORIES, zero_division=0)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    joblib.dump(clf, os.path.join(MODEL_DIR, "baseline_model.pkl"))

    return {"macro_f1": macro_f1, "micro_f1": micro_f1, "report": report}


def load_model():
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    clf = joblib.load(os.path.join(MODEL_DIR, "baseline_model.pkl"))
    return vectorizer, clf


def predict(text: str, vectorizer=None, clf=None) -> list:
    if vectorizer is None or clf is None:
        vectorizer, clf = load_model()

    vec = vectorizer.transform([text])
    pred = clf.predict(vec)[0]
    probs = clf.predict_proba(vec)[0]

    return [
        (cat, round(float(prob), 3))
        for cat, flag, prob in zip(CATEGORIES, pred, probs)
        if flag == 1
    ]


if __name__ == "__main__":
    metrics = train()
    print(f"Macro F1: {metrics['macro_f1']:.3f}")
    print(f"Micro F1: {metrics['micro_f1']:.3f}")
    print(metrics["report"])
