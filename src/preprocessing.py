"""Extracts and filters CUAD contract clauses into a multi-label training set."""

import json
import re
import os
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "cuad_extracted", "CUADv1.json")
OUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "clauses_dataset.csv")

CATEGORIES = [
    "Termination For Convenience",
    "Renewal Term",
    "Notice Period To Terminate Renewal",
    "Cap On Liability",
    "Uncapped Liability",
    "Non-Compete",
]


def _extract_category(question: str) -> str:
    match = re.search(r'related to "(.*?)"', question)
    return match.group(1) if match else None


def _load_and_filter(raw_path: str = RAW_PATH) -> pd.DataFrame:
    with open(raw_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for contract in data["data"]:
        title = contract["title"]
        for para in contract["paragraphs"]:
            for qa in para["qas"]:
                category = _extract_category(qa["question"])
                if category not in CATEGORIES or qa["is_impossible"]:
                    continue
                for ans in qa["answers"]:
                    rows.append({
                        "contract_title": title,
                        "category": category,
                        "clause_text": ans["text"].strip(),
                    })

    df = pd.DataFrame(rows)
    return df.drop_duplicates(subset=["contract_title", "category", "clause_text"])


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"_{2,}", "", text)
    return text.strip()


def _build_multilabel_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df["clause_text_clean"] = df["clause_text"].apply(_clean_text)
    df = df[df["clause_text_clean"].str.len() > 15]

    pivot = (
        df.groupby(["contract_title", "clause_text_clean"])["category"]
        .apply(set)
        .reset_index()
    )
    for cat in CATEGORIES:
        pivot[cat] = pivot["category"].apply(lambda cats: int(cat in cats))

    return pivot.drop(columns=["category"]).rename(columns={"clause_text_clean": "clause_text"})


def build_dataset() -> pd.DataFrame:
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    long_df = _load_and_filter()
    final_df = _build_multilabel_dataset(long_df)
    final_df.to_csv(OUT_PATH, index=False)
    return final_df


if __name__ == "__main__":
    df = build_dataset()
    print(f"Dataset built: {len(df)} clauses across {len(CATEGORIES)} categories")
    for cat in CATEGORIES:
        print(f"  {cat:38s} {df[cat].sum():4d}")
