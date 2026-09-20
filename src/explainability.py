"""Highlights the terms or tokens driving a clause's classification."""

import numpy as np

_SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]"}


def from_tfidf(text: str, vectorizer, top_n: int = 6) -> list:
    feature_names = np.array(vectorizer.get_feature_names_out())
    row = vectorizer.transform([text]).toarray()[0]
    top_idx = [i for i in row.argsort()[::-1][:top_n] if row[i] > 0]
    return [(feature_names[i], round(float(row[i]), 3)) for i in top_idx]


def from_attention(token_weight_pairs: list, top_n: int = 8) -> list:
    filtered = [(tok, w) for tok, w in token_weight_pairs if tok not in _SPECIAL_TOKENS]
    filtered.sort(key=lambda pair: -pair[1])
    return [(tok.replace("##", ""), round(float(w), 4)) for tok, w in filtered[:top_n]]
