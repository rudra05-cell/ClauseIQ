"""Rule-based risk assessment applied to classified contract clauses."""

import re

NOTICE_PATTERN = re.compile(
    r"\b(\d+|thirty|sixty|ninety|forty[- ]?five)\s*(day|days)\b.*\bnotice\b|"
    r"\bnotice\b.*\b(\d+|thirty|sixty|ninety|forty[- ]?five)\s*(day|days)\b",
    re.IGNORECASE,
)
AUTOMATIC_LANGUAGE = re.compile(r"\bautomatically renew|\bauto[- ]renew|\bshall renew\b", re.IGNORECASE)
UNLESS_TERMINATED = re.compile(r"\bunless\b.*\b(terminat|cancel)", re.IGNORECASE)
LOW_CAP_PATTERN = re.compile(r"\bfees paid\b|\bamounts paid\b.*\b(12|twelve|6|six|3|three)\s*month", re.IGNORECASE)
UNCAPPED_LANGUAGE = re.compile(r"\bno limit\b|\bunlimited liability\b|\bwithout limitation\b.*\bliab", re.IGNORECASE)
BROAD_SCOPE = re.compile(r"\bany business\b|\bany capacity\b|\bworldwide\b", re.IGNORECASE)
LONG_DURATION = re.compile(r"\b([2-9]|[1-9]\d)\s*year", re.IGNORECASE)
MUTUAL_LANGUAGE = re.compile(r"\beither party\b|\bboth parties\b", re.IGNORECASE)
ONE_SIDED_LANGUAGE = re.compile(r"\b(Company|Licensor|Provider)\s+may\s+terminate\b", re.IGNORECASE)


def _score_renewal(text: str) -> dict:
    has_auto = bool(AUTOMATIC_LANGUAGE.search(text)) or bool(UNLESS_TERMINATED.search(text))
    has_notice = bool(NOTICE_PATTERN.search(text))
    if has_auto and not has_notice:
        return {"level": "High", "reason": "Auto-renewal language is present with no clearly defined notice period."}
    if has_auto and has_notice:
        return {"level": "Low", "reason": "Auto-renewal is present, but a notice period is clearly defined."}
    return {"level": "Medium", "reason": "Renewal terms are present; review manually for notice requirements."}


def _score_liability(text: str) -> dict:
    if UNCAPPED_LANGUAGE.search(text):
        return {"level": "High", "reason": "Clause language suggests liability may be uncapped or unlimited."}
    if LOW_CAP_PATTERN.search(text):
        return {"level": "Medium", "reason": "Liability is capped to a short historical payment window, which may be low relative to potential damages."}
    return {"level": "Low", "reason": "A liability cap is present with no red-flag indicators."}


def _score_noncompete(text: str) -> dict:
    is_broad = bool(BROAD_SCOPE.search(text))
    is_long = bool(LONG_DURATION.search(text))
    if is_broad and is_long:
        return {"level": "High", "reason": "Broad scope (worldwide / any business) combined with a long duration (2+ years)."}
    if is_broad or is_long:
        return {"level": "Medium", "reason": "Scope or duration appears broad — review the specific limits."}
    return {"level": "Low", "reason": "Scope and duration appear reasonably limited."}


def _score_termination(text: str) -> dict:
    if ONE_SIDED_LANGUAGE.search(text) and not MUTUAL_LANGUAGE.search(text):
        return {"level": "Medium", "reason": "Termination rights may be one-sided — confirm both parties hold equal rights."}
    return {"level": "Low", "reason": "Termination rights appear mutual."}


_SCORERS = {
    "Renewal Term": _score_renewal,
    "Notice Period To Terminate Renewal": _score_renewal,
    "Cap On Liability": _score_liability,
    "Uncapped Liability": _score_liability,
    "Non-Compete": _score_noncompete,
    "Termination For Convenience": _score_termination,
}


def assess(text: str, categories: list) -> list:
    """Returns a risk assessment for each predicted category found in the clause."""
    results = []
    for category in categories:
        scorer = _SCORERS.get(category)
        if scorer:
            results.append({"category": category, **scorer(text)})
    return results
