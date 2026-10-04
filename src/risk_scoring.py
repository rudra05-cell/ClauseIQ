"""Rule-based risk assessment applied to classified contract clauses."""

import re

# --- Renewal / notice ---------------------------------------------------
NOTICE_PATTERN = re.compile(
    r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirty|forty[- ]?five|sixty|ninety)\s*(?:\(\d+\))?\s*(day|days|month|months|year|years)\b.{0,60}?\bnotice\b|"
    r"\bnotice\b.{0,60}?\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirty|forty[- ]?five|sixty|ninety)\s*(?:\(\d+\))?\s*(day|days|month|months|year|years)\b",
    re.IGNORECASE,
)
AUTOMATIC_LANGUAGE = re.compile(r"\bautomatically renew|\bauto[- ]renew|\bshall renew\b", re.IGNORECASE)
UNLESS_TERMINATED = re.compile(r"\bunless\b.*\b(terminat|cancel)", re.IGNORECASE)

# --- Liability ------------------------------------------------------------
LOW_CAP_PATTERN = re.compile(r"\bfees paid\b|\bamounts paid\b.*\b(12|twelve|6|six|3|three)\s*month", re.IGNORECASE)
# Bounded gaps (not .*) so these don't span unrelated text in a long clause.
UNCAPPED_LANGUAGE = re.compile(
    r"\bunlimited liability\b|"
    r"\bliability\s+shall\s+not\s+be\s+limited\b|"
    r"\bno\s+limit(?:ation)?\s+(?:shall\s+apply\s+to|on|to)\s+.{0,30}?liabilit|"
    r"\bthere\s+shall\s+be\s+no\s+limit\s+on\s+.{0,30}?liabilit|"
    r"\bno\s+cap\s+on\s+.{0,30}?liabilit",
    re.IGNORECASE,
)
# A damages-exclusion clause or an explicit cap statement is protective
# language, not a red flag.
DAMAGES_EXCLUSION_OR_CAP = re.compile(
    r"\b(indirect|incidental|consequential|punitive|special)\s+damages\b|"
    r"\bmaximum aggregate liability\b|"
    r"\bshall not exceed\b",
    re.IGNORECASE,
)

# --- Non-compete ------------------------------------------------------------
BROAD_SCOPE = re.compile(r"\bany business\b|\bany capacity\b|\bworldwide\b", re.IGNORECASE)
LONG_DURATION = re.compile(r"\b([2-9]|[1-9]\d)\s*year", re.IGNORECASE)

# --- Termination ------------------------------------------------------------
MUTUAL_LANGUAGE = re.compile(r"\beither party\b|\bboth parties\b", re.IGNORECASE)
ONE_SIDED_LANGUAGE = re.compile(
    r"\b(Company|Licensor|Provider|Manufacturer|Landlord|Buyer|Vendor|Employer|Client|Supplier|Distributor)\s+may\s+terminate\b",
    re.IGNORECASE,
)

# --- Audit rights ------------------------------------------------------------
UNRESTRICTED_AUDIT = re.compile(r"\bat any time\b.{0,40}?\baudit\b|\baudit\b.{0,40}?\bat any time\b|\bwithout (?:prior )?notice\b.{0,40}?\baudit\b", re.IGNORECASE)
REASONABLE_AUDIT = re.compile(r"\breasonable notice\b|\bonce\s+(?:per|a|each)\s+year\b|\bannually\b|\bduring (?:normal|regular) business hours\b", re.IGNORECASE)

# --- Insurance ------------------------------------------------------------
SPECIFIC_COVERAGE_AMOUNT = re.compile(r"\$[\d,]+(?:\.\d+)?|\b\d+\s*(?:million|thousand)\b", re.IGNORECASE)

# --- Exclusivity ------------------------------------------------------------
INDEFINITE_LANGUAGE = re.compile(r"\bperpetuity\b|\bperpetual\b|\bno expiration\b|\bindefinite(?:ly)?\b", re.IGNORECASE)

# --- Warranty ------------------------------------------------------------
SHORT_WARRANTY = re.compile(r"\b(\d{1,2}|thirty|sixty)\s*days?\b", re.IGNORECASE)
DISCLAIMED_WARRANTY = re.compile(r"\bas[- ]is\b|\bno warrant(?:y|ies)\b|\bdisclaims? (?:all )?warrant", re.IGNORECASE)

# --- Volume restriction ------------------------------------------------------------
PENALTY_LANGUAGE = re.compile(r"\bpenalty\b|\bshortfall\b|\bfail(?:ure|s)? to (?:purchase|order|meet)\b.{0,40}?\b(pay|penalty|charge)\b", re.IGNORECASE)
FLEXIBLE_LANGUAGE = re.compile(r"\bno obligation\b|\bbest efforts\b|\bmay, but is not required\b", re.IGNORECASE)


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
    if DAMAGES_EXCLUSION_OR_CAP.search(text):
        return {"level": "Low", "reason": "Clause explicitly excludes certain damages or states a liability cap — protective language."}
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


def _score_audit_rights(text: str) -> dict:
    if UNRESTRICTED_AUDIT.search(text) and not REASONABLE_AUDIT.search(text):
        return {"level": "Medium", "reason": "Audit rights appear unrestricted in timing or frequency — consider requiring advance notice and reasonable scheduling."}
    return {"level": "Low", "reason": "Audit rights include reasonable limits on timing or frequency."}


def _score_insurance(text: str) -> dict:
    if not SPECIFIC_COVERAGE_AMOUNT.search(text):
        return {"level": "Medium", "reason": "An insurance obligation is present but no specific coverage amount is stated — the requirement may be ambiguous."}
    return {"level": "Low", "reason": "A specific insurance coverage amount is stated."}


def _score_exclusivity(text: str) -> dict:
    is_broad = bool(BROAD_SCOPE.search(text))
    is_indefinite = bool(INDEFINITE_LANGUAGE.search(text))
    if is_broad and is_indefinite:
        return {"level": "High", "reason": "Exclusivity is broad in scope and has no stated end date."}
    if is_broad or is_indefinite:
        return {"level": "Medium", "reason": "Exclusivity scope or duration appears broad — review the specific limits."}
    return {"level": "Low", "reason": "Exclusivity scope and duration appear reasonably limited."}


def _score_warranty(text: str) -> dict:
    if DISCLAIMED_WARRANTY.search(text):
        return {"level": "High", "reason": "Warranty appears to be disclaimed or provided on an as-is basis."}
    if SHORT_WARRANTY.search(text):
        return {"level": "Medium", "reason": "Warranty duration appears short — confirm it provides adequate coverage."}
    return {"level": "Low", "reason": "Warranty duration appears standard."}


def _score_volume_restriction(text: str) -> dict:
    if PENALTY_LANGUAGE.search(text) and not FLEXIBLE_LANGUAGE.search(text):
        return {"level": "Medium", "reason": "A minimum volume commitment is tied to a penalty or shortfall charge for non-compliance."}
    return {"level": "Low", "reason": "Volume terms do not appear to carry a penalty for shortfalls."}


_SCORERS = {
    "Renewal Term": _score_renewal,
    "Notice Period To Terminate Renewal": _score_renewal,
    "Cap On Liability": _score_liability,
    "Uncapped Liability": _score_liability,
    "Non-Compete": _score_noncompete,
    "Termination For Convenience": _score_termination,
    "Audit Rights": _score_audit_rights,
    "Insurance": _score_insurance,
    "Exclusivity": _score_exclusivity,
    "Warranty Duration": _score_warranty,
    "Volume Restriction": _score_volume_restriction,
}


def assess(text: str, categories: list) -> list:
    """Returns a risk assessment for each predicted category found in the clause."""
    results = []
    for category in categories:
        scorer = _SCORERS.get(category)
        if scorer:
            results.append({"category": category, **scorer(text)})
    return results
