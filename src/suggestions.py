"""Generates rule-based suggested rewrites for clauses flagged High risk.

These are template suggestions based on common contract negotiation
practice, not legal advice -- they illustrate what a stronger version of
the clause could require, so a reviewer knows what to ask for.
"""

_SUGGESTIONS = {
    "Renewal Term": (
        "Consider requiring written notice of non-renewal at least 60-90 days "
        "before the term ends, so the contract does not renew automatically "
        "without an opportunity to exit."
    ),
    "Notice Period To Terminate Renewal": (
        "Consider adding an explicit notice period (e.g., 60 days) for "
        "declining renewal, stated in the same clause as the renewal terms."
    ),
    "Uncapped Liability": (
        "Consider negotiating a liability cap (e.g., tied to fees paid in "
        "the prior 12 months) rather than leaving liability unlimited, "
        "except for carve-outs like IP infringement or confidentiality "
        "breaches where uncapped liability is more standard."
    ),
    "Cap On Liability": (
        "If the cap is tied to a short payment window, consider negotiating "
        "a longer look-back period or a minimum fixed cap amount."
    ),
    "Non-Compete": (
        "Consider narrowing the geographic scope (e.g., to markets actually "
        "served) and shortening the duration (6-12 months is more common "
        "than multi-year restrictions)."
    ),
    "Termination For Convenience": (
        "Consider requiring mutual termination rights, or a minimum notice "
        "period, so termination is not one-sided."
    ),
    "Exclusivity": (
        "Consider adding a fixed expiration date or a periodic review right, "
        "rather than leaving an exclusive arrangement open-ended."
    ),
    "Warranty Duration": (
        "Consider negotiating a longer warranty period, or removing broad "
        "'as-is' disclaimer language in favor of a specific coverage period."
    ),
}


def suggest(category: str, risk_level: str) -> str:
    """Returns a suggested rewrite direction for a High-risk clause, or an
    empty string if no suggestion applies (Medium/Low risk, or unknown category)."""
    if risk_level != "High":
        return ""
    return _SUGGESTIONS.get(category, "")
