"""Compares the trained classifier against prompting an LLM directly on the
same clause -- an optional module for exploring the "why not just use an
LLM" question empirically rather than by assertion.

This requires an Anthropic API key set as the ANTHROPIC_API_KEY environment
variable. Without one, comparisons fall back gracefully with a clear
message rather than crashing, so the rest of the app is unaffected.

This is intentionally a comparison tool, not a replacement classifier --
the trained model remains the one used throughout the rest of the app.
"""

import os
import json

from src.baseline_model import CATEGORIES

_PROMPT_TEMPLATE = """You are analyzing a single contract clause. The possible categories are:
{categories}

Clause:
\"\"\"{clause}\"\"\"

Return ONLY a JSON array of category names (from the list above) that apply to this clause. If none apply, return an empty array. No explanation, no markdown, just the JSON array."""


def is_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def classify_with_llm(text: str) -> list:
    """Returns a list of category names the LLM assigned to this clause.
    Raises RuntimeError with a clear message if no API key is configured."""
    if not is_available():
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. This comparison feature requires "
            "an API key and is optional -- the rest of ClauseIQ does not "
            "depend on it."
        )

    import anthropic

    client = anthropic.Anthropic()
    prompt = _PROMPT_TEMPLATE.format(categories="\n".join(f"- {c}" for c in CATEGORIES), clause=text)

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    try:
        categories = json.loads(raw)
        return [c for c in categories if c in CATEGORIES]
    except json.JSONDecodeError:
        return []


def compare(text: str, classical_categories: list) -> dict:
    """Returns a comparison dict: classical prediction, LLM prediction (or
    an error message if unavailable), and whether they agree."""
    result = {
        "classical": classical_categories,
        "llm": None,
        "llm_error": None,
        "agree": None,
    }

    try:
        llm_categories = classify_with_llm(text)
        result["llm"] = llm_categories
        result["agree"] = set(classical_categories) == set(llm_categories)
    except RuntimeError as e:
        result["llm_error"] = str(e)

    return result
