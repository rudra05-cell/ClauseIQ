"""Tests for document text extraction and clause splitting."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import document_processing


def test_splits_on_numbered_headers():
    text = """
    1. Termination
    Either party may terminate this Agreement with 30 days notice.

    2. Renewal
    This Agreement renews automatically unless notice is given.
    """
    clauses = document_processing.split_into_clauses(text)
    assert len(clauses) == 2


def test_falls_back_to_paragraph_split_when_no_numbering():
    text = "First paragraph with enough length to pass the minimum threshold check.\n\nSecond paragraph also long enough to pass the minimum length check."
    clauses = document_processing.split_into_clauses(text)
    assert len(clauses) == 2


def test_drops_fragments_below_minimum_length():
    text = "1. Short\nHi\n\n2. Longer\nThis is a properly sized clause with enough content to be kept."
    clauses = document_processing.split_into_clauses(text, min_length=40)
    assert all(len(c) >= 40 for c in clauses)


def test_splits_overlong_sections_by_sentence():
    long_sentence_block = " ".join(
        [f"This is sentence number {i} in a very long clause that keeps going." for i in range(60)]
    )
    clauses = document_processing.split_into_clauses(long_sentence_block, max_length=500)
    assert all(len(c) <= 600 for c in clauses)
    assert len(clauses) > 1


def test_empty_text_returns_no_clauses():
    assert document_processing.split_into_clauses("") == []


def test_unsupported_file_type_raises():
    class FakeFile:
        name = "contract.xyz"

    try:
        document_processing.extract_text(FakeFile())
        assert False, "expected ValueError for unsupported file type"
    except ValueError:
        pass
