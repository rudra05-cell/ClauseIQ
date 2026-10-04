"""Tests for feedback logging."""

import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import feedback


def test_log_and_summarize_feedback(monkeypatch, tmp_path):
    log_path = tmp_path / "feedback_log.csv"
    monkeypatch.setattr(feedback, "FEEDBACK_PATH", str(log_path))

    feedback.log_feedback("Some clause", "Renewal Term", "High", "correct")
    feedback.log_feedback("Another clause", "Non-Compete", "Medium", "incorrect")

    summary = feedback.load_feedback_summary()
    assert summary["correct"] == 1
    assert summary["incorrect"] == 1
    assert summary["total"] == 2


def test_summary_with_no_feedback_file(monkeypatch, tmp_path):
    log_path = tmp_path / "does_not_exist.csv"
    monkeypatch.setattr(feedback, "FEEDBACK_PATH", str(log_path))

    summary = feedback.load_feedback_summary()
    assert summary == {"correct": 0, "incorrect": 0, "total": 0}
