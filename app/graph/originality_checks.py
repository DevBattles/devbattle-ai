"""Deterministic checks for blank or unmodified starter-code submissions.

The LLM grader is instructed to give these submissions a zero, but that rule is important
enough to enforce before the model's result is aggregated.  The check is deliberately
conservative: a submission is only called unmodified when its complete set of submitted
files matches the teacher-provided starter files after insignificant whitespace differences.
"""
from __future__ import annotations

import re
from typing import Any, Dict


# Whitespace-only changes do not demonstrate that the student implemented the challenge.
_WHITESPACE = re.compile(r"\s+")


def _extract_content(file_data: Any) -> str:
    """Return a file's source while tolerating the API's string-or-object file shape."""
    if isinstance(file_data, dict):
        value = file_data.get("content", "")
    elif isinstance(file_data, str):
        value = file_data
    else:
        value = ""
    return value if isinstance(value, str) else str(value or "")


def _normalized_files(files: Dict[str, Any]) -> Dict[str, str]:
    """Normalize paths and insignificant source whitespace for a deterministic comparison."""
    if not isinstance(files, dict):
        return {}
    return {
        str(filename).replace("\\", "/").lstrip("./"): _WHITESPACE.sub("", _extract_content(data))
        for filename, data in files.items()
    }


def _is_blank(files: Dict[str, Any]) -> bool:
    """A missing file collection or files containing only whitespace is a blank submission."""
    return not files or not any(_extract_content(data).strip() for data in files.values())


def detect_unmodified_submission(
    question_meta: Dict[str, Any], student_files: Dict[str, Any]
) -> Dict[str, Any]:
    """Detect a blank submission or one that is unchanged from its starter files.

    Returns the same ``issues``/``score_cap`` shape as structural checks so callers can merge
    deterministic guardrails.  A score cap of zero is intentional: neither blank work nor a
    copied starter has evidence of a solution to grade.
    """
    result = {"issues": [], "score_cap": None}
    files = student_files if isinstance(student_files, dict) else {}

    if _is_blank(files):
        result["issues"].append(
            "The submission is blank. Add your solution before submitting for grading."
        )
        result["score_cap"] = 0
        return result

    starter_files = (question_meta or {}).get("starter_files") or {}
    if not isinstance(starter_files, dict) or not starter_files:
        # There is no baseline to compare against. A non-blank submission should be graded.
        return result

    if _normalized_files(files) == _normalized_files(starter_files):
        result["issues"].append(
            "The submission is unchanged from the teacher-provided starter code."
        )
        result["score_cap"] = 0

    return result
