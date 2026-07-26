"""
Deterministic (non-LLM) structural validation for HTML-based submissions.

Why this exists
----------------
LLM-based grading (see CODE_EVALUATE_PROMPT in app/graph/nodes.py) is inherently
non-deterministic and has been observed to still award partial credit (e.g. ~15%) to
submissions that are fundamentally broken -- for example, a student who deletes the
teacher-provided opening `<body>` tag from boilerplate/starter HTML. A generative model can
be persuaded by surrounding context (comments, CSS, other markup) that a submission is
"mostly fine" even when the page cannot render correctly at all.

This module performs cheap, deterministic regex-based checks against the actual submitted
HTML source to catch these cases with certainty, and produces a hard ceiling on the final
score that the LLM's opinion cannot override. This runs independently of (and before) the AI
evaluation, so it fails safe even if the model call fails or hallucinates.
"""
import re
from typing import Any, Dict, List, Optional, Tuple

# Tags that constitute a minimally well-formed HTML document. Each entry maps the tag name to
# the score ceiling (0-100) that should be enforced if that tag is entirely missing from a
# submission that is expected to render a full HTML page.
CRITICAL_STRUCTURAL_TAGS: Dict[str, int] = {
    "body": 15,
    "html": 20,
    "head": 40,
}

# Categories / workspace types / evaluation strategies for which structural HTML validation is
# relevant. Question types like SQL, Python, MCQ, Theory, etc. never render a browser page, so
# this check should not run (and must not incorrectly penalize them).
HTML_RELEVANT_CATEGORIES = {"html", "css", "react", "javascript", "typescript", "next.js"}
HTML_RELEVANT_WORKSPACE_TYPES = {"html", "css", "react"}
HTML_RELEVANT_EVALUATION_STRATEGIES = {"ui_playwright"}


def _open_tag_pattern(tag: str) -> re.Pattern:
    # Matches "<tag" optionally followed by attributes/whitespace, but not "<tagXYZ" (e.g. so
    # "body" doesn't accidentally match a custom element like "<bodyish-widget>").
    return re.compile(rf"<\s*{tag}(?:\s[^>]*)?>", re.IGNORECASE)


def _close_tag_pattern(tag: str) -> re.Pattern:
    return re.compile(rf"</\s*{tag}\s*>", re.IGNORECASE)


def _extract_content(file_data: Any) -> str:
    if isinstance(file_data, dict):
        return file_data.get("content", "") or ""
    if isinstance(file_data, str):
        return file_data
    return ""


def _is_html_filename(filename: str) -> bool:
    return filename.lower().endswith((".html", ".htm"))


def _pick_primary_html_file(files: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """
    Choose the HTML file most likely to be the page entry point, mirroring the same
    "index.html first, else any *.html" heuristic used by BrowserRenderer.
    """
    html_files = {name: _extract_content(data) for name, data in (files or {}).items() if _is_html_filename(name)}
    if not html_files:
        return None

    for name, content in html_files.items():
        if name.lower() == "index.html" or name.lower().endswith("index.html"):
            return name, content

    # Fall back to the first HTML file found (dict preserves insertion order).
    name = next(iter(html_files))
    return name, html_files[name]


def is_structural_check_applicable(question_meta: Dict[str, Any]) -> bool:
    meta = question_meta or {}
    category = str(meta.get("category") or "").strip().lower()
    workspace_type = str(meta.get("workspace_type") or "").strip().lower()
    evaluation_strategy = str(meta.get("evaluation_strategy") or "").strip().lower()

    return (
        category in HTML_RELEVANT_CATEGORIES
        or workspace_type in HTML_RELEVANT_WORKSPACE_TYPES
        or evaluation_strategy in HTML_RELEVANT_EVALUATION_STRATEGIES
        or bool(meta.get("preview_required"))
    )


def analyze_html_structure(question_meta: Dict[str, Any], student_files: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare the student's primary HTML file against the teacher's starter HTML file (if any)
    and flag missing/broken critical structural tags.

    Returns:
        {
            "issues": List[str],              # human-readable descriptions for feedback
            "score_cap": Optional[int],        # hard ceiling to apply to the final score, or None
        }
    """
    result = {"issues": [], "score_cap": None}

    if not is_structural_check_applicable(question_meta):
        return result

    student_pick = _pick_primary_html_file(student_files)
    if not student_pick:
        # No HTML file submitted at all for an HTML-relevant question -- this is a total miss.
        result["issues"].append(
            "No HTML file was found in the submission, but this question requires one."
        )
        result["score_cap"] = 0
        return result

    student_filename, student_content = student_pick

    starter_files = (question_meta or {}).get("starter_files") or {}
    starter_content = ""
    if isinstance(starter_files, dict):
        # Prefer the starter file with the same name; otherwise fall back to any starter HTML.
        starter_entry = starter_files.get(student_filename)
        if starter_entry is not None:
            starter_content = _extract_content(starter_entry)
        else:
            starter_pick = _pick_primary_html_file(starter_files)
            if starter_pick:
                starter_content = starter_pick[1]

    applicable_caps: List[int] = []

    for tag, cap in CRITICAL_STRUCTURAL_TAGS.items():
        student_has_open = bool(_open_tag_pattern(tag).search(student_content))
        student_has_close = bool(_close_tag_pattern(tag).search(student_content))

        starter_had_open = bool(_open_tag_pattern(tag).search(starter_content)) if starter_content else False

        if not student_has_open:
            if starter_had_open:
                result["issues"].append(
                    f"The required <{tag}> tag was present in the teacher-provided starter code "
                    f"but is missing from the submission ({student_filename})."
                )
            else:
                result["issues"].append(
                    f"The submission ({student_filename}) is missing the required <{tag}> tag."
                )
            applicable_caps.append(cap)
        elif student_has_open and not student_has_close:
            result["issues"].append(
                f"The <{tag}> tag in {student_filename} is opened but never closed, producing "
                f"invalid/broken HTML structure."
            )
            applicable_caps.append(cap)

    if applicable_caps:
        result["score_cap"] = min(applicable_caps)

    return result
