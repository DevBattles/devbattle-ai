from typing import TypedDict, List, Dict, Any, Optional

class SubmissionState(TypedDict):
    # Inputs
    question_id: str
    version: int
    student_files: Dict[str, Any]
    github_url: Optional[str]
    
    # Context loaded at runtime
    question_meta: Optional[Dict[str, Any]]
    rubric: Optional[Dict[str, Any]]
    similar_solutions: Optional[List[Dict[str, Any]]]
    screenshot_bytes: Optional[bytes]
    
    # Internal node evaluation outputs
    code_evaluation: Optional[Dict[str, Any]]
    visual_evaluation: Optional[Dict[str, Any]]

    # Deterministic (non-LLM) structural HTML validation, e.g. detecting a student who deleted
    # a critical boilerplate tag such as <body> that was present in the teacher's starter files.
    # This exists as a hard guardrail because LLM grading alone has been observed to still hand
    # out partial credit (e.g. ~15%) for a submission with fundamentally broken HTML structure.
    structural_issues: Optional[List[str]]
    structural_score_cap: Optional[int]

    # Final aggregated reports
    score: Optional[int]
    grade: Optional[str]
    strengths: Optional[List[str]]
    weaknesses: Optional[List[str]]
    improvements: Optional[List[str]]
    rubric_scores: Optional[Dict[str, float]]
    feedback: Optional[str]
    error: Optional[str]
