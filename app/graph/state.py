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
    
    # Final aggregated reports
    score: Optional[int]
    grade: Optional[str]
    strengths: Optional[List[str]]
    weaknesses: Optional[List[str]]
    improvements: Optional[List[str]]
    rubric_scores: Optional[Dict[str, float]]
    feedback: Optional[str]
    error: Optional[str]
