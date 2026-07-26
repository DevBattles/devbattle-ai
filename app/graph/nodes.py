from app.graph.state import SubmissionState
from app.providers.gemini import GeminiProvider
from app.embeddings.vector_client import VectorClient
from app.vision.browser_renderer import BrowserRenderer
from app.graph.structural_checks import analyze_html_structure
from app.graph.originality_checks import detect_unmodified_submission
from app.utils.logger import logger
from sqlalchemy import text
import uuid
import json

# Instantiate singletons for the LangGraph pipeline
provider = GeminiProvider()
vector_client = VectorClient()
renderer = BrowserRenderer()

VISION_CHECK_PROMPT = """
You are a design and layout expert. Evaluate this screenshot of a rendered student submission.
The question expected output description is: {expected_output}
Evaluate whether the UI correctly implements this layout, checks design completeness, responsiveness indicators, and structural spacing.
Give a visual rating score out of 100, and lists strengths, weaknesses, and structural visual recommendations.
Format response as JSON:
{{
  "visual_score": 85,
  "visual_feedback": "Description of layout alignment and style matches",
  "strengths": ["Visual check 1"],
  "weaknesses": ["Visual check 2"]
}}
Return only valid JSON.
"""


def _parse_llm_json(raw_report: str) -> dict:
    """
    Robustly parse a JSON object out of an LLM text response. Handles the common cases where
    the model wraps its answer in a markdown code fence (` ```json ... ``` `, ` ``` ... ``` `,
    with/without trailing whitespace or a leading language tag with different casing), or adds
    stray prose before/after the JSON object. Raises json.JSONDecodeError if no valid JSON
    object can be recovered.
    """
    clean_str = (raw_report or "").strip()

    # Strip a leading ``` or ```json / ```JSON fence (any casing) and an optional trailing ```.
    if clean_str.startswith("```"):
        first_newline = clean_str.find("\n")
        if first_newline != -1:
            clean_str = clean_str[first_newline + 1:]
        else:
            clean_str = clean_str[3:]
    if clean_str.endswith("```"):
        clean_str = clean_str[:-3]

    clean_str = clean_str.strip()

    try:
        return json.loads(clean_str)
    except json.JSONDecodeError as jde:
        # Try to extract content inside the outermost { } braces in case of stray prose.
        start_idx = clean_str.find('{')
        end_idx = clean_str.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                return json.loads(clean_str[start_idx:end_idx + 1])
            except Exception:
                logger.error(f"Fallback brace parsing failed: {jde}. Raw report was: {raw_report}")
                raise jde
        logger.error(f"JSON brace boundaries not found: {jde}. Raw report was: {raw_report}")
        raise jde

CODE_EVALUATE_PROMPT = """
You are a Principal AI backend grading assistant.
Evaluate the following student submission:
Files: {student_files}

Question Category: {category}
Evaluation Strategy: {evaluation_strategy}

Compare it against:
1. Question metadata:
Title: {title}
Description: {description}
Requirements: {requirements}

2. Rubric details:
{rubric}

3. Similar reference solutions:
{similar_solutions}

You MUST follow the specific evaluation strategy for the "{category}" category:
- For JavaScript/Python/C++: Evaluate code correctness, logic accuracy, execution output correctness, edge cases, and time/space complexity.
- For SQL: Evaluate the query structure, join correctness, filtering constraints, and expected output dataset.
- For React/HTML/CSS: Evaluate the component structure, DOM layout semantics, visual layout, responsive styles, and browser rendering.
- For Theory: Perform a semantic comparison of the student's explanation/text against the expected answer, utilizing the rubric. Do not execute any code.
- For MCQ: Simply compare the selected choice directly.

SCORING CRITERIA:
Evaluate the code holistically based on correctness, edge cases, requirements, code quality, and performance.
Give a single final `score` integer between 0 and 100.

STRICT SCORING RULES:
- If the submission is entirely blank or just unmodified starter code, the score MUST be 0.
- Otherwise, evaluate it fairly on a 0-100 scale.
- NEVER fabricate high scores for incorrect solutions.

Format response as JSON:
{{
  "score": 0,
  "strengths": ["Detail 1"],
  "weaknesses": ["Detail 2"],
  "improvements": ["Detail 3"],
  "feedback": "Overall summary feedback"
}}
Return only valid JSON.
"""

async def validate_input_node(state: SubmissionState) -> dict:
    logger.info("Executing validate_input node...")
    qid = state.get("question_id")
    ver = state.get("version")
    files = state.get("student_files")

    # NOTE: use explicit None/falsy-container checks rather than `not ver`, since version 0 is a
    # legitimate (if unusual) integer version and Python truthiness treats 0 the same as missing.
    if not qid or ver is None or not files:
        return {"error": "Validation failed: question_id, version, and student_files must be supplied"}
    
    try:
        uuid.UUID(qid)
    except ValueError:
        return {"error": "Validation failed: question_id must be a valid UUID"}

    return {}

async def retrieve_question_node(state: SubmissionState) -> dict:
    logger.info("Executing retrieve_question node...")
    qid = state["question_id"]
    ver = state["version"]
    
    try:
        async with vector_client.async_session() as session:
            result = await session.execute(
                text("""
                    SELECT title, description, starter_files, expected_output,
                           category, workspace_type, evaluation_strategy, supported_language,
                           preview_required, execution_mode, options
                    FROM question_versions 
                    WHERE question_id = :qid AND version = :ver
                """),
                {"qid": uuid.UUID(qid), "ver": ver}
            )
            row = result.fetchone()
            
            if not row:
                # Fallback to question_bank main record if version entry not found
                fallback_result = await session.execute(
                    text("""
                        SELECT title, description, expected_output,
                               category, workspace_type, evaluation_strategy, supported_language,
                               preview_required, execution_mode, options
                        FROM question_bank 
                        WHERE id = :qid
                    """),
                    {"qid": uuid.UUID(qid)}
                )
                fb_row = fallback_result.fetchone()
                if not fb_row:
                    return {"error": f"Question {qid} not found in database"}
                
                meta = {
                    "title": _row_get(fb_row, 0, ""),
                    "description": _row_get(fb_row, 1, ""),
                    "starter_files": {},
                    "expected_output": _row_get(fb_row, 2, "") or "",
                    "category": _row_get(fb_row, 3),
                    "workspace_type": _row_get(fb_row, 4),
                    "evaluation_strategy": _row_get(fb_row, 5),
                    "supported_language": _row_get(fb_row, 6),
                    "preview_required": _row_get(fb_row, 7, False) if _row_get(fb_row, 7, False) is not None else False,
                    "execution_mode": _row_get(fb_row, 8),
                    "options": _row_get(fb_row, 9) if _row_get(fb_row, 9) else None
                }
            else:
                meta = {
                    "title": _row_get(row, 0, ""),
                    "description": _row_get(row, 1, ""),
                    "starter_files": _row_get(row, 2, {}) or {},
                    "expected_output": _row_get(row, 3, "") or "",
                    "category": _row_get(row, 4),
                    "workspace_type": _row_get(row, 5),
                    "evaluation_strategy": _row_get(row, 6),
                    "supported_language": _row_get(row, 7),
                    "preview_required": _row_get(row, 8, False) if _row_get(row, 8, False) is not None else False,
                    "execution_mode": _row_get(row, 9),
                    "options": _row_get(row, 10) if _row_get(row, 10) else None
                }
            
            return {"question_meta": meta}
    except Exception as e:
        logger.error(f"Retrieve question failed: {e}")
        return {"error": f"Database read exception in retrieve_question: {str(e)}"}

async def retrieve_rubric_node(state: SubmissionState) -> dict:
    logger.info("Executing retrieve_rubric node...")
    qid = state["question_id"]
    ver = state["version"]
    
    try:
        async with vector_client.async_session() as session:
            result = await session.execute(
                text("SELECT rubric FROM question_versions WHERE question_id = :qid AND version = :ver"),
                {"qid": uuid.UUID(qid), "ver": ver}
            )
            row = result.fetchone()
            rubric = row[0] if row and row[0] else {}
            
            # If no rubric, create a default rubric fallback
            if not rubric:
                rubric = {
                    "correctness": {"max_points": 30, "checklist": ["Code resolves requirements"]},
                    "responsiveness": {"max_points": 20, "checklist": ["Styles fit viewports"]},
                    "accessibility": {"max_points": 15, "checklist": ["ARIA roles used"]},
                    "performance": {"max_points": 15, "checklist": ["Minimized code weights"]},
                    "code_quality": {"max_points": 20, "checklist": ["Best practices clean code"]}
                }
            return {"rubric": rubric}
    except Exception as e:
        logger.warning(f"Failed to load rubric, using fallback: {e}")
        return {"rubric": {}}

def _extract_file_content(file_data) -> str:
    """
    Student file entries are expected to look like {"content": "..."}, but studentFiles is a
    loosely-typed Dict[str, Any] at the API boundary, so tolerate a bare string value instead
    of crashing with AttributeError.
    """
    if isinstance(file_data, dict):
        return file_data.get("content", "") or ""
    if isinstance(file_data, str):
        return file_data
    return "" if file_data is None else str(file_data)


async def retrieve_similar_solutions_node(state: SubmissionState) -> dict:
    logger.info("Executing retrieve_similar_solutions node...")
    qid = state["question_id"]
    files = state["student_files"]

    # Concat all student code to make embedding search index robust
    student_code = ""
    for name, f_data in files.items():
        student_code += f"\n// File: {name}\n" + _extract_file_content(f_data)

    try:
        embedding = await provider.get_embedding(student_code)
        solutions = await vector_client.get_similar_solutions(qid, embedding, limit=3)
        return {"similar_solutions": solutions}
    except Exception as e:
        logger.warning(f"pgvector retrieval failed: {e}. Continuing evaluation without similar solutions.")
        return {"similar_solutions": []}

async def structural_validation_node(state: SubmissionState) -> dict:
    """
    Deterministic guardrail that runs before the LLM-based evaluators. Combines two checks
    that the LLM grader has been observed to not reliably self-enforce, even though its own
    prompt instructs it to:
      1. Structural HTML validation (app/graph/structural_checks.py) -- e.g. a missing/removed
         <body> tag that makes the page fundamentally broken.
      2. Unmodified/blank submission detection (app/graph/originality_checks.py) -- the student
         submitted the teacher's starter code as-is (or left it blank), which per the grading
         prompt's own "STRICT SCORING RULES" must score 0, but the LLM can still hallucinate
         partial credit for it.
    Both checks produce a hard ceiling on the final score that the LLM's opinion cannot
    override (enforced later in aggregate_scores_node).
    """
    logger.info("Executing structural_validation node...")
    files = state["student_files"]
    meta = state.get("question_meta", {}) or {}

    issues: list = []
    caps: list = []

    try:
        structural_analysis = analyze_html_structure(meta, files)
        issues.extend(structural_analysis["issues"])
        if structural_analysis["score_cap"] is not None:
            caps.append(structural_analysis["score_cap"])
    except Exception as e:
        # Never let a bug in this deterministic check block the rest of the pipeline.
        logger.error(f"Structural HTML validation check failed unexpectedly: {e}")

    try:
        originality_analysis = detect_unmodified_submission(meta, files)
        issues.extend(originality_analysis["issues"])
        if originality_analysis["score_cap"] is not None:
            caps.append(originality_analysis["score_cap"])
    except Exception as e:
        logger.error(f"Unmodified-submission check failed unexpectedly: {e}")

    score_cap = min(caps) if caps else None

    if issues:
        logger.warning(f"Structural validation found issues: {issues} (score cap: {score_cap})")

    return {
        "structural_issues": issues,
        "structural_score_cap": score_cap
    }

async def vision_check_node(state: SubmissionState) -> dict:
    logger.info("Executing vision_check node...")
    files = state["student_files"]
    meta = state.get("question_meta", {})
    if not meta.get("preview_required"):
        logger.info("Bypassing vision check node.")
        return {"visual_evaluation": None}
    expected_output = meta.get("expected_output", "A responsive webpage matching instructions.")

    try:
        screenshot_bytes = await renderer.capture_screenshot(files)
        if not screenshot_bytes:
            logger.warning("Playwright failed to capture page preview.")
            return {"visual_evaluation": {"visual_score": 75, "visual_feedback": "Preview layout rendering failed. Vision check skipped."}}

        prompt = VISION_CHECK_PROMPT.format(expected_output=expected_output)
        raw_report = await provider.generate_multimodal(prompt, screenshot_bytes)

        parsed = _parse_llm_json(raw_report)
        return {
            "screenshot_bytes": screenshot_bytes,
            "visual_evaluation": parsed
        }
    except Exception as e:
        logger.error(f"Vision check failed: {e}")
        return {"visual_evaluation": {"visual_score": 80, "visual_feedback": f"Vision error: {str(e)}"}}

async def gemini_evaluate_node(state: SubmissionState) -> dict:
    logger.info("Executing gemini_evaluate node...")
    files = state["student_files"]
    meta = state.get("question_meta", {})
    rubric = state.get("rubric", {})
    solutions = state.get("similar_solutions", [])

    # Format checklists parameters
    prompt = CODE_EVALUATE_PROMPT.format(
        student_files=json.dumps(files),
        category=meta.get("category", "General"),
        evaluation_strategy=meta.get("evaluation_strategy", "ui_playwright"),
        title=meta.get("title", ""),
        description=meta.get("description", ""),
        requirements=meta.get("requirements", ""),
        rubric=json.dumps(rubric),
        similar_solutions=json.dumps(solutions)
    )

    try:
        raw_report = await provider.generate_text(
            prompt=prompt,
            system_instruction="You are a Principal AI backend grading assistant. Always return valid JSON matching schemas exactly.",
            json_mode=True
        )

        parsed = _parse_llm_json(raw_report)

        return {"code_evaluation": parsed}
    except Exception as e:
        logger.error(f"Code evaluation failed: {e}")
        return {
            "code_evaluation": {
                "score": 0,
                "strengths": [],
                "weaknesses": [f"Code evaluation unavailable: {str(e)}"],
                "improvements": ["Retry the evaluation after the model service recovers."],
                "feedback": "Code evaluation could not be completed because the model service failed."
            }
        }


def _extract_metric_score(code_eval: dict, metric_name: str, fallback_score: float = 0.0) -> float:
    metric_value = code_eval.get(metric_name)
    if isinstance(metric_value, dict):
        try:
            return float(metric_value.get("score", metric_value.get("value", fallback_score)))
        except Exception:
            return fallback_score
    if isinstance(metric_value, (int, float)):
        return float(metric_value)
    return fallback_score


def _row_get(row, index: int, default=None):
    try:
        if row is None:
            return default
        if hasattr(row, "__len__") and len(row) <= index:
            return default
        return row[index]
    except Exception:
        return default

def _safe_list(value) -> list:
    """
    Coerce a value that should be a list of strings into an actual list, tolerating None
    (which `dict.get(key, default)` does NOT catch when the key is present but explicitly
    null -- a real possibility since this data originates from LLM-generated JSON) as well
    as a single bare string.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [value]
    return []


def _safe_float(value, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


async def aggregate_scores_node(state: SubmissionState) -> dict:
    logger.info("Executing aggregate_scores node...")
    code_eval = state.get("code_evaluation")
    visual_eval = state.get("visual_evaluation")

    if not code_eval:
        code_eval = {
            "score": 0,
            "strengths": [],
            "weaknesses": ["Code evaluation data was unavailable."],
            "improvements": ["Retry the evaluation after the grader service recovers."],
            "feedback": "Code evaluation could not be completed."
        }

    # Extract score
    raw_code_score = _safe_float(code_eval.get("score"), default=0.0)

    correctness_score = _extract_metric_score(code_eval, "correctness")
    edge_score = _extract_metric_score(code_eval, "edge_cases")
    req_score = _extract_metric_score(code_eval, "requirements")
    qual_score = _extract_metric_score(code_eval, "code_quality")
    perf_score = _extract_metric_score(code_eval, "performance")

    derived_component_total = correctness_score + edge_score + req_score + qual_score + perf_score
    if raw_code_score <= 0 and derived_component_total > 0:
        raw_code_score = derived_component_total

    if visual_eval:
        visual_score = _safe_float(visual_eval.get("visual_score"), default=100.0)
        aggregated = (raw_code_score * 0.8) + (visual_score * 0.2)
    else:
        visual_score = 0.0
        aggregated = raw_code_score

    final_score = int(round(aggregated))

    # Hard guardrail: deterministic structural HTML validation (see
    # app/graph/structural_checks.py) can enforce a ceiling on the final score regardless of
    # what the LLM grader concluded -- e.g. a submission with a deleted <body> tag should never
    # score above 15%, even if the AI evaluator was persuaded otherwise.
    structural_issues = state.get("structural_issues") or []
    structural_score_cap = state.get("structural_score_cap")
    if structural_score_cap is not None and final_score > structural_score_cap:
        logger.warning(
            f"Capping final score from {final_score} to {structural_score_cap} due to structural "
            f"HTML validation failures: {structural_issues}"
        )
        final_score = structural_score_cap

    # Apply grading standard
    if final_score >= 90:
        grade = "A"
    elif final_score >= 80:
        grade = "B"
    elif final_score >= 70:
        grade = "C"
    elif final_score >= 60:
        grade = "D"
    else:
        grade = "F"

    # Merge feedback parameters. `code_eval.get(key, default)` only falls back when the key is
    # absent -- if the LLM emits an explicit JSON `null` (which it is free to do; nothing
    # enforces the response schema), these would otherwise stay None and crash the slicing /
    # extend() calls below.
    strengths = _safe_list(code_eval.get("strengths"))
    weaknesses = _safe_list(code_eval.get("weaknesses"))
    improvements = _safe_list(code_eval.get("improvements"))
    
    if visual_eval:
        strengths = strengths + _safe_list(visual_eval.get("strengths"))
        weaknesses = weaknesses + _safe_list(visual_eval.get("weaknesses"))

    if structural_issues:
        # Surface the deterministic findings prominently so the student understands exactly
        # why their score is capped, independent of whatever the LLM said.
        weaknesses = structural_issues + weaknesses
        improvements = [
            "Make sure you actually modify the starter code to solve the challenge, and keep "
            "the HTML structure intact (matching opening/closing tags) -- these issues are "
            "checked automatically and cap your score regardless of other feedback."
        ] + improvements

    feedback = code_eval.get("feedback") or "Excellent effort. Recheck weaknesses and improvements checklist."


    rubric_scores = {
        "correctness": correctness_score,
        "edge_cases": edge_score,
        "requirements": req_score,
        "code_quality": qual_score,
        "performance": perf_score,
        "visual_comparison": visual_score
    }

    return {
        "score": final_score,
        "grade": grade,
        "strengths": strengths[:5], # cap at top 5
        "weaknesses": weaknesses[:5],
        "improvements": improvements[:5],
        "feedback": feedback,
        "rubric_scores": rubric_scores
    }

async def generate_report_node(state: SubmissionState) -> dict:
    logger.info("Executing generate_report node...")
    # This node finalizes state returns.
    return {
        "score": state.get("score"),
        "grade": state.get("grade"),
        "strengths": state.get("strengths"),
        "weaknesses": state.get("weaknesses"),
        "improvements": state.get("improvements"),
        "rubric_scores": state.get("rubric_scores"),
        "feedback": state.get("feedback")
    }
