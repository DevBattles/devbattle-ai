from app.graph.state import SubmissionState
from app.providers.gemini import GeminiProvider
from app.embeddings.vector_client import VectorClient
from app.vision.browser_renderer import BrowserRenderer
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

    if not qid or not ver or not files:
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

async def retrieve_similar_solutions_node(state: SubmissionState) -> dict:
    logger.info("Executing retrieve_similar_solutions node...")
    qid = state["question_id"]
    files = state["student_files"]

    # Concat all student code to make embedding search index robust
    student_code = ""
    for name, f_data in files.items():
        student_code += f"\n// File: {name}\n" + f_data.get("content", "")

    try:
        embedding = await provider.get_embedding(student_code)
        solutions = await vector_client.get_similar_solutions(qid, embedding, limit=3)
        return {"similar_solutions": solutions}
    except Exception as e:
        logger.warning(f"pgvector retrieval failed: {e}. Continuing evaluation without similar solutions.")
        return {"similar_solutions": []}

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
        
        # Parse JSON
        clean_str = raw_report.strip()
        if clean_str.startswith("```json"):
            clean_str = clean_str[7:]
        if clean_str.endswith("```"):
            clean_str = clean_str[:-3]
        
        parsed = json.loads(clean_str.strip())
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

        clean_str = raw_report.strip()
        if clean_str.startswith("```json"):
            clean_str = clean_str[7:]
        if clean_str.endswith("```"):
            clean_str = clean_str[:-3]

        clean_str = clean_str.strip()
        try:
            parsed = json.loads(clean_str)
        except json.JSONDecodeError as jde:
            # Try to extract content inside the outermost { } braces
            start_idx = clean_str.find('{')
            end_idx = clean_str.rfind('}')
            if start_idx != -1 and end_idx != -1:
                try:
                    parsed = json.loads(clean_str[start_idx:end_idx+1])
                except Exception:
                    logger.error(f"Fallback brace parsing failed: {jde}. Raw report was: {raw_report}")
                    raise jde
            else:
                logger.error(f"JSON brace boundaries not found: {jde}. Raw report was: {raw_report}")
                raise jde

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
    try:
        raw_code_score = float(code_eval.get("score", 0))
    except Exception as e:
        logger.error(f"Error reading raw code score: {e}")
        raw_code_score = 0

    correctness_score = _extract_metric_score(code_eval, "correctness")
    edge_score = _extract_metric_score(code_eval, "edge_cases")
    req_score = _extract_metric_score(code_eval, "requirements")
    qual_score = _extract_metric_score(code_eval, "code_quality")
    perf_score = _extract_metric_score(code_eval, "performance")

    derived_component_total = correctness_score + edge_score + req_score + qual_score + perf_score
    if raw_code_score <= 0 and derived_component_total > 0:
        raw_code_score = derived_component_total

    if visual_eval:
        visual_score = float(visual_eval.get("visual_score", 100))
        aggregated = (raw_code_score * 0.8) + (visual_score * 0.2)
    else:
        visual_score = 0.0
        aggregated = raw_code_score

    final_score = int(round(aggregated))

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

    # Merge feedback parameters
    strengths = code_eval.get("strengths", [])
    weaknesses = code_eval.get("weaknesses", [])
    improvements = code_eval.get("improvements", [])
    
    if visual_eval:
        strengths.extend(visual_eval.get("strengths", []))
        weaknesses.extend(visual_eval.get("weaknesses", []))

    feedback = code_eval.get("feedback", "Excellent effort. Recheck weaknesses and improvements checklist.")

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
