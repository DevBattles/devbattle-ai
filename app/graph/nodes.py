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
You are a Principal AI backend grading assistant. Evaluate the following student submission:
Files: {student_files}

Compare it against:
1. Question metadata:
Title: {title}
Description: {description}
Requirements: {requirements}

2. Rubric details:
{rubric}

3. Similar reference solutions:
{similar_solutions}

Provide a detailed evaluation of:
1. correctness (up to {correctness_max} points)
2. responsiveness (up to {responsiveness_max} points)
3. accessibility (up to {accessibility_max} points)
4. performance (up to {performance_max} points)
5. code_quality (up to {code_quality_max} points)

GRADING RULES:
- If the files are empty, blank, or have no real code content matching the assignment, all scores must be exactly 0.
- If the student's code does not implement the core functional requirements or expected output logic described in the metadata/rubric, the correctness score must be 0 or extremely low. Do not give any default/pity points.
- If there is unescaped double quotes or syntax errors in the student's solution, penalize the code_quality and correctness score heavily.
- For all feedback strings, ensure that any double quotes inside text values are properly escaped (e.g., use \\\" or single quotes instead) to keep the JSON payload valid.

Format response as JSON:
{{
  "correctness": {{ "score": 0, "feedback": "Detailed explanation of why correctness requirements were met or failed" }},
  "responsiveness": {{ "score": 0, "feedback": "Detailed design layout feedback" }},
  "accessibility": {{ "score": 0, "feedback": "Accessibility / Semantic HTML assessment" }},
  "performance": {{ "score": 0, "feedback": "Performance & optimization feedback" }},
  "code_quality": {{ "score": 0, "feedback": "Code architecture evaluation" }},
  "strengths": ["Strength detail 1"],
  "weaknesses": ["Failure detail 1"],
  "improvements": ["Suggested improvement 1"],
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
                    SELECT title, description, starter_files, expected_output 
                    FROM question_versions 
                    WHERE question_id = :qid AND version = :ver
                """),
                {"qid": uuid.UUID(qid), "ver": ver}
            )
            row = result.fetchone()
            
            if not row:
                # Fallback to question_bank main record if version entry not found
                fallback_result = await session.execute(
                    text("SELECT title, description, expected_output FROM question_bank WHERE id = :qid"),
                    {"qid": uuid.UUID(qid)}
                )
                fb_row = fallback_result.fetchone()
                if not fb_row:
                    return {"error": f"Question {qid} not found in database"}
                
                meta = {
                    "title": fb_row[0],
                    "description": fb_row[1],
                    "starter_files": {},
                    "expected_output": fb_row[2] or ""
                }
            else:
                meta = {
                    "title": row[0],
                    "description": row[1],
                    "starter_files": row[2] or {},
                    "expected_output": row[3] or ""
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
    correctness_max = rubric.get("correctness", {}).get("max_points", 30)
    responsiveness_max = rubric.get("responsiveness", {}).get("max_points", 20)
    accessibility_max = rubric.get("accessibility", {}).get("max_points", 15)
    performance_max = rubric.get("performance", {}).get("max_points", 15)
    code_quality_max = rubric.get("code_quality", {}).get("max_points", 20)

    prompt = CODE_EVALUATE_PROMPT.format(
        student_files=json.dumps(files),
        title=meta.get("title", ""),
        description=meta.get("description", ""),
        requirements=meta.get("requirements", ""),
        rubric=json.dumps(rubric),
        similar_solutions=json.dumps(solutions),
        correctness_max=correctness_max,
        responsiveness_max=responsiveness_max,
        accessibility_max=accessibility_max,
        performance_max=performance_max,
        code_quality_max=code_quality_max
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
        return {"error": f"LLM code evaluation failure: {str(e)}"}

async def aggregate_scores_node(state: SubmissionState) -> dict:
    logger.info("Executing aggregate_scores node...")
    code_eval = state.get("code_evaluation")
    visual_eval = state.get("visual_evaluation")

    if not code_eval:
        return {"error": "Score aggregation failed due to missing code evaluation data"}

    # Extract score keys
    try:
        corr_score = float(code_eval.get("correctness", {}).get("score", 0))
        resp_score = float(code_eval.get("responsiveness", {}).get("score", 0))
        acc_score = float(code_eval.get("accessibility", {}).get("score", 0))
        perf_score = float(code_eval.get("performance", {}).get("score", 0))
        qual_score = float(code_eval.get("code_quality", {}).get("score", 0))
        
        raw_code_score = corr_score + resp_score + acc_score + perf_score + qual_score
    except Exception as e:
        logger.error(f"Error reading subscores: {e}")
        raw_code_score = 70

    visual_score = float(visual_eval.get("visual_score", 80) if visual_eval else 80)

    # 80% code score + 20% visual vision layout rendering score
    aggregated = (raw_code_score * 0.8) + (visual_score * 0.2)
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
        "correctness": corr_score,
        "responsiveness": resp_score,
        "accessibility": acc_score,
        "performance": perf_score,
        "code_quality": qual_score,
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
