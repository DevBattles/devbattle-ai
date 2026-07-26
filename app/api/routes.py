from fastapi import APIRouter, HTTPException, Header, Depends
from app.schemas.schemas import (
    GenerateQuestionRequest,
    EvaluateSubmissionRequest,
    MentorChatRequest
)
from app.providers.gemini import GeminiProvider
from app.embeddings.vector_client import VectorClient
from app.services.solution_generator import SolutionGeneratorService
from app.services.mentor_service import MentorService
from app.graph.workflow import app_workflow
from app.utils.logger import logger
from app.config.config import settings
from typing import Dict, Any, Optional

router = APIRouter()


def verify_internal_api_key(x_internal_api_key: Optional[str] = Header(default=None)):
    """
    Guards all /internal/* routes with a shared-secret header. If INTERNAL_API_KEY is not
    configured in the environment, auth is skipped (with a startup warning already logged)
    to preserve local-dev/backwards compatibility -- but any internet-facing deployment
    MUST set INTERNAL_API_KEY.
    """
    if not settings.internal_api_key:
        return
    if not x_internal_api_key or x_internal_api_key != settings.internal_api_key:
        raise HTTPException(status_code=401, detail="Missing or invalid X-Internal-Api-Key header")


@router.post("/internal/router/reset", dependencies=[Depends(verify_internal_api_key)])
def reset_router_health():
    from app.services.model_router import reset_health_registry
    reset_health_registry()
    return {"status": "reset"}

# Singletons shared from LangGraph nodes module context
from app.graph.nodes import provider, vector_client

solution_service = SolutionGeneratorService(provider, vector_client)
mentor_service = MentorService(provider)

@router.get("/health")
def health_check():
    from app.services.model_router import get_health_status_report
    from app.services.mentor_service import HEALTH_STATUS as mentor_health_status
    return {
        "status": "ok",
        "service": "devbattle-ai-backend",
        "health_details": get_health_status_report(),
        "mentor_chat_health": mentor_health_status
    }

@router.post("/internal/questions/generate", dependencies=[Depends(verify_internal_api_key)])
async def generate_solutions(payload: GenerateQuestionRequest):
    try:
        logger.info(f"Received question generation request for {payload.questionId}")
        result = await solution_service.generate_solutions_and_rubric(payload.model_dump())
        return {"success": True, "message": "Solutions and rubric generated", "data": result}
    except Exception as e:
        logger.error(f"Generate solutions endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/internal/submissions/evaluate", dependencies=[Depends(verify_internal_api_key)])
async def evaluate_submission(payload: EvaluateSubmissionRequest):
    try:
        logger.info(f"Received submission evaluation request for {payload.questionId} v{payload.version}")
        input_state = {
            "question_id": payload.questionId,
            "version": payload.version,
            "student_files": payload.studentFiles,
            "github_url": payload.githubUrl,
            "error": None
        }

        # Invoke the compiled LangGraph pipeline
        output_state = await app_workflow.ainvoke(input_state)

        # Catch workflow errors
        if output_state.get("error"):
            logger.error(f"LangGraph execution error: {output_state['error']}")
            raise HTTPException(status_code=400, detail=output_state["error"])

        # Format return data structure
        response_data = {
            "score": output_state.get("score", 0),
            "grade": output_state.get("grade", "F"),
            "feedback": {
                "strengths": output_state.get("strengths", []),
                "weaknesses": output_state.get("weaknesses", []),
                "improvements": output_state.get("improvements", []),
                "generalFeedback": output_state.get("feedback", ""),
                "rubricScores": output_state.get("rubric_scores", {})
            }
        }
        return {"success": True, "message": "Evaluation completed successfully", "data": response_data}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Evaluate submission endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/internal/mentor/chat", dependencies=[Depends(verify_internal_api_key)])
async def mentor_chat(payload: MentorChatRequest):
    try:
        logger.info("Received AI Mentor chat request")
        response = await mentor_service.get_mentor_response(payload.model_dump())
        return {"success": True, "message": "Mentor query responded", "data": {"response": response}}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Mentor chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/ai/mentor/chat")
async def mentor_chat_public(payload: MentorChatRequest):
    return await mentor_chat(payload)
