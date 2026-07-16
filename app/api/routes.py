from fastapi import APIRouter, HTTPException
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
from typing import Dict, Any

router = APIRouter()

# Singletons shared from LangGraph nodes module context
from app.graph.nodes import provider, vector_client

solution_service = SolutionGeneratorService(provider, vector_client)
mentor_service = MentorService(provider)

@router.get("/health")
def health_check():
    return {"status": "ok", "service": "devbattle-ai-backend"}

@router.post("/internal/questions/generate")
async def generate_solutions(payload: GenerateQuestionRequest):
    try:
        logger.info(f"Received question generation request for {payload.questionId}")
        result = await solution_service.generate_solutions_and_rubric(payload.model_dump())
        return {"success": True, "message": "Solutions and rubric generated", "data": result}
    except Exception as e:
        logger.error(f"Generate solutions endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/internal/submissions/evaluate")
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

@router.post("/internal/mentor/chat")
async def mentor_chat(payload: MentorChatRequest):
    try:
        logger.info("Received AI Mentor chat request")
        response = await mentor_service.get_mentor_response(payload.model_dump())
        return {"success": True, "message": "Mentor query responded", "data": {"response": response}}
    except Exception as e:
        logger.error(f"Mentor chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
