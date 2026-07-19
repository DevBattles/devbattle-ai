from app.providers.gemini import GeminiProvider
from app.embeddings.vector_client import VectorClient
from app.prompts.solution_prompts import SOLUTION_GENERATOR_PROMPT
from app.utils.logger import logger
from sqlalchemy import text
import uuid
import json

class SolutionGeneratorService:
    def __init__(self, provider: GeminiProvider, vector_client: VectorClient):
        self.provider = provider
        self.vector_client = vector_client

    async def generate_solutions_and_rubric(self, question_data: dict) -> dict:
        """
        Generate 10-15 semantic variants of valid solution code, generate embeddings,
        save to database vector table, and return the rubric configuration.
        """
        title = question_data.get("title", "")
        description = question_data.get("description", "")
        requirements = json.dumps(question_data.get("requirements", []))
        starter_files = json.dumps(question_data.get("starterFiles", {}))
        expected_output = question_data.get("expectedOutput", "")
        question_id = question_data.get("questionId")
        version = question_data.get("version", 1)

        prompt = SOLUTION_GENERATOR_PROMPT.format(
            title=title,
            description=description,
            requirements=requirements,
            starter_files=starter_files,
            expected_output=expected_output
        )

        logger.info(f"Requesting Gemini to generate solutions and rubric for question {question_id} v{version}")
        raw_response = await self.provider.generate_text(
            prompt=prompt,
            system_instruction="You are a Principal AI backend grading assistant. Always return valid JSON matching schemas exactly.",
            json_mode=True
        )

        try:
            clean_str = raw_response.strip()
            if clean_str.startswith("```json"):
                clean_str = clean_str[7:]
            if clean_str.endswith("```"):
                clean_str = clean_str[:-3]
            clean_str = clean_str.strip()

            parsed = json.loads(clean_str)
        except Exception as e:
            logger.error(f"Failed to parse generated solution JSON: {e}. Raw response: {raw_response}")
            raise ValueError("LLM response did not contain valid JSON")

        generated_solutions = parsed.get("solutions", [])
        rubric = parsed.get("rubric", {})
        category = parsed.get("category")
        workspace_type = parsed.get("workspaceType")
        evaluation_strategy = parsed.get("evaluationStrategy")
        supported_language = parsed.get("supportedLanguage")
        preview_required = parsed.get("previewRequired", False)
        execution_mode = parsed.get("executionMode")
        options = parsed.get("options")
        starter_files = parsed.get("starterFiles", {})

        solutions_to_save = []
        for sol in generated_solutions:
            code = sol.get("code", "")
            sol_type = sol.get("type", "unknown")
            if code:
                embedding = await self.provider.get_embedding(code)
                solutions_to_save.append({
                    "code": code,
                    "type": sol_type,
                    "embedding": embedding
                })

        if solutions_to_save:
            await self.vector_client.save_solutions(question_id, version, solutions_to_save)

        # Synchronize generated rubric and metadata configuration directly with the question_versions table in Supabase
        try:
            async with self.vector_client.async_session() as session:
                await session.execute(
                    text("""
                        UPDATE question_versions 
                        SET rubric = :rubric,
                            category = :category,
                            workspace_type = :workspace_type,
                            evaluation_strategy = :evaluation_strategy,
                            supported_language = :supported_language,
                            preview_required = :preview_required,
                            execution_mode = :execution_mode,
                            options = :options,
                            starter_files = :starter_files
                        WHERE question_id = :qid AND version = :ver
                    """),
                    {
                        "rubric": json.dumps(rubric),
                        "category": category,
                        "workspace_type": workspace_type,
                        "evaluation_strategy": evaluation_strategy,
                        "supported_language": supported_language,
                        "preview_required": preview_required,
                        "execution_mode": execution_mode,
                        "options": json.dumps(options) if options is not None else None,
                        "starter_files": json.dumps(starter_files),
                        "qid": uuid.UUID(question_id),
                        "ver": version
                    }
                )
                await session.commit()
            logger.info(f"Synchronized rubric and metadata to question_versions table for {question_id} v{version}")
        except Exception as db_err:
            logger.warning(f"Could not persist rubric and metadata to database question_versions record: {db_err}")

        return {
            "questionId": question_id,
            "version": version,
            "solutions_count": len(solutions_to_save),
            "rubric": rubric,
            "category": category,
            "workspaceType": workspace_type,
            "evaluationStrategy": evaluation_strategy,
            "supportedLanguage": supported_language,
            "previewRequired": preview_required,
            "executionMode": execution_mode,
            "options": options,
            "starterFiles": starter_files,
            "solutions": solutions_to_save
        }
