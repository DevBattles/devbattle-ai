from app.providers.gemini import GeminiProvider
from app.embeddings.vector_client import VectorClient
from app.prompts.solution_prompts import SOLUTION_GENERATOR_PROMPT
from app.utils.logger import logger
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

        return {
            "questionId": question_id,
            "version": version,
            "solutions_count": len(solutions_to_save),
            "rubric": rubric
        }
