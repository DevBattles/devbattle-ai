import google.generativeai as genai
from app.providers.base import BaseAIProvider
from app.config.config import settings
from app.utils.logger import logger
from typing import List, Optional
import asyncio

class GeminiProvider(BaseAIProvider):
    def __init__(self):
        logger.info("Initializing Google Gemini provider...")
        genai.configure(api_key=settings.gemini_api_key)

    async def generate_text(self, prompt: str, system_instruction: Optional[str] = None, json_mode: bool = False) -> str:
        """
        Runs text generation via Gemini 1.5 Flash.
        """
        model_name = "gemini-1.5-flash"
        logger.debug(f"Calling Gemini generate_text on model: {model_name}, json_mode: {json_mode}")

        generation_config = {}
        if json_mode:
            generation_config["response_mime_type"] = "application/json"

        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction,
            generation_config=generation_config
        )

        loop = asyncio.get_running_loop()
        # Execute blocking API call in executor threadpool to keep FastAPI async loop responsive
        response = await loop.run_in_executor(
            None,
            lambda: model.generate_content(prompt)
        )
        return response.text

    async def generate_multimodal(self, prompt: str, image_bytes: bytes, mime_type: str = "image/png") -> str:
        """
        Vision evaluation checks layout rendering.
        """
        model_name = "gemini-1.5-flash"
        logger.debug(f"Calling Gemini generate_multimodal on model: {model_name}")

        model = genai.GenerativeModel(model_name=model_name)
        contents = [
            prompt,
            {
                "mime_type": mime_type,
                "data": image_bytes
            }
        ]

        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: model.generate_content(contents)
        )
        return response.text

    async def get_embedding(self, text: str) -> List[float]:
        """
        Generates 768-dimension semantic embeddings vector.
        """
        model_name = "models/text-embedding-004"
        logger.debug(f"Generating embedding via Gemini model: {model_name}")

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: genai.embed_content(
                model=model_name,
                content=text,
                task_type="retrieval_document"
            )
        )
        return result["embedding"]
