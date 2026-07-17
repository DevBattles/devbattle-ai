import os
import asyncio
from google import genai
from google.genai import types
from app.providers.base import BaseAIProvider
from app.config.config import settings
from app.utils.logger import logger
from typing import List, Optional
from app.services.model_router import generate_content_with_router

class GeminiProvider(BaseAIProvider):
    def __init__(self):
        logger.info("Initializing Google GenAI client...")
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self._text_model = "models/gemini-2.5-flash"
        self._chat_model = "models/gemini-2.5-pro"
        self._embed_model = "models/gemini-embedding-001"
        self._initialized = False
        self._lock = asyncio.Lock()

    async def _detect_models(self):
        if self._initialized:
            return
        async with self._lock:
            if self._initialized:
                return
            try:
                loop = asyncio.get_running_loop()
                models_list = await loop.run_in_executor(
                    None,
                    lambda: list(self.client.models.list())
                )
                supported = [m.name for m in models_list]
                logger.info(f"Supported Gemini models: {supported}")

                # Detect Text/Vision/JSON model
                for opt in ["models/gemini-2.5-flash", "models/gemini-3.5-flash", "models/gemini-2.0-flash", "models/gemini-1.5-flash"]:
                    if opt in supported or opt.replace("models/", "") in supported:
                        self._text_model = opt
                        break

                # Detect Chat / Strongest Reasoning model
                self._chat_model = "models/gemini-2.5-pro"
                for opt in ["models/gemini-2.5-pro", "models/gemini-2.0-pro-exp-02-05", "models/gemini-1.5-pro", "models/gemini-2.5-flash"]:
                    if opt in supported or opt.replace("models/", "") in supported:
                        self._chat_model = opt
                        break

                # Detect Embedding model
                for opt in ["models/gemini-embedding-001", "models/text-embedding-004", "models/gemini-embedding-2"]:
                    if opt in supported or opt.replace("models/", "") in supported:
                        self._embed_model = opt
                        break

                # Override from env variables if configured
                text_model_env = os.environ.get("TEXT_MODEL")
                if text_model_env:
                    self._text_model = text_model_env
                chat_model_env = os.environ.get("CHAT_MODEL")
                if chat_model_env:
                    self._chat_model = chat_model_env

                logger.info(f"Auto-detected models: text = {self._text_model}, chat = {self._chat_model}, embedding = {self._embed_model}")
                self._initialized = True
            except Exception as e:
                logger.error(f"Error detecting supported models: {e}. Falling back to defaults.")
                self._initialized = True # prevent endless retries

    async def generate_text(self, prompt: str, system_instruction: Optional[str] = None, json_mode: bool = False, model: Optional[str] = None) -> str:
        await self._detect_models()
        return await generate_content_with_router(
            client=self.client,
            prompt=prompt,
            system_instruction=system_instruction,
            json_mode=json_mode,
            model=model
        )

    async def generate_multimodal(self, prompt: str, image_bytes: bytes, mime_type: str = "image/png", model: Optional[str] = None) -> str:
        await self._detect_models()
        target_model = model or self._text_model
        logger.debug(f"Calling GenAI generate_multimodal on model: {target_model}")

        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type=mime_type
        )

        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.models.generate_content(
                model=target_model,
                contents=[prompt, image_part]
            )
        )
        return response.text or ""

    async def get_embedding(self, text: str) -> List[float]:
        await self._detect_models()
        logger.debug(f"Generating embedding via GenAI model: {self._embed_model}")

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.client.models.embed_content(
                model=self._embed_model,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=768
                )
            )
        )
        try:
            if hasattr(result, "embeddings") and result.embeddings:
                return result.embeddings[0].values
            if hasattr(result, "embedding") and result.embedding:
                return result.embedding.values
            if isinstance(result, dict):
                return result.get("embedding", {}).get("values") or result.get("embeddings", [{}])[0].get("values")
        except Exception as e:
            logger.error(f"Error extracting embedding values: {e}")
            raise e
        return []
