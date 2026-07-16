from abc import ABC, abstractmethod
from typing import List, Optional

class BaseAIProvider(ABC):
    @abstractmethod
    async def generate_text(self, prompt: str, system_instruction: Optional[str] = None, json_mode: bool = False) -> str:
        """
        Generate a text response from the LLM.
        """
        pass

    @abstractmethod
    async def generate_multimodal(self, prompt: str, image_bytes: bytes, mime_type: str = "image/png") -> str:
        """
        Analyze text along with visual frames (Gemini Vision capabilities).
        """
        pass

    @abstractmethod
    async def get_embedding(self, text: str) -> List[float]:
        """
        Generate text embeddings (768 dimensions by default).
        """
        pass
