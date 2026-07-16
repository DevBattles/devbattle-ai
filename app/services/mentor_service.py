from app.providers.gemini import GeminiProvider
from app.prompts.mentor_prompts import MENTOR_SYSTEM_INSTRUCTION
from app.utils.logger import logger
from datetime import datetime, timezone
import json

class MentorService:
    def __init__(self, provider: GeminiProvider):
        self.provider = provider

    async def get_mentor_response(self, chat_data: dict) -> str:
        """
        Evaluate time deadlines and active contests to determine AI tutor hint levels,
        and generate a context-aware chat helper output.
        """
        role = chat_data.get("currentRole", "student")
        question = chat_data.get("currentQuestion", {})
        context = chat_data.get("currentContext", {})
        chat_history = chat_data.get("chatHistory", [])
        user_message = chat_data.get("message", "")

        q_title = question.get("title", "Generic Question")
        q_desc = question.get("description", "Solve the challenge.")
        q_out = question.get("expectedOutput", "")

        # Extract context settings
        context_type = context.get("type", "practice")  # contest, homework, practice
        is_active = context.get("isActive", False)
        deadline_str = context.get("deadline")
        
        constraint_level = "practice"
        if context_type == "contest" and is_active:
            constraint_level = "contest_active"
        elif context_type == "homework":
            if deadline_str:
                try:
                    # Clean UTC timestamp
                    cleaned_ts = deadline_str.replace("Z", "+00:00")
                    deadline = datetime.fromisoformat(cleaned_ts)
                    # Use timezone-aware comparison
                    if datetime.now(timezone.utc) < deadline:
                        constraint_level = "homework_before_deadline"
                    else:
                        constraint_level = "homework_after_deadline"
                except Exception as e:
                    logger.warning(f"Could not parse deadline timestamp: {deadline_str}. Defaulting to homework_before_deadline: {e}")
                    constraint_level = "homework_before_deadline"
            else:
                constraint_level = "homework_before_deadline"

        system_instruction = MENTOR_SYSTEM_INSTRUCTION.format(
            role=role,
            question_title=q_title,
            question_description=q_desc,
            expected_output=q_out,
            context_mode=context_type,
            constraint_level=constraint_level
        )

        logger.info(f"AI Mentor request processing. Constraint mode set to: {constraint_level}")

        # Stitch prompt sequences
        prompt = ""
        for msg in chat_history:
            msg_role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt += f"{msg_role}: {content}\n"
        prompt += f"user: {user_message}\nmodel:"

        response_text = await self.provider.generate_text(
            prompt=prompt,
            system_instruction=system_instruction
        )
        return response_text
