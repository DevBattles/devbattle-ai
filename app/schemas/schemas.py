from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

class GenerateQuestionRequest(BaseModel):
    questionId: str = Field(..., description="Unique question UUID")
    version: int = Field(1, description="Question version index")
    title: str = Field(..., description="Title of the challenge")
    description: str = Field(..., description="Details and instructions")
    requirements: List[str] = Field(default=[], description="Strict requirements list")
    starterFiles: Dict[str, Any] = Field(default={}, description="Files provided to student")
    expectedOutput: str = Field(..., description="Expected visual structure result")

class EvaluateSubmissionRequest(BaseModel):
    questionId: str = Field(..., description="Question UUID")
    version: int = Field(1, description="Question version number")
    studentFiles: Dict[str, Any] = Field(..., description="Files submitted by student")
    githubUrl: Optional[str] = Field(None, description="Student code repository path")

class MentorChatRequest(BaseModel):
    currentRole: str = Field("student", description="Role of the user")
    currentQuestion: Dict[str, Any] = Field(..., description="Target question metadata context")
    currentContext: Dict[str, Any] = Field(..., description="Timeline and event context (active contest, homework, deadline)")
    chatHistory: List[Dict[str, Any]] = Field(default=[], description="Prior chat logs list")
    message: str = Field(..., description="User prompt text")
