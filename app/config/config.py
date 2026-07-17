from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    gemini_api_key: str = Field(..., validation_alias="GEMINI_API_KEY")
    database_url: str = Field(..., validation_alias="DATABASE_URL")
    supabase_url: str = Field(..., validation_alias="SUPABASE_URL")
    supabase_service_role_key: str = Field(..., validation_alias="SUPABASE_SERVICE_ROLE_KEY")
    playwright_headless: bool = Field(True, validation_alias="PLAYWRIGHT_HEADLESS")
    gemini_models_fallback_chain: str = Field(
        "models/gemini-2.5-pro,models/gemini-2.5-flash,models/gemini-2.0-flash,models/gemini-1.5-pro",
        validation_alias="GEMINI_MODELS_FALLBACK_CHAIN"
    )
    model_cooldown_seconds: int = Field(60, validation_alias="MODEL_COOLDOWN_SECONDS")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def async_database_url(self) -> str:
        """
        Convert standard postgresql:// URL to postgresql+asyncpg:// for async pgvector driver
        """
        url = self.database_url
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        if url.startswith("postgresql://") and not url.startswith("postgresql+asyncpg://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return url

settings = Settings()
