from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional

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

    # Security: shared secret required on all /internal/* routes. If left unset, internal
    # auth is disabled (backwards compatible) but a startup warning is logged -- this MUST
    # be set in any environment reachable from the public internet.
    internal_api_key: Optional[str] = Field(None, validation_alias="INTERNAL_API_KEY")

    # Comma separated list of allowed CORS origins, or "*" for all (development only).
    cors_allowed_origins: str = Field("*", validation_alias="CORS_ALLOWED_ORIGINS")

    # Fault-injection testing hooks (used by app/tests/test_router_simulation.py) that let a
    # chat message like "simulate 429" force the model router down a specific failure path.
    # MUST stay disabled in production -- it must never be reachable via real user input.
    enable_router_fault_simulation: bool = Field(False, validation_alias="ENABLE_ROUTER_FAULT_SIMULATION")

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

    @property
    def cors_origins_list(self) -> list:
        """
        Parse the comma-separated CORS_ALLOWED_ORIGINS setting into a list.
        Returns ["*"] if unset/blank (development default).
        """
        raw = (self.cors_allowed_origins or "*").strip()
        if raw == "*":
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]

settings = Settings()
