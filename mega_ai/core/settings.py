"""Application configuration loaded exclusively from environment variables."""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = Field(validation_alias="DATABASE_URL")
    database_sync_url: str = Field(validation_alias="DATABASE_SYNC_URL")
    redis_url: str = Field(validation_alias="REDIS_URL")

    litellm_model: str = Field(validation_alias="LITELLM_MODEL", default="gpt-4o")
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    ollama_api_base: str | None = Field(default=None, validation_alias="OLLAMA_API_BASE")
    llm_mode: str = Field(validation_alias="LLM_MODE", default="mock")

    api_host: str = Field(default="0.0.0.0", validation_alias="API_HOST")
    api_port: int = Field(default=8000, validation_alias="API_PORT")

    dramatiq_processes: int = Field(default=2, validation_alias="DRAMATIQ_PROCESSES")
    dramatiq_threads: int = Field(default=8, validation_alias="DRAMATIQ_THREADS")

    log_viewer_port: int = Field(default=8080, validation_alias="LOG_VIEWER_PORT")

    sandbox_timeout_seconds: int = Field(default=5, validation_alias="SANDBOX_TIMEOUT_SECONDS")
    rag_embedding_model: str = Field(
        default="text-embedding-3-small",
        validation_alias="RAG_EMBEDDING_MODEL",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
