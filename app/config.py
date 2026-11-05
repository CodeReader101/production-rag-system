import os
from pydantic_settings import BaseSettings
from pydantic import SecretStr
from functools import lru_cache

class Settings(BaseSettings):

    # LLM configuration
    openrouter_api_key: SecretStr = ""
    primary_model: str = "nvidia/nemotron-3-ultra-550b-a55b:free" 
    fallback_model: str = "recursive/flint-v1.0:free"

    # Langsmith
    langsmith_tracing: bool = True
    langsmith_api_key: str = "" 
    langsmith_project: str = "production-rag"

    # Application
    app_env: str = "devleopment"
    log_level: str = "INFO"
    rate_limit: str = "20/minute"
    cache_ttl_seconds: int = 300
    max_retries: int = 3

    model_config = {"env_file": ".env", "extra": "ignore"}  

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


def _export_langsmith_env(settings: "Settings") -> None:
    if settings.langsmith_api_key:
        os.environ["LANGSMITH_TRACING"] = str(settings.langsmith_tracing).lower()
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    _export_langsmith_env(settings)
    return settings