import os
from pydantic_settings import BaseSettings
from pydantic import SecretStr
from functools import lru_cache

class Settings(BaseSettings):

    # LLM configuration
    openrouter_api_key: SecretStr
    primary_model: str = "nvidia/nemotron-3-ultra-550b-a55b:free" 
    judge_model: str = "baidu/ernie-5.5-flash-preview:free"
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

    # RAG
    rag_collection: str = "documents"
    rag_table: str = "rag_documents"
    rag_top_k: int = 4
    rag_min_score: float = 0.35
    rag_hybrid: bool = True
    embedding_model: str = "openai/text-embedding-3-small"
    embedding_dim: int = 1536
    qdrant_url: str = ""
    admin_api_key: str

    database_url: str
    redis_url: str = "redis://localhost:6379/0"

    model_config = {"env_file": ".env", "extra": "ignore"}  

    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # empty string disables reranking
    rag_candidates: int = 20
    rag_min_rerank_score: float = -1e9               # disabled until tuned from eval data

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


def _export_langsmith_env(settings: "Settings") -> None:
    if settings.langsmith_api_key:
        os.environ["LANGSMITH_TRACING"] = str(settings.langsmith_tracing).lower()
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        os.environ["JUDGE_MODEL"] = settings.judge_model

@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    _export_langsmith_env(settings)
    return settings