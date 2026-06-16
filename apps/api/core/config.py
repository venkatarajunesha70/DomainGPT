"""
Central application configuration using Pydantic Settings.
All values come from environment variables or .env file.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────────────────
    app_env: str = "development"
    app_secret_key: str = "change-me"
    debug: bool = True

    # ── JWT ──────────────────────────────────────────────────────────────
    jwt_secret_key: str = "change-me-jwt"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 60

    # ── PostgreSQL ────────────────────────────────────────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "domaingpt"
    postgres_user: str = "domaingpt"
    postgres_password: str = "domaingpt"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def sync_database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # ── Redis ─────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── Pinecone ──────────────────────────────────────────────────────────
    pinecone_api_key: str = ""
    pinecone_env: str = "us-east-1"
    pinecone_index_name: str = "domaingpt"

    # ── LLM providers ────────────────────────────────────────────────────
    groq_api_key: str = ""
    groq_model: str = "llama3-70b-8192"
    openai_api_key: str = ""

    # ── HuggingFace / local model ─────────────────────────────────────────
    hf_token: str = ""
    hf_base_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"

    # ── AWS / S3 ──────────────────────────────────────────────────────────
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    s3_bucket_name: str = "domaingpt-documents"

    # ── Celery ────────────────────────────────────────────────────────────
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # ── Embeddings ────────────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_dimension: int = 1024

    # ── LoRA ─────────────────────────────────────────────────────────────
    lora_adapter_path: str = "./models/lora/adapter"

    # ── Rate limiting ─────────────────────────────────────────────────────
    rate_limit_per_minute: int = 60

    # ── Virus scan ────────────────────────────────────────────────────────
    clamav_host: str = "localhost"
    clamav_port: int = 3310


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
