import math
from urllib.parse import urlsplit

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator, model_validator


class Settings(BaseSettings):
    app_env: str = "development"
    app_name: str = "Pramaan API"
    api_prefix: str = "/api/v1"

    # LLM
    openai_api_key: str | None = None
    openrouter_api_key: str | None = None
    llm_provider: str = "stub"
    llm_model: str = "placeholder"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 2048
    llm_base_url: str | None = None  # For OpenAI-compatible providers

    # Embeddings
    embedding_provider: str = "sentence_transformer"
    embedding_model: str = "intfloat/multilingual-e5-large"
    embedding_dimension: int = 1024
    embedding_batch_size: int = 32

    # Database (MongoDB)
    database_url: str = "mongodb://localhost:27017/pramaan"
    database_name: str = "pramaan"
    atlas_vector_search: bool = False

    # Retrieval — candidate generation sizes
    vector_top_k: int = 20
    keyword_top_k: int = 20
    rerank_top_k: int = 10
    final_top_k: int = 5

    # Hybrid retrieval weights
    vector_weight: float = 0.6
    keyword_weight: float = 0.4

    # RRF constant
    rrf_k: int = 60

    # Reranker
    reranker_provider: str = ""
    reranker_model: str = ""

    # Evidence selection
    max_evidence_chunks: int = 8
    max_context_tokens: int = 6000
    min_rerank_score: float = -10.0  # Very permissive default; let reranker decide
    evidence_min_score: float = -10.0
    max_retrieval_attempts: int = 2
    max_validation_retries: int = 1
    max_upload_size_mb: int = 25
    document_analysis_top_k: int = 8
    document_comparison_top_k: int = 6
    max_comparison_documents: int = 2
    max_query_length: int = 4000
    max_request_body_mb: int = 30
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60
    log_level: str = "INFO"

    # Legacy
    top_k: int = 5

    # File storage
    upload_dir: str = "data/uploads"
    processed_dir: str = "data/processed"

    # Server
    cors_origins: str = "http://localhost:3000"
    chunk_size: int = 1200
    chunk_overlap: int = 150

    @field_validator("app_env", mode="before")
    @classmethod
    def normalize_app_env(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        aliases = {"dev": "development", "prod": "production"}
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"development", "test", "production"}:
            raise ValueError("APP_ENV must be development, test, or production.")
        return normalized

    @field_validator("llm_provider", mode="before")
    @classmethod
    def validate_llm_provider(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized not in {
            "stub", "mock", "openai", "openai-compatible", "openai_compatible",
            "openrouter",
        }:
            raise ValueError(
                "LLM_PROVIDER must be stub, mock, openai, openai-compatible, or openrouter."
            )
        return normalized

    @field_validator("embedding_provider", mode="before")
    @classmethod
    def validate_embedding_provider(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized not in {"stub", "sentence_transformer", "sentence-transformer", "st"}:
            raise ValueError(
                "EMBEDDING_PROVIDER must be stub or sentence_transformer."
            )
        return normalized

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        try:
            parsed.port
        except ValueError as exc:
            raise ValueError("DATABASE_URL must be a valid MongoDB connection URI.") from exc
        if parsed.scheme not in {"mongodb", "mongodb+srv"} or not parsed.hostname:
            raise ValueError("DATABASE_URL must be a valid MongoDB connection URI.")
        return value

    @field_validator("llm_base_url")
    @classmethod
    def validate_llm_base_url(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("LLM_BASE_URL must be an HTTP(S) URL.")
        return value

    @field_validator("cors_origins")
    @classmethod
    def validate_cors_origins(cls, value: str) -> str:
        origins: list[str] = []
        for candidate in value.split(","):
            origin = candidate.strip().rstrip("/")
            if not origin:
                continue
            parsed = urlsplit(origin)
            try:
                parsed.port
            except ValueError as exc:
                raise ValueError("CORS_ORIGINS contains an invalid port.") from exc
            if (
                origin == "*"
                or parsed.scheme not in {"http", "https"}
                or not parsed.netloc
                or parsed.username is not None
                or parsed.password is not None
                or parsed.path
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError(
                    "CORS_ORIGINS must be comma-separated HTTP(S) origins without paths or wildcards."
                )
            origins.append(origin)
        if not origins:
            raise ValueError("CORS_ORIGINS must contain at least one allowed origin.")
        return ",".join(origins)

    @field_validator(
        "embedding_dimension",
        "embedding_batch_size",
        "vector_top_k",
        "keyword_top_k",
        "rerank_top_k",
        "final_top_k",
        "rrf_k",
        "max_evidence_chunks",
        "max_context_tokens",
        "llm_max_tokens",
        "max_retrieval_attempts",
        "max_validation_retries",
        "max_upload_size_mb",
        "max_request_body_mb",
        "rate_limit_requests",
        "rate_limit_window_seconds",
        "chunk_size",
        "document_analysis_top_k",
        "document_comparison_top_k",
        "max_comparison_documents",
        "max_query_length",
        "top_k",
        mode="after",
    )
    @classmethod
    def validate_positive_integer(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Configuration values must be positive integers.")
        return value

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, value: int) -> int:
        if value < 0:
            raise ValueError("CHUNK_OVERLAP cannot be negative.")
        return value

    @field_validator("llm_temperature")
    @classmethod
    def validate_llm_temperature(cls, value: float) -> float:
        if not math.isfinite(value) or not 0 <= value <= 2:
            raise ValueError("LLM_TEMPERATURE must be between 0 and 2.")
        return value

    @field_validator("vector_weight", "keyword_weight")
    @classmethod
    def validate_retrieval_weight(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0:
            raise ValueError("Retrieval weights must be finite and non-negative.")
        return value

    @model_validator(mode="after")
    def validate_related_settings(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
        if self.vector_weight + self.keyword_weight <= 0:
            raise ValueError("At least one hybrid retrieval weight must be positive.")
        if self.max_request_body_mb < self.max_upload_size_mb:
            raise ValueError(
                "MAX_REQUEST_BODY_MB must be at least MAX_UPLOAD_SIZE_MB."
            )
        if self.max_comparison_documents != 2:
            raise ValueError(
                "MAX_COMPARISON_DOCUMENTS must be 2; the API compares pairs."
            )
        if self.llm_provider in {
            "openai", "openai-compatible", "openai_compatible"
        } and not (self.openai_api_key and self.openai_api_key.strip()):
            raise ValueError(
                "OPENAI_API_KEY is required when using an OpenAI-compatible LLM provider."
            )
        if self.llm_provider == "openrouter" and not (
            self.openrouter_api_key and self.openrouter_api_key.strip()
        ):
            raise ValueError(
                "OPENROUTER_API_KEY is required when LLM_PROVIDER=openrouter."
            )
        if self.reranker_provider not in {"", "cross_encoder"}:
            raise ValueError("RERANKER_PROVIDER must be empty or cross_encoder.")
        if bool(self.reranker_provider) != bool(self.reranker_model):
            raise ValueError(
                "Set both RERANKER_PROVIDER and RERANKER_MODEL to enable reranking."
            )
        if not math.isfinite(self.min_rerank_score) or not math.isfinite(self.evidence_min_score):
            raise ValueError("Evidence score thresholds must be finite numbers.")
        if (
            self.app_env == "production"
            and self.cors_origins == "http://localhost:3000"
        ):
            raise ValueError(
                "Set CORS_ORIGINS to the production frontend origin in production."
            )
        return self

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


settings = Settings()
