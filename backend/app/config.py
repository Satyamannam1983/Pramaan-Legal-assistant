from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    app_name: str = "Pramaan API"
    api_prefix: str = "/api/v1"

    # LLM
    openai_api_key: str | None = None
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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


settings = Settings()
