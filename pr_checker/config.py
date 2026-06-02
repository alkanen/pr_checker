from pydantic_settings import BaseSettings


class ServerConfig(BaseSettings):
    github_token: str = ""
    github_webhook_secret: str = ""
    database_url: str = "sqlite+aiosqlite:///./pr_checker.db"
    port: int = 8000
    root_path: str = ""
    forwarded_allow_ips: str = "127.0.0.1"
    config_file: str | None = None
    lm_studio_url: str = ""
    lm_studio_api_key: str = ""
    openai_api_key: str = ""
    openai_base_url: str = ""
    llm_model: str = ""
    llm_timeout: float | None = 600.0
    log_level: str = "INFO"
    log_format: str = "text"
    qdrant_url: str = ""
    embedding_model: str = "text-embedding-nomic-ai-nomic-embed-text-v2-moe"
    embedding_max_tokens: int = 512
    embedding_max_concurrent_files: int = 5
    max_prefetch_chunks: int = 16
    max_search_chunks_per_call: int = 5
    llm_debug_dir: str = ""
