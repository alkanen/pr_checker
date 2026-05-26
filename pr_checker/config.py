from pydantic_settings import BaseSettings


class ServerConfig(BaseSettings):
    github_token: str = ""
    github_webhook_secret: str = ""
    database_url: str = "sqlite+aiosqlite:///./pr_checker.db"
    port: int = 8000
    root_path: str = ""
    forwarded_allow_ips: str = "127.0.0.1"
