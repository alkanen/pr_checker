import hashlib
import hmac
from collections.abc import AsyncGenerator

import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from pr_checker.config import ServerConfig
from pr_checker.main import create_app


@pytest.fixture
def test_config() -> ServerConfig:
    return ServerConfig(
        github_token="test-token",  # noqa: S106
        github_webhook_secret="test-secret",  # noqa: S106
        database_url="sqlite+aiosqlite:///:memory:",
    )


@pytest.fixture
async def managed_app(test_config: ServerConfig) -> AsyncGenerator[FastAPI, None]:
    the_app = create_app(test_config)
    async with LifespanManager(the_app):
        yield the_app


@pytest.fixture
async def client(managed_app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    async with AsyncClient(transport=ASGITransport(app=managed_app), base_url="http://test") as c:
        yield c


def make_signature(payload: bytes, secret: str) -> str:
    return "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
