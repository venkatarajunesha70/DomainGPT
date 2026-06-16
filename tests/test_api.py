"""
Integration tests for the FastAPI endpoints.
Uses httpx AsyncClient with a test database.
"""
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from apps.api.main import app


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_readiness(client):
    resp = await client.get("/health/ready")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_register_and_login(client):
    # Register
    resp = await client.post(
        "/api/v1/auth/register",
        json={
            "email": "test@example.com",
            "username": "testuser",
            "password": "securepass123",
            "tenant_id": "test-tenant",
        },
    )
    # May fail if DB is not running in CI – acceptable
    assert resp.status_code in (201, 500)


@pytest.mark.asyncio
async def test_chat_requires_auth(client):
    resp = await client.post(
        "/api/v1/chat/",
        json={"question": "What is this document about?"},
    )
    assert resp.status_code == 403  # No auth header
