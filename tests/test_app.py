from fastapi.testclient import TestClient
from starlette.middleware.cors import CORSMiddleware

from src.main import app


def test_fastapi_app_initializes():
    """Test that FastAPI app can initialize."""
    assert app is not None
    assert app.title == "FastAPI"


def test_routes_are_registered():
    """Test that /generate and /evaluate routes are registered."""
    routes = [route.path for route in app.routes]

    assert "/generate" in routes
    assert "/evaluate" in routes


def test_cors_middleware_configured():
    """Test that CORS middleware is configured."""
    assert len(app.user_middleware) > 0

    middleware_classes = [m.cls for m in app.user_middleware]
    assert CORSMiddleware in middleware_classes


def test_app_health_check():
    """Basic smoke test - app can handle requests."""
    client = TestClient(app)
    response = client.get("/")

    assert response.status_code in [404, 200, 405]
