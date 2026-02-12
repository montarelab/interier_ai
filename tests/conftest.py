import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    os.environ.setdefault("GOOGLE_API_KEY", "test-key")
    os.environ.setdefault("HOST", "0.0.0.0")
    os.environ.setdefault("PORT", "8000")
    os.environ.setdefault("IMG_GEN_MODEL", "gpt-image-1-mini")
    os.environ.setdefault("PLAN_MODEL", "gpt-4.5-preview-2025-01-21")
    os.environ.setdefault("EVAL_MODEL", "gpt-4.5-preview-2025-01-21")


@pytest.fixture
def temp_job_path(tmp_path: Path):
    """Create a temporary job path for testing."""
    job_path = tmp_path / "test_job"
    job_path.mkdir(parents=True, exist_ok=True)
    return job_path
