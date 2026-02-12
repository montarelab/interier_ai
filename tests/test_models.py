from unittest.mock import MagicMock

from src.models import UsageMetadata


def test_usage_metadata_from_openai_usage():
    """Test UsageMetadata.from_openai_usage() factory method."""
    openai_usage = MagicMock()
    openai_usage.input_tokens = 100
    openai_usage.output_tokens = 50
    openai_usage.total_tokens = 150

    result = UsageMetadata.from_openai_usage(model="gpt-4", usage=openai_usage)

    assert result.model == "gpt-4"
    assert result.input_tokens == 100
    assert result.output_tokens == 50
    assert result.total_tokens == 150
    assert isinstance(result.total_cost_usd, float)


def test_usage_metadata_from_google_usage():
    """Test UsageMetadata.from_google_usage() factory method."""
    google_usage = MagicMock()
    google_usage.prompt_token_count = 200
    google_usage.candidates_token_count = 100
    google_usage.total_token_count = 300

    result = UsageMetadata.from_google_usage(
        model="gemini-2.5-flash-image", usage=google_usage
    )

    assert result.model == "gemini-2.5-flash-image"
    assert result.input_tokens == 200
    assert result.output_tokens == 100
    assert result.total_tokens == 300
    assert isinstance(result.total_cost_usd, float)


def test_usage_metadata_direct_initialization():
    """Test UsageMetadata direct initialization."""
    usage = UsageMetadata(
        model="test-model",
        input_tokens=50,
        output_tokens=25,
        total_tokens=75,
    )

    assert usage.model == "test-model"
    assert usage.input_tokens == 50
    assert usage.output_tokens == 25
    assert usage.total_tokens == 75
    assert isinstance(usage.total_cost_usd, float)
