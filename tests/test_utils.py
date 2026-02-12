from src.models import UsageMetadata
from src.utils import compound_usages


def test_compound_usages_single_model():
    """Test compound_usages with a single model."""
    usages = [
        UsageMetadata(
            model="gpt-4",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        ),
        UsageMetadata(
            model="gpt-4",
            input_tokens=200,
            output_tokens=100,
            total_tokens=300,
        ),
    ]

    result = compound_usages(usages)

    assert len(result) == 1
    assert result[0].model == "gpt-4"
    assert result[0].input_tokens == 300
    assert result[0].output_tokens == 150
    assert result[0].total_tokens == 450


def test_compound_usages_multiple_models():
    """Test compound_usages with multiple models."""
    usages = [
        UsageMetadata(
            model="gpt-4",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        ),
        UsageMetadata(
            model="gemini-pro",
            input_tokens=200,
            output_tokens=100,
            total_tokens=300,
        ),
        UsageMetadata(
            model="gpt-4",
            input_tokens=50,
            output_tokens=25,
            total_tokens=75,
        ),
    ]

    result = compound_usages(usages)

    assert len(result) == 2

    result_sorted = sorted(result, key=lambda x: x.model)

    gemini_usage = next(u for u in result_sorted if u.model == "gemini-pro")
    assert gemini_usage.input_tokens == 200
    assert gemini_usage.output_tokens == 100
    assert gemini_usage.total_tokens == 300

    gpt4_usage = next(u for u in result_sorted if u.model == "gpt-4")
    assert gpt4_usage.input_tokens == 150
    assert gpt4_usage.output_tokens == 75
    assert gpt4_usage.total_tokens == 225


def test_compound_usages_empty_list():
    """Test compound_usages with an empty list."""
    result = compound_usages([])
    assert result == []
