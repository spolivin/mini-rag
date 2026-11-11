import pytest

from rag.config import TextGenerationConfig


def test_text_generation_config():
    gen_params = TextGenerationConfig()
    assert gen_params.max_tokens == 16000
    assert gen_params.effort == "low"


def test_invalid_effort():
    with pytest.raises(ValueError):
        TextGenerationConfig(effort="extreme")


def test_invalid_max_tokens():
    with pytest.raises(ValueError):
        TextGenerationConfig(max_tokens=0)
