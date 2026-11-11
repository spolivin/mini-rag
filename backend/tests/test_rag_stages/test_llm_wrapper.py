"""Unit tests for LLMWrapper with a fake client: no network, no API key, no database."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import anthropic
import pytest

from rag.stages import LLMUnavailableError, LLMWrapper

PARAMS = {"max_tokens": 1000, "effort": "low", "temperature": 0.7}


def text(value):
    return SimpleNamespace(type="text", text=value)


def thinking():
    return SimpleNamespace(type="thinking", thinking="")


def make_wrapper(blocks=None, stop_reason="end_turn", error=None):
    client = MagicMock()
    if error is not None:
        client.messages.create.side_effect = error
    else:
        client.messages.create.return_value = SimpleNamespace(
            content=blocks or [], stop_reason=stop_reason
        )
    return LLMWrapper("claude-haiku-5-5", PARAMS, client=client), client


def test_sends_the_expected_request():
    wrapper, client = make_wrapper([text("An answer.")])
    assert wrapper.generate("What?", ["best", "worst"]) == "An answer."

    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-haiku-5-5"
    assert kwargs["max_tokens"] == 1000
    assert kwargs["output_config"] == {"effort": "low"}
    assert kwargs["system"] == LLMWrapper.SYSTEM_PROMPT
    # Sampling parameters are rejected by current Claude models: none may be sent
    assert not {"temperature", "top_p", "top_k"} & set(kwargs)
    # Context is reversed so the most relevant chunk sits closest to the question
    assert kwargs["messages"] == [
        {"role": "user", "content": "Context:\nworst\n\nbest\n\nQuestion:\nWhat?"}
    ]


def test_only_text_blocks_are_returned():
    wrapper, _ = make_wrapper([thinking(), text("  First. "), text("Second.  ")])
    assert wrapper.generate("What?", ["ctx"]) == "First. Second."


def test_refusal_returns_a_message():
    wrapper, _ = make_wrapper([], stop_reason="refusal")
    assert "declined" in wrapper.generate("What?", ["ctx"])


def test_empty_answer_returns_a_message():
    wrapper, _ = make_wrapper([thinking()], stop_reason="max_tokens")
    assert wrapper.generate("What?", ["ctx"]) == "The model returned an empty answer."


def test_api_errors_become_llm_unavailable():
    wrapper, _ = make_wrapper(error=anthropic.APIError("boom", None, body=None))
    with pytest.raises(LLMUnavailableError, match="boom"):
        wrapper.generate("What?", ["ctx"])


def test_missing_credentials_become_llm_unavailable():
    error = TypeError('"Could not resolve authentication method. Expected one of api_key..."')
    wrapper, _ = make_wrapper(error=error)
    with pytest.raises(LLMUnavailableError, match="ANTHROPIC_API_KEY"):
        wrapper.generate("What?", ["ctx"])


def test_other_type_errors_are_not_swallowed():
    wrapper, _ = make_wrapper(error=TypeError("a real bug"))
    with pytest.raises(TypeError, match="a real bug"):
        wrapper.generate("What?", ["ctx"])
