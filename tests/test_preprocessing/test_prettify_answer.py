import pytest

from mini_rag.preprocessing import prettify_answer


@pytest.mark.parametrize(
    "input,output",
    [
        (
            "this is a test. it should be capitalized.",
            "This is a test. It should be capitalized.",
        ),
        (
            "This is a test , with extra       space .",
            "This is a test, with extra space.",
        ),
        ("", ""),
        ("This is already perfect.", "This is already perfect."),
    ],
)
def test_prettify_answer(input, output):
    """Tests the summary prettification function."""
    prettified = prettify_answer(input)
    assert prettified == output
